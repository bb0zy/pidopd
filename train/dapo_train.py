r"""Full-parameter DAPO for Qwen3 on one or more GPUs (Linux/CUDA).

    CUDA_VISIBLE_DEVICES=0 python -m train.dapo_train
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
        --standalone --nproc_per_node=2 --module train.dapo_train

Default: non-thinking, 10240 new tokens, checkpoints every 100 optimizer steps
in train/dapo_checkpoint. Resume with --resume-from-checkpoint latest.

Implements clip-higher, token-normalized loss, dynamic sampling and soft overlong
penalties. Based on https://arxiv.org/abs/2503.14476 and TRL 1.13.0.
Uses this project's 0/1 accuracy reward, G=8 and small GPU-budget batch sizes;
this is not a reproduction of the paper's 32B training scale.
"""

import argparse
from pathlib import Path
import random
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.rl_train import accuracy_reward, run_training


def make_overlong_reward(max_length, buffer_length, penalty):
    """Linear penalty on the final buffer_length tokens, using actual rollout IDs.

    0/1 correctness is half the scale of the paper's -1/+1 reward, so the default
    maximum penalty is also halved (0.5). Truncated trajectories are loss-masked
    by TRL, but their shaped rewards still contribute to the group's baseline.
    """
    def soft_overlong_reward(completion_ids, **kwargs):
        start = max_length - buffer_length
        return [-penalty * min(1.0, max(0.0, (len(ids) - start) / buffer_length))
                for ids in completion_ids]

    return soft_overlong_reward


def mixed_accuracy_group_starts(inputs, scores, completion_ids, group_size, stop_ids):
    """Keep whole, mixed-correctness groups with at least one unmasked response."""
    if len(inputs) % group_size or len(scores) != len(inputs) or len(completion_ids) != len(inputs):
        raise ValueError("Dynamic sampling requires aligned, complete local question groups")
    starts = []
    for start in range(0, len(inputs), group_size):
        group = inputs[start:start + group_size]
        if any(row["prompt"] != group[0]["prompt"] or row["answer"] != group[0]["answer"] for row in group):
            raise ValueError("A generation group contains different questions or reference answers")
        rewards = scores[start:start + group_size]
        terminated = any(ids and ids[-1] in stop_ids for ids in completion_ids[start:start + group_size])
        if min(rewards) < max(rewards) and terminated:
            starts.append(start)
    return starts


class DynamicSamplingMixin:
    """Text-only, colocate TP=1 adapter for the pinned TRL 1.13.0 GRPOTrainer.

    Filter before training-model forwards. Keep the actual sampled token IDs AND
    logprobs; replay through TRL's rollout_func hook without generating again.
    Each rank fills its local batch with complete groups. All ranks participate
    in every sampling round (even already-full ranks) because vLLM weight sync
    can contain collectives. Only scalar statistics cross ranks.
    """

    def __init__(self, *args, max_sampling_rounds=16, **kwargs):
        self.max_sampling_rounds = max_sampling_rounds
        super().__init__(*args, **kwargs)

    def _sampling_totals(self, values):
        import torch

        local = torch.tensor([values], dtype=torch.float64, device=self.accelerator.device)
        return self.accelerator.gather(local).sum(dim=0).tolist()

    def _refill_inputs(self, size, rng):
        rows = []
        for _ in range(size // self.num_generations):
            example = self.train_dataset[rng.randrange(len(self.train_dataset))]
            rows.extend(dict(example) for _ in range(self.num_generations))
        return rows

    def _generate_and_score_completions(self, inputs):
        if not self.model.training:
            return super()._generate_and_score_completions(inputs)
        if self.rollout_func is not None:
            raise ValueError("DAPO dynamic sampling owns rollout_func; custom rollouts are unsupported")
        target = len(inputs)
        group_size = self.num_generations
        if not target or target % group_size:
            raise ValueError("Per-rank generation batch must be divisible by num_generations")
        # No private sampler state to lose on resume: refill choices depend on
        # restored global_step and rank. vLLM sampling itself is not bit-exact on resume.
        rng = random.Random(f"{self.args.seed}:{self.state.global_step}:{self.accelerator.process_index}")
        accepted = []
        cached = {"prompt_ids": [], "completion_ids": [], "logprobs": []}
        candidates = inputs
        correct = generated = eligible = tokens = clipped = 0
        stop_ids = {self._tokenizer.eos_token_id, self._tokenizer.pad_token_id}
        for sampling_round in range(1, self.max_sampling_rounds + 1):
            prompt_ids, images, fields = self._tokenize_prompts([row["prompt"] for row in candidates])
            if images is not None:
                raise ValueError("This DAPO entry point supports text-only math datasets")
            completion_ids, logprobs = self._generate_single_turn(prompt_ids, images, fields)
            if logprobs is None or len(logprobs) != len(candidates):
                raise RuntimeError("vLLM must return sampled-token logprobs for importance correction")
            texts = self._tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            scores = accuracy_reward(texts, [row["answer"] for row in candidates])
            starts = mixed_accuracy_group_starts(candidates, scores, completion_ids, group_size, stop_ids)
            correct += sum(scores)
            generated += len(candidates)
            eligible += len(starts)
            tokens += sum(map(len, completion_ids))
            clipped += sum(not ids or ids[-1] not in stop_ids for ids in completion_ids)
            for start in starts:
                if len(accepted) == target:
                    break
                end = start + group_size
                accepted.extend(dict(row) for row in candidates[start:end])
                cached["prompt_ids"].extend(prompt_ids[start:end])
                cached["completion_ids"].extend(completion_ids[start:end])
                cached["logprobs"].extend(logprobs[start:end])
            totals = self._sampling_totals([
                int(len(accepted) == target), correct, generated, eligible, tokens, clipped,
            ])
            if totals[0] == self.accelerator.num_processes:
                break
            # Full ranks still generate one group so weight synchronization stays
            # collective. Rejected groups are replaced with NEW dataset draws.
            candidates = self._refill_inputs(max(group_size, target - len(accepted)), rng)
        else:
            raise RuntimeError(
                f"DAPO could not fill every rank's batch after {self.max_sampling_rounds} sampling rounds "
                f"({int(totals[0])}/{self.accelerator.num_processes} ranks ready). "
                "Inspect rewards/truncation or raise --max-sampling-rounds; no partial optimizer step was taken."
            )

        metrics = self._metrics["train"]
        metrics["dapo/sampling_rounds"].append(sampling_round)
        metrics["dapo/raw_accuracy"].append(totals[1] / totals[2])
        metrics["dapo/group_acceptance_rate"].append(totals[3] / (totals[2] / group_size))
        metrics["dapo/generated_completions"].append(totals[2])
        metrics["dapo/generated_tokens"].append(totals[4])
        metrics["dapo/raw_clipped_ratio"].append(totals[5] / totals[2])

        def replay_rollout(prompts, trainer):
            if prompts != [row["prompt"] for row in accepted]:
                raise RuntimeError("Cached DAPO rollout no longer matches the selected prompts")
            return cached

        self.rollout_func = replay_rollout
        try:
            # TRL handles padding, global token denominator, shaped advantages,
            # vLLM importance correction, truncation masks and batch splitting.
            return super()._generate_and_score_completions(accepted)
        finally:
            self.rollout_func = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", default="/models/qwen3-1.7b")
    parser.add_argument("--data-path", type=Path, default=PROJECT_ROOT / "dataset/train/dapo-17k.parquet")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "train/dapo_checkpoint")
    parser.add_argument("--max-steps", type=int, default=1000, help="Optimizer updates, not sampling rounds")
    parser.add_argument("--per-device-train-batch-size", type=int, default=2,
                        help="Conservative microbatch for 10240-token full-parameter training")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.0, help="Reference KL coefficient; DAPO default is zero")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon-high", type=float, default=0.28)
    parser.add_argument("--max-prompt-length", type=int, default=1024, help="Filter longer prompts without truncation")
    parser.add_argument("--max-completion-length", "--max-new-tokens", type=int, default=10240)
    parser.add_argument("--overlong-buffer-length", type=int, default=2048,
                        help="Default penalty starts after 8192 generated tokens")
    parser.add_argument("--overlong-penalty", type=float, default=0.5, help="Maximum penalty on the 0/1 reward scale")
    parser.add_argument("--max-sampling-rounds", type=int, default=16,
                        help="Fail if dynamic sampling cannot fill every rank within this many rounds")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--vllm-enable-sleep-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from-checkpoint", nargs="?", const="latest", default=None)
    parser.add_argument("--report-to", choices=("none", "wandb", "tensorboard"), default="none")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    for name in ("max_steps", "per_device_train_batch_size", "gradient_accumulation_steps", "num_generations",
                 "max_prompt_length", "max_completion_length", "overlong_buffer_length", "max_sampling_rounds"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_generations < 2:
        parser.error("DAPO needs at least two generations per question")
    if args.per_device_train_batch_size * args.gradient_accumulation_steps % args.num_generations:
        parser.error("Per-device batch size x accumulation must be divisible by num-generations")
    if args.overlong_buffer_length > args.max_completion_length:
        parser.error("overlong-buffer-length cannot exceed max-completion-length")
    if not 0 <= args.overlong_penalty < 1:
        parser.error("overlong-penalty must be in [0, 1) to preserve correctness ordering")
    if not 0 < args.epsilon < 1 or args.epsilon_high < args.epsilon:
        parser.error("Require 0 < epsilon < 1 and epsilon-high >= epsilon")
    if not 0 < args.vllm_gpu_memory_utilization < 1:
        parser.error("vllm-gpu-memory-utilization must be between 0 and 1")
    if args.learning_rate <= 0 or args.temperature <= 0 or args.beta < 0:
        parser.error("learning-rate/temperature must be positive and beta nonnegative")
    return args


def main():
    cli = parse_args()
    run_training(cli, trainer_mixin=DynamicSamplingMixin, reward_funcs=[
        accuracy_reward,
        make_overlong_reward(cli.max_completion_length, cli.overlong_buffer_length, cli.overlong_penalty),
    ])


if __name__ == "__main__":
    main()
