r"""Qwen3-1.7B full-parameter GRPO on one or more GPUs (Linux/CUDA).

From the project root, using the environment in requirements.txt and its custom
vLLM 0.19.1 build (do not start vllm/start_vllm.sh for this script):

    CUDA_VISIBLE_DEVICES=0 python -m train.grpo_train

    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
        -m train.grpo_train

Resume with the same command plus --resume-from-checkpoint latest.
Checkpoints: train/grpo_checkpoint/checkpoint-{100,200,...}/; final model: final/.
Checkpoints contain full model/tokenizer AND optimizer/scheduler/RNG state.

Configuration basis:
https://arxiv.org/html/2402.03300v3#S4.SS2
https://huggingface.co/docs/trl/v1.13.0/en/grpo_trainer
DeepSeekMath uses lr=1e-6, beta=0.04, G=64, batch=1024 and one update per rollout.
Here G=8 and lr=1e-6. The effective batch is 64 completions per GPU (128 with
two GPUs). At 1000 optimizer steps this is 8000 prompt draws per GPU (~0.92
dataset pass with two GPUs before filtering). This is a starting point, not a paper's
universal step count. The default loss is explicitly classic GRPO; --loss-type
dapo selects TRL's token-normalized variant without changing the dataset.

A100 80GB starting point: microbatch=4, accumulation=16, vLLM budget=0.5,
sleep mode enabled. This uses more training/inference memory while preserving
64 completions per GPU per optimizer update. Peak memory depends on response
lengths; this is not a guaranteed 70GB cap or a measured memory prediction.
"""

import argparse
from pathlib import Path
import sys

# Keep both `python -m train.grpo_train` and direct script execution working.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.rl_train import PROJECT_ROOT, run_training


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", default="/models/qwen3-1.7b")
    parser.add_argument("--data-path", type=Path, default=PROJECT_ROOT / "dataset/train/dapo-17k.parquet")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "train/grpo_checkpoint")
    parser.add_argument("--max-steps", type=int, default=1000, help="Optimizer updates, not microbatches")
    parser.add_argument("--per-device-train-batch-size", type=int, default=4,
                        help="Training microbatch per GPU; lower if long responses cause OOM")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16,
                        help="4 x 16 preserves 64 completions per GPU per optimizer update")
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--max-prompt-length", type=int, default=1024, help="Filter longer prompts; never truncate questions")
    parser.add_argument("--max-completion-length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--loss-type", choices=("grpo", "dapo", "dr_grpo"), default="grpo")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False,
                        help="Default off to fit a 4096-token budget; step-by-step solving is still requested")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5,
                        help="vLLM-only GPU budget, not a cap on total GRPO memory")
    parser.add_argument("--vllm-enable-sleep-mode", action=argparse.BooleanOptionalAction, default=True,
                        help="Release vLLM weights/cache during full-parameter optimization")
    parser.add_argument("--resume-from-checkpoint", nargs="?", const="latest", default=None,
                        help="Checkpoint path, or latest in output-dir")
    parser.add_argument("--report-to", choices=("none", "wandb", "tensorboard"), default="none")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    for name in ("max_steps", "per_device_train_batch_size", "gradient_accumulation_steps",
                 "num_generations", "max_prompt_length", "max_completion_length"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_generations < 2:
        parser.error("GRPO needs at least two generations per question")
    if not 0 < args.vllm_gpu_memory_utilization < 1:
        parser.error("--vllm-gpu-memory-utilization must be between 0 and 1")
    if args.learning_rate <= 0 or args.temperature <= 0 or args.beta < 0:
        parser.error("learning-rate/temperature must be positive and beta must be nonnegative")
    return args


def main():
    run_training(parse_args())


if __name__ == "__main__":
    main()
