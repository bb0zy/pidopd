r"""Qwen3-1.7B LoRA GRPO on two A100 80GB GPUs (Linux/CUDA).

From the project root, using the environment in requirements.txt and its custom
vLLM 0.19.1 build (do not start vllm/start_vllm.sh for this script):

    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
        -m train.grpo_train

Resume with the same command plus --resume-from-checkpoint latest.
Checkpoints: train/grpo_checkpoint/checkpoint-{50,100,...}/; final adapter: final/.
Checkpoints contain LoRA/tokenizer AND optimizer/scheduler/RNG state for resume;
they do not contain a second copy of the frozen base-model weights.

Configuration basis:
https://arxiv.org/html/2402.03300v3#S4.SS2
https://huggingface.co/docs/trl/v1.13.0/en/grpo_trainer
DeepSeekMath uses lr=1e-6, beta=0.04, G=64, batch=1024 and one update per rollout.
Here G=8 and batch=128 are scaled to two GPUs; lr=5e-6 is a LoRA starting point.
1000 optimizer steps x (128/8) prompts = 16000 prompt draws (~0.92 dataset pass
before length filtering). This is a budget-based starting point, not a paper's
universal step count. The default loss is explicitly classic GRPO; --loss-type
dapo selects TRL's token-normalized variant without changing the dataset.
"""

import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = (
    "Solve the following math problem step by step.\n"
    r"The last line of your response should contain only the final answer enclosed in \boxed{}."
)


def prepare_example(example):
    """Use the clean question, not source_prompt's conflicting Answer: format."""
    question = example.get("prompt")
    reward_model = example.get("reward_model")
    gold = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Every DAPO row must contain a nonempty string prompt")
    if gold is None or not str(gold).strip():
        raise ValueError("Every DAPO row must contain reward_model.ground_truth")
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question.strip()},
        ],
        "answer": str(gold).strip(),
    }


def extract_final_answer(text):
    """Read the last balanced box outside thinking; reject unfinished/trailing prose."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        return None
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                # Accept common closing math delimiters and sentence punctuation.
                tail = text[i + 1:].strip()
                tail = tail.replace(r"\]", "").replace(r"\)", "")
                if tail.strip("$ .。\t\r\n"):
                    return None
                return text[start:i].strip() or None
    return None


@lru_cache(maxsize=4096)
def get_answer_grader(answer):
    # Reuse exactly the project's evaluation semantics (including math equivalence).
    # Also support direct `python train/grpo_train.py` entry, not only `python -m`.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from eval.answer_grader import AnswerGrader

    return AnswerGrader(answer)


def accuracy_reward(completions, answer, **kwargs):
    """Binary correctness only; never parse numbers from the reasoning transcript."""
    rewards = []
    for completion, gold in zip(completions, answer, strict=True):
        text = completion if isinstance(completion, str) else completion[-1]["content"]
        prediction = extract_final_answer(text)
        rewards.append(float(get_answer_grader(gold).grade(prediction)["correct"]))
    return rewards


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", default="/models/qwen3-1.7b")
    parser.add_argument("--data-path", type=Path, default=PROJECT_ROOT / "dataset/train/dapo-17k.parquet")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "train/grpo_checkpoint")
    parser.add_argument("--max-steps", type=int, default=1000, help="Optimizer updates, not microbatches")
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--max-prompt-length", type=int, default=1024, help="Filter longer prompts; never truncate questions")
    parser.add_argument("--max-completion-length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--loss-type", choices=("grpo", "dapo", "dr_grpo"), default="grpo")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False,
                        help="Default off to fit a 4096-token budget; step-by-step solving is still requested")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--vllm-enable-sleep-mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume-from-checkpoint", nargs="?", const="latest", default=None,
                        help="Checkpoint path, or latest in output-dir")
    parser.add_argument("--report-to", choices=("none", "wandb", "tensorboard"), default="none")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    for name in ("max_steps", "per_device_train_batch_size", "gradient_accumulation_steps",
                 "num_generations", "max_prompt_length", "max_completion_length", "lora_rank", "lora_alpha"):
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
    cli = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOConfig, GRPOTrainer

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2 or not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Use CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m train.grpo_train")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    effective_batch = cli.per_device_train_batch_size * world_size * cli.gradient_accumulation_steps
    if effective_batch % cli.num_generations:
        raise ValueError("Effective batch size must be divisible by --num-generations")
    model_path = Path(cli.model_path).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local model directory not found: {model_path}")
    if not cli.data_path.is_file():
        raise FileNotFoundError(cli.data_path)
    output_dir = cli.output_dir.resolve()
    last_checkpoint = get_last_checkpoint(str(output_dir)) if output_dir.is_dir() else None
    resume = cli.resume_from_checkpoint
    if resume == "latest":
        if last_checkpoint is None:
            raise FileNotFoundError(f"No checkpoint-* directory in {output_dir}")
        resume = last_checkpoint
    elif resume is not None:
        resume = str(Path(resume).expanduser().resolve())
        if not Path(resume, "trainer_state.json").is_file():
            raise ValueError("Resume requires a training checkpoint containing trainer_state.json")
    elif last_checkpoint is not None or (output_dir / "final").exists():
        raise FileExistsError("Output already contains a run; use --resume-from-checkpoint latest or a new --output-dir")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    chat_kwargs = {"enable_thinking": cli.enable_thinking}
    raw_dataset = Dataset.from_parquet(str(cli.data_path.resolve()), cache_dir=str(PROJECT_ROOT / "tmp/datasets"))
    train_dataset = raw_dataset.map(prepare_example, remove_columns=raw_dataset.column_names,
                                    desc="Prepare DAPO prompts and reference answers")

    def within_context(example):
        # TRL 1.13 has no max_prompt_length; account for the actual chat template.
        ids = tokenizer.apply_chat_template(example["prompt"], tokenize=True,
                                            add_generation_prompt=True, **chat_kwargs)
        return len(ids) <= cli.max_prompt_length

    train_dataset = train_dataset.filter(within_context, desc="Filter overlong prompts without truncation")
    if len(train_dataset) < effective_batch // cli.num_generations:
        raise ValueError("Not enough usable questions for one generation batch")
    # Fail before allocating the model if any reference cannot be graded.
    for gold in set(train_dataset["answer"]):
        get_answer_grader(gold)

    args = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=cli.max_steps,
        per_device_train_batch_size=cli.per_device_train_batch_size,
        gradient_accumulation_steps=cli.gradient_accumulation_steps,
        # One rollout batch per optimizer update, split into small training batches.
        steps_per_generation=cli.gradient_accumulation_steps,
        num_generations=cli.num_generations,
        num_iterations=1,
        learning_rate=cli.learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=min(20, max(0, cli.max_steps - 1)),
        optim="adamw_torch_fused",
        weight_decay=0.0,
        max_grad_norm=1.0,
        beta=cli.beta,
        epsilon=0.2,
        loss_type=cli.loss_type,
        scale_rewards="group",
        max_completion_length=cli.max_completion_length,
        temperature=cli.temperature,
        top_p=1.0,
        top_k=0,
        chat_template_kwargs=chat_kwargs,
        mask_truncated_completions=True,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=cli.vllm_gpu_memory_utilization,
        vllm_max_model_length=cli.max_prompt_length + cli.max_completion_length,
        vllm_enable_sleep_mode=cli.vllm_enable_sleep_mode,
        vllm_importance_sampling_correction=True,
        vllm_importance_sampling_mode="token_truncate",
        vllm_importance_sampling_clip_max=2.0,
        bf16=True,
        fp16=False,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        disable_dropout=True,
        remove_unused_columns=False,
        eval_strategy="no",
        report_to=cli.report_to,
        run_name="qwen3-1.7b-dapo-grpo-lora",
        logging_steps=1,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=None,
        save_only_model=False,
        seed=cli.seed,
        data_seed=cli.seed,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa",  # No separate flash-attn installation needed.
    )
    model.config.use_cache = False
    lora_config = LoraConfig(task_type="CAUSAL_LM", r=cli.lora_rank,
                             lora_alpha=cli.lora_alpha, lora_dropout=0.0,
                             target_modules="all-linear", bias="none")
    trainer = GRPOTrainer(model=model, args=args, train_dataset=train_dataset,
                          reward_funcs=accuracy_reward, processing_class=tokenizer,
                          peft_config=lora_config)
    if trainer.is_world_process_zero():
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {**vars(cli), "world_size": world_size,
                    "raw_questions": len(raw_dataset), "usable_questions": len(train_dataset),
                    "completions_per_update": effective_batch,
                    "questions_per_update": effective_batch // cli.num_generations}
        (output_dir / "grpo_run_config.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        print(f"Questions: {len(train_dataset)}/{len(raw_dataset)}; {effective_batch} completions/update; "
              f"{effective_batch // cli.num_generations} questions/update; {cli.max_steps} optimizer steps", flush=True)
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output_dir / "final"))
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
