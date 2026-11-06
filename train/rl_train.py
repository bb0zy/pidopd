"""Shared GRPO/DAPO runtime, mathematical rewards and full-model checkpoint helpers."""

from functools import lru_cache
import json
import os
from pathlib import Path
import re

from .kd_data import PROJECT_ROOT, SYSTEM_PROMPT


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


def configure_launch():
    """Select one GPU for plain Python; respect torchrun's per-process local rank.

    Call before importing Torch so a non-distributed launch cannot accidentally
    activate Transformers DataParallel over every visible GPU.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or local_rank < 0:
        raise ValueError("WORLD_SIZE must be positive and LOCAL_RANK must be nonnegative")
    if world_size > 1 and "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Multi-GPU training requires torchrun --nproc_per_node=<GPU count>")
    if world_size == 1:
        if local_rank != 0:
            raise ValueError("A single-process launch must use LOCAL_RANK=0")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        os.environ["CUDA_VISIBLE_DEVICES"] = visible.split(",")[0].strip()
    return world_size, local_rank


def validate_full_checkpoint(checkpoint):
    """Check training state and model weights before allocating GPU memory."""
    checkpoint = Path(checkpoint)
    weights = ("model.safetensors", "model.safetensors.index.json",
               "pytorch_model.bin", "pytorch_model.bin.index.json")
    if not (checkpoint / "trainer_state.json").is_file() or not any(
        (checkpoint / name).is_file() for name in weights
    ):
        raise ValueError("Resume requires a full-model training checkpoint with weights and trainer_state.json")


def resolve_full_resume(output_dir, resume):
    """Select a full-model checkpoint and reject accidental overwrites before model allocation."""
    from transformers.trainer_utils import get_last_checkpoint

    output_dir = Path(output_dir)
    latest = get_last_checkpoint(str(output_dir)) if output_dir.is_dir() else None
    if resume == "latest":
        if latest is None:
            raise FileNotFoundError(f"No checkpoint-* directory in {output_dir}")
        resume = latest
    elif resume is not None:
        resume = str(Path(resume).expanduser().resolve())
    elif latest is not None or (output_dir / "final").exists():
        raise FileExistsError("Output already contains a run; use --resume-from-checkpoint latest or a new --output-dir")
    if resume is not None:
        validate_full_checkpoint(resume)
    return resume


def run_training(cli, *, trainer_mixin=None, reward_funcs=None):
    """Run the shared full-model RL pipeline; DAPO supplies dynamic sampling and rewards."""
    is_dapo = trainer_mixin is not None
    algorithm = "DAPO" if is_dapo else "GRPO"
    world_size, local_rank = configure_launch()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    from importlib.metadata import version

    if is_dapo and version("trl") != "1.13.0":
        raise RuntimeError("DAPO dynamic sampling requires the project's pinned trl==1.13.0")
    trainer_class = type("DAPOTrainer", (trainer_mixin, GRPOTrainer), {}) if is_dapo else GRPOTrainer

    if not torch.cuda.is_available():
        raise RuntimeError(f"{algorithm} with vLLM requires an NVIDIA CUDA GPU")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(f"LOCAL_RANK={local_rank} has no visible CUDA device; check CUDA_VISIBLE_DEVICES and torchrun")
    torch.cuda.set_device(local_rank)
    effective_batch = cli.per_device_train_batch_size * world_size * cli.gradient_accumulation_steps
    if effective_batch % cli.num_generations:
        raise ValueError("Effective batch size must be divisible by --num-generations")
    model_path = Path(cli.model_path).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local model directory not found: {model_path}")
    if not cli.data_path.is_file():
        raise FileNotFoundError(cli.data_path)
    output_dir = cli.output_dir.resolve()
    resume = resolve_full_resume(output_dir, cli.resume_from_checkpoint)

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
        ids = ids if isinstance(ids, list) else ids["input_ids"]
        return len(ids) <= cli.max_prompt_length

    train_dataset = train_dataset.filter(within_context, desc="Filter overlong prompts without truncation")
    if len(train_dataset) < effective_batch // cli.num_generations:
        raise ValueError("Not enough usable questions for one generation batch")
    # Fail before allocating the model if any reference cannot be graded.
    for gold in set(train_dataset["answer"]):
        get_answer_grader(gold)

    loss_options = {"epsilon": 0.2, "loss_type": cli.loss_type} if not is_dapo else {
        "epsilon": cli.epsilon, "epsilon_high": cli.epsilon_high, "loss_type": "dapo",
    }
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
        **loss_options,
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
        run_name="qwen3-1.7b-dapo-full" if is_dapo else "qwen3-1.7b-dapo-grpo-full",
        logging_steps=1,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=100,
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
    model.requires_grad_(True)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainer_kwargs = {"max_sampling_rounds": cli.max_sampling_rounds} if is_dapo else {}
    trainer = trainer_class(model=model, args=args, train_dataset=train_dataset,
                            reward_funcs=accuracy_reward if reward_funcs is None else reward_funcs,
                            processing_class=tokenizer, **trainer_kwargs)
    if trainer.is_world_process_zero():
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {**vars(cli), "world_size": world_size, "finetuning_type": "full",
                    "trainable_parameters": total_parameters,
                    "raw_questions": len(raw_dataset), "usable_questions": len(train_dataset),
                    "completions_per_update": effective_batch,
                    "questions_per_update": effective_batch // cli.num_generations}
        if is_dapo:
            metadata.update(algorithm="dapo", loss_type="dapo", trl_version=version("trl"))
        config_name = "dapo_run_config.json" if is_dapo else "grpo_run_config.json"
        (output_dir / config_name).write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        question_unit = "accepted question groups" if is_dapo else "questions"
        print(f"Full-parameter {algorithm}: {total_parameters:,} parameters on {world_size} GPU(s); "
              f"questions: {len(train_dataset)}/{len(raw_dataset)}; {effective_batch} completions/update; "
              f"{effective_batch // cli.num_generations} {question_unit}/update; {cli.max_steps} optimizer steps", flush=True)
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output_dir / "final"))
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
