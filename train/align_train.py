r"""Stage 2: offline forward-KL distillation of cached teacher responses.

    CUDA_VISIBLE_DEVICES=0 python -m train.align_train
    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m train.align_train

Student: /models/qwen3-1.7b; frozen teacher: /models/qwen3-4b-1.
Every 100 optimizer updates save a full student checkpoint in train/align_checkpoint.
Resume with --resume-from-checkpoint latest; final student is saved in final/.

Run python -m train.teacher_rollout first. This stage NEVER generates responses.
TRL 1.13 GKD: lmbda=0, seq_kd=False, beta=0, temperature=1.
The student matches the frozen teacher's full-vocabulary distribution at every
cached response token, including reasoning and EOS. Prompts/padding are masked.
Length-limited and overlong responses are filtered, never cropped. Liger's fused
GKD loss bounds logits memory for long responses; no vLLM engine is loaded here.

Parameter basis: https://huggingface.co/docs/trl/v1.13.0/en/gkd_trainer
GKD's default learning rate is 2e-5. Here 1e-5 is a conservative starting point
for full-parameter alignment of an already post-trained Qwen3 on 17k examples.
Use one epoch, effective batch 32 per GPU, linear decay, and 3% warmup. These
are project starting values, not an experimentally established optimum.
"""

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

from .kd_data import (DEFAULT_ROLLOUT_PATH, PROJECT_ROOT, alignment_example, file_sha256,
                      metadata_path, padded_batch, tokenizer_fingerprint, usable_rollout)


@dataclass
class OfflineKDCollator:
    pad_token_id: int

    def __call__(self, features):
        import torch

        return {name: torch.tensor(values, dtype=torch.long)
                for name, values in padded_batch(features, self.pad_token_id).items()}


def validate_rollout_metadata(data_path, teacher_path, tokenizer):
    metadata = json.loads(metadata_path(data_path).read_text(encoding="utf-8"))
    signature = metadata["signature"]
    if signature["format_version"] != 1 or not metadata["complete"]:
        raise ValueError("Run stage 1 to completion before alignment (teacher_rollout --resume if interrupted)")
    if signature["tokenizer_fingerprint"] != tokenizer_fingerprint(tokenizer):
        raise ValueError("Cached rollout vocabulary/special token IDs do not match the training tokenizer")
    if signature["teacher_config_sha256"] != file_sha256(teacher_path / "config.json"):
        raise ValueError("Cached rollout teacher config differs from --teacher-model-path")
    return metadata


def validate_tokenizers(student, teacher):
    if student.get_vocab() != teacher.get_vocab():
        raise ValueError("Forward-KL KD requires identical token-to-ID mappings for student and teacher")
    for name in ("bos_token_id", "eos_token_id"):
        if getattr(student, name) != getattr(teacher, name):
            raise ValueError(f"Student and teacher must share {name}")
    if student.eos_token_id is None or not student.chat_template:
        raise ValueError("The student tokenizer must have an EOS token and a chat template")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student-model-path", default="/models/qwen3-1.7b")
    parser.add_argument("--teacher-model-path", default="/models/qwen3-4b-1")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_ROLLOUT_PATH, help="Stage 1 teacher rollout JSONL")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "train/align_checkpoint")
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="Positive value overrides epoch count")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-length", type=int, default=32768, help="Filter longer full sequences; never crop responses")
    parser.add_argument("--use-liger-kernel", action=argparse.BooleanOptionalAction, default=True,
                        help="Fused full-vocabulary GKD loss; disabling may use much more GPU memory")
    parser.add_argument("--resume-from-checkpoint", nargs="?", const="latest", default=None)
    parser.add_argument("--report-to", choices=("none", "wandb", "tensorboard"), default="none")
    parser.add_argument("--seed", type=int, default=42)
    cli = parser.parse_args(argv)
    if cli.max_steps != -1 and cli.max_steps < 1:
        parser.error("--max-steps must be -1 or positive")
    for name in ("per_device_train_batch_size", "gradient_accumulation_steps", "max_length"):
        if getattr(cli, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(cli.learning_rate) or cli.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if not math.isfinite(cli.num_train_epochs) or cli.num_train_epochs <= 0:
        parser.error("--num-train-epochs must be finite and positive")
    if not 0 <= cli.warmup_ratio < 1:
        parser.error("--warmup-ratio must be in [0, 1)")
    return cli


def main():
    cli = parse_args()
    # These lightweight shared helpers do not import Torch/TRL or start GRPO.
    from .grpo_train import configure_launch, validate_full_checkpoint

    world_size, local_rank = configure_launch()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from trl.experimental.gkd import GKDConfig
    from trl.import_utils import is_liger_kernel_available
    from .offline_gkd import OfflineGKDTrainer

    if cli.use_liger_kernel and not is_liger_kernel_available():
        raise ImportError("Long-sequence alignment requires liger-kernel>=0.8.2; install project requirements")

    if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
        raise RuntimeError("KD training requires a visible CUDA GPU for this process")
    torch.cuda.set_device(local_rank)
    set_seed(cli.seed)
    student_path = Path(cli.student_model_path).expanduser().resolve()
    teacher_path = Path(cli.teacher_model_path).expanduser().resolve()
    for path in (student_path, teacher_path):
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"Local model config not found: {path / 'config.json'}")
    if not cli.data_path.is_file():
        raise FileNotFoundError(cli.data_path)
    output_dir = cli.output_dir.expanduser().resolve()
    last = get_last_checkpoint(str(output_dir)) if output_dir.is_dir() else None
    resume = cli.resume_from_checkpoint
    if resume == "latest":
        if last is None:
            raise FileNotFoundError(f"No checkpoint-* directory in {output_dir}")
        resume = last
    elif resume is not None:
        resume = str(Path(resume).expanduser().resolve())
    elif last is not None or (output_dir / "final").exists():
        raise FileExistsError("Output contains a run; use --resume-from-checkpoint latest or a new --output-dir")
    if resume is not None:
        validate_full_checkpoint(resume)

    tokenizer = AutoTokenizer.from_pretrained(str(student_path), local_files_only=True, padding_side="right")
    teacher_tokenizer = AutoTokenizer.from_pretrained(str(teacher_path), local_files_only=True)
    validate_tokenizers(tokenizer, teacher_tokenizer)
    del teacher_tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rollout_metadata = validate_rollout_metadata(cli.data_path, teacher_path, tokenizer)
    raw = Dataset.from_json(str(cli.data_path.resolve()), cache_dir=str(PROJECT_ROOT / "tmp/datasets"))
    if len(raw) != rollout_metadata["expected_responses"]:
        raise ValueError("Cached rollout count differs from stage 1 metadata")
    required = {"prompt_token_ids", "completion_token_ids", "finish_reason"}
    if not required.issubset(raw.column_names):
        raise ValueError("--data-path must point to stage 1 teacher rollout JSONL, not raw DAPO")
    accepted = raw.filter(usable_rollout, fn_kwargs={"max_length": cli.max_length},
                          desc="Keep complete teacher responses without truncation")
    train_dataset = accepted.map(alignment_example, remove_columns=accepted.column_names,
                                 desc="Use exact cached teacher tokens for full-response KL")
    if not len(train_dataset):
        raise ValueError("No training examples remain after length filtering")
    effective_batch = cli.per_device_train_batch_size * world_size * cli.gradient_accumulation_steps
    updates_per_epoch = math.ceil(math.ceil(len(train_dataset) / world_size)
                                  / cli.per_device_train_batch_size / cli.gradient_accumulation_steps)
    total_steps = cli.max_steps if cli.max_steps > 0 else math.ceil(updates_per_epoch * cli.num_train_epochs)
    warmup_steps = min(max(total_steps - 1, 0), math.ceil(total_steps * cli.warmup_ratio))

    args = GKDConfig(
        output_dir=str(output_dir),
        lmbda=0.0,
        seq_kd=False,
        beta=0.0,
        temperature=1.0,
        max_length=cli.max_length,
        num_train_epochs=cli.num_train_epochs,
        max_steps=cli.max_steps,
        per_device_train_batch_size=cli.per_device_train_batch_size,
        gradient_accumulation_steps=cli.gradient_accumulation_steps,
        learning_rate=cli.learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=warmup_steps,
        optim="adamw_torch_fused",
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        disable_dropout=True,
        use_liger_kernel=cli.use_liger_kernel,
        packing=False,
        shuffle_dataset=True,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        eval_strategy="no",
        report_to=cli.report_to,
        run_name="qwen3-1.7b-align-qwen3-4b-1",
        logging_steps=10,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=None,
        save_only_model=False,
        seed=cli.seed,
        data_seed=cli.seed,
    )
    # FP32 trainable parameters retain small optimizer updates; BF16 autocast
    # handles the forward pass. The frozen teacher only needs BF16 weights.
    student = AutoModelForCausalLM.from_pretrained(
        str(student_path), local_files_only=True, dtype=torch.float32, attn_implementation="sdpa")
    teacher = AutoModelForCausalLM.from_pretrained(
        str(teacher_path), local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa")
    student.config.use_cache = False
    teacher.config.use_cache = False
    student.requires_grad_(True)
    teacher.requires_grad_(False)
    teacher.eval()
    for name, model in (("student", student), ("teacher", teacher)):
        context = model.config.get_text_config().max_position_embeddings
        if cli.max_length > context:
            raise ValueError(f"max-length exceeds {name} context length {context}")
    trainer = OfflineGKDTrainer(model=student, teacher_model=teacher, args=args,
                         processing_class=tokenizer, data_collator=OfflineKDCollator(tokenizer.pad_token_id),
                         train_dataset=train_dataset)
    # Cached labels are available before training_step, so Trainer can count the
    # actual response tokens across accumulation batches for GKD normalization.
    trainer.model_accepts_loss_kwargs = True
    if trainer.is_world_process_zero():
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {**vars(cli), "world_size": world_size, "effective_batch": effective_batch,
                    "raw_examples": len(raw), "usable_examples": len(train_dataset),
                    "estimated_optimizer_steps": total_steps, "warmup_steps": warmup_steps,
                    "objective": "forward_kl_teacher_to_student", "temperature": 1.0,
                    "lmbda": 0.0, "seq_kd": False, "beta": 0.0,
                    "loss_positions": "all_teacher_generated_response_tokens_and_eos",
                    "loss_normalization": "response_token_count_across_accumulation_batches",
                    "rollout_metadata": rollout_metadata,
                    "student_parameter_dtype": "float32", "teacher_parameter_dtype": "bfloat16"}
        (output_dir / "align_run_config.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"Offline forward-KL KD: {len(train_dataset)}/{len(raw)} responses kept (no cropping); effective batch {effective_batch}; "
              f"~{total_steps} optimizer updates; save every 100 updates", flush=True)
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output_dir / "final"))
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
