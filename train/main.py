"""Direct-OPD LoRA training with two frozen teachers and live vLLM student rollouts."""

import argparse
import json
import os
from pathlib import Path

from .direct_opd import DirectOPDState, resolve_model_source, resolve_resume, tokenizer_source

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-model-path", "--student_model_path", "--student-model", default="/models/qwen3-1.7b",
                        help="Local full student model/checkpoint or LoRA adapter for a new run")
    parser.add_argument("--model-before", "--teacher-model-before-path", required=True, help="Pre-RL teacher model/checkpoint")
    parser.add_argument("--model-after", "--teacher-model-after-path", required=True, help="Post-RL teacher model/checkpoint")
    parser.add_argument("--data-path", type=Path, default=PROJECT_ROOT / "dataset/train/dapo-17k.parquet")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "train/checkpoint/direct-opd")
    parser.add_argument("--resume-from-checkpoint", nargs="?", const="latest")
    parser.add_argument("--student-vllm-server-url", "--student_vllm_server_url", default="http://127.0.0.1:8000")
    parser.add_argument("--student-vllm-group-port", type=int, default=51216)
    parser.add_argument("--teacher-gpu", default="1", help="Physical GPU ID for both teachers and the frozen student")
    parser.add_argument("--teacher-max-model-len", type=int, default=8192)
    parser.add_argument("--teacher-score-chunk-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1, help="Prompts per training microbatch")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", "--max-completion-length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--loss-top-k", "--top-k", type=int, default=16)
    parser.add_argument("--kl-beta", "--kl_beta", "--alpha", type=float, default=1.0)
    parser.add_argument("--adaptive-kl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--alpha-epsilon", type=float, default=0.01)
    parser.add_argument("--alpha-min", type=float, default=0.5)
    parser.add_argument("--alpha-max", type=float, default=2.5)
    parser.add_argument("--save-steps", type=int, default=45)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", default="none", choices=("none", "wandb"))
    args = parser.parse_args(argv)
    for name in ("max_steps", "per_device_train_batch_size", "gradient_accumulation_steps", "num_generations",
                 "max_prompt_length", "max_new_tokens", "loss_top_k", "save_steps", "logging_steps",
                 "teacher_max_model_len", "teacher_score_chunk_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0 or args.temperature <= 0 or not 0 < args.top_p <= 1:
        parser.error("learning-rate and temperature must be positive; top-p must be in (0, 1]")
    if args.kl_beta < 0 or args.alpha_epsilon < 0 or not 0 <= args.alpha_min <= args.alpha_max:
        parser.error("Invalid KL coefficient or adaptive KL bounds")
    if args.adaptive_kl and not args.alpha_min <= args.kl_beta <= args.alpha_max:
        parser.error("Adaptive KL requires alpha within [alpha-min, alpha-max]; disable it with --no-adaptive-kl")
    if args.max_prompt_length + args.max_new_tokens > args.teacher_max_model_len:
        parser.error("Teacher context must fit max-prompt-length + max-new-tokens")
    return args


def run_signature(cli, tokenizer):
    from .kd_data import file_sha256, tokenizer_fingerprint

    settings = ("model_before", "model_after", "student_model_path", "loss_top_k", "kl_beta", "adaptive_kl",
                "alpha_epsilon", "alpha_min", "alpha_max", "enable_thinking", "num_generations",
                "temperature", "top_p", "max_prompt_length", "max_new_tokens", "seed",
                "per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate")
    return {**{key: getattr(cli, key) for key in settings}, "data_path": str(cli.data_path),
            "data_sha256": file_sha256(cli.data_path), "tokenizer": tokenizer_fingerprint(tokenizer)}


def main(argv=None):
    cli = parse_args(argv)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This entry uses one training process plus a separate vLLM server; launch with python -m train.main")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    from .myconfig import MyConfig
    from .mytrainer import MyTrainer
    from .utils import prepare_messages, prompt_token_ids

    if not torch.cuda.is_available():
        raise RuntimeError("Direct-OPD training requires NVIDIA CUDA GPUs")
    set_seed(cli.seed)
    for name in ("student_model_path", "model_before", "model_after"):
        value = str(Path(getattr(cli, name)).expanduser().resolve())
        resolve_model_source(value)
        setattr(cli, name, value)
    cli.data_path = cli.data_path.expanduser().resolve()
    cli.output_dir = cli.output_dir.expanduser().resolve()
    resume = resolve_resume(cli.output_dir, cli.resume_from_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source(cli.student_model_path), local_files_only=True, padding_side="left")
    if tokenizer.eos_token_id is None or not tokenizer.chat_template:
        raise ValueError("The student needs an EOS token and a chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    opd_state = DirectOPDState(cli.kl_beta, run_signature(cli, tokenizer))
    if resume:
        opd_state.restore(resume)

    train_dataset = Dataset.from_parquet(str(cli.data_path), cache_dir=str(PROJECT_ROOT / "tmp/datasets"))
    train_dataset = train_dataset.map(prepare_messages, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.filter(lambda row: 0 < len(prompt_token_ids(
        tokenizer, row["message"], cli.enable_thinking)) <= cli.max_prompt_length)
    if not len(train_dataset):
        raise ValueError("No usable prompts remain after length filtering")

    base, initial_adapter = resolve_model_source(cli.student_model_path)
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, dtype="auto")
    model.config.use_cache = False
    adapter = resume or initial_adapter
    lora_config = None
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True, local_files_only=True)
    else:
        lora_config = LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    args = MyConfig(
        output_dir=str(cli.output_dir), student_model_path=cli.student_model_path,
        model_before=cli.model_before, model_after=cli.model_after,
        student_vllm_server_url=cli.student_vllm_server_url, student_vllm_group_port=cli.student_vllm_group_port,
        teacher_gpu=cli.teacher_gpu, teacher_max_model_len=cli.teacher_max_model_len,
        teacher_score_chunk_size=cli.teacher_score_chunk_size,
        per_device_train_batch_size=cli.per_device_train_batch_size,
        gradient_accumulation_steps=cli.gradient_accumulation_steps,
        learning_rate=cli.learning_rate, max_steps=cli.max_steps, max_length=cli.max_prompt_length,
        max_new_tokens=cli.max_new_tokens, num_generations=cli.num_generations,
        temperature=cli.temperature, top_p=cli.top_p, loss_top_k=cli.loss_top_k,
        kl_beta=cli.kl_beta, adaptive_kl=cli.adaptive_kl, alpha_epsilon=cli.alpha_epsilon,
        alpha_min=cli.alpha_min, alpha_max=cli.alpha_max, enable_thinking=cli.enable_thinking,
        eval_strategy="no", report_to=cli.report_to, run_name="direct-opd",
        save_strategy="steps", save_steps=cli.save_steps, save_only_model=False,
        logging_strategy="steps", logging_steps=cli.logging_steps,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True, seed=cli.seed, dataloader_num_workers=0,
        lr_scheduler_type="constant", warmup_ratio=0.0, weight_decay=0.0,
    )
    trainer = MyTrainer(model=model, args=args, train_dataset=train_dataset,
                        processing_class=tokenizer, peft_config=lora_config, opd_state=opd_state)
    try:
        cli.output_dir.mkdir(parents=True, exist_ok=True)
        (cli.output_dir / "run_config.json").write_text(json.dumps({
            **vars(cli), "usable_prompts": len(train_dataset), "vllm_base_model": base,
            "signature": opd_state.signature,
        }, indent=2, default=str) + "\n")
        trainer.train(resume_from_checkpoint=resume)
        trainer.save_model(str(cli.output_dir / "final"))
        trainer.save_state()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
