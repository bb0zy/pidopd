import argparse
import os
from pathlib import Path

if __name__ == "__main__":
    # Training uses GPU0; the teacher process and manually started vLLM server use GPU1.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .myconfig import MyConfig
    from .mytrainer import MyTrainer
    from .utils import prepare_messages

    parser = argparse.ArgumentParser(description="Train with a frozen Transformers teacher.")
    for name, value_type in (
        ("student_model_path", str),
        ("teacher_model_path", str),
        ("student_vllm_server_url", str),
        ("kl_beta", float),
        ("kl_top_k", int),
    ):
        config_field = MyConfig.__dataclass_fields__[name]
        parser.add_argument(
            f"--{name}", f"--{name.replace('_', '-')}",
            type=value_type,
            default=config_field.default,
            help=config_field.metadata["help"],
        )
    cli_args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "dataset" / "train"

    train_files = [
        str(data_dir / "train-00000-of-00002.parquet"),
        str(data_dir / "train-00001-of-00002.parquet"),
    ]

    train_dataset = Dataset.from_parquet(
        train_files,
        cache_dir=str(project_root / "tmp" / "datasets"),
    )
    print(train_dataset)

    stu_sys_prompt = r"""Solve the following math problem step by step.
The last line of your response should contain only the final answer enclosed in \boxed{}."""

    teacher_sys_prompt = stu_sys_prompt + "\n" + """A reference solution is provided below as additional information.
Use it to understand the mathematical relationships and the correct result.
Alternative valid solution methods and wording are equally acceptable.
Present a self-contained solution without mentioning the reference solution."""
    train_dataset = train_dataset.map(
        prepare_messages,
        fn_kwargs={
            "stu_sys_prompt": stu_sys_prompt,
            "teacher_sys_prompt": teacher_sys_prompt,
        },
        remove_columns=train_dataset.column_names,
    )

    model_path = cli_args.student_model_path
    model_name = Path(model_path).name
    checkpoint_dir = project_root / "train" / "checkpoint" / model_name
    final_model_dir = project_root / "train" / "outputs" / model_name
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype="auto",
    )

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules="all-linear",
        bias="none",
    )

    args = MyConfig(
        output_dir=str(checkpoint_dir),
        student_model_path=cli_args.student_model_path,
        teacher_model_path=cli_args.teacher_model_path,
        student_vllm_server_url=cli_args.student_vllm_server_url,
        teacher_gpu="1",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=5e-6,
        num_train_epochs=1,
        max_length=1024,
        max_new_tokens=2048,
        loss_top_k=16,
        kl_beta=cli_args.kl_beta,
        kl_top_k=cli_args.kl_top_k,
        eval_strategy="no",
        report_to="wandb",
        run_name="pidopd-run",
        save_strategy="steps",
        save_steps=50, 
        logging_strategy="steps",
        logging_steps=10
    )

    trainer = MyTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    try:
        trainer.train()
        trainer.save_model(str(final_model_dir))
    finally:
        try:
            trainer.student_generation.close()
        finally:
            trainer.teacher_client.close()
