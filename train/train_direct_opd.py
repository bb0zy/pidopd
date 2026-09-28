"""Local-only top-1 on-policy training with a solution-conditioned reference teacher."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE",
            "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK"):
    os.environ[key] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
from trl.data_utils import apply_chat_template
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from objective import loss_terms, selected_log_probs, update_alpha

ROOT = Path(__file__).resolve().parents[1]
DAPO_PROMPT = '''Solve the following math problem step by step.
The last line of your response should be of the form
Answer: $Answer (without quotes) where $Answer is the answer to the problem.

{question}

Remember to put your answer on its own line after "Answer:".'''
PRIVILEGED_SUFFIX = "\n\nReference solution (privileged information):\n{solution}\n\nAnswer the original problem."


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-root", type=Path, default=Path("/models"))
    for name in ("student", "teacher-ref"):
        p.add_argument(f"--{name}", required=True, help="Directory name under models-root, or absolute local path")
    p.add_argument("--data", type=Path, nargs="+", default=[
        ROOT / "dataset/train/train-00000-of-00005.parquet",
        ROOT / "dataset/train/train-00001-of-00005.parquet"],
        help="Ordered training Parquet files; each file is read in row order")
    p.add_argument("--output", type=Path, default=ROOT / "train/outputs")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--prompts-per-step", type=int, default=64)
    p.add_argument("--rollouts-per-prompt", type=int, default=1)
    p.add_argument("--max-prompt-tokens", type=int, default=1024)
    p.add_argument("--max-teacher-prompt-tokens", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--fixed-alpha", action="store_true")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--student-device", default="cuda:0")
    p.add_argument("--teacher-ref-device", default="cuda:1")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj",
                                                               "gate_proj", "up_proj", "down_proj"])
    p.add_argument("--prompt-key", default="problem", help="Field holding text or user-ended messages")
    p.add_argument("--solution-key", default="solution", help="Teacher-only reference solution field")
    p.add_argument("--prompt-format", choices=["dapo", "existing"], default="existing")
    p.add_argument("--chat-template-kwargs", default="{}")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--resume", type=Path, help="Local checkpoint created by this script")
    a = p.parse_args()
    for key in ("steps", "prompts_per_step", "rollouts_per_prompt", "max_prompt_tokens", "max_teacher_prompt_tokens", "max_new_tokens", "save_every", "lora_r", "lora_alpha"):
        if getattr(a, key) < 1:
            p.error(f"{key} must be positive")
    for key in ("learning_rate", "alpha", "max_grad_norm"):
        if not math.isfinite(getattr(a, key)) or getattr(a, key) <= 0:
            p.error(f"{key} must be finite and positive")
    if not a.fixed_alpha and not 0.5 <= a.alpha <= 2.5:
        p.error("Adaptive alpha must start within [0.5, 2.5]")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        p.error("Single-process implementation: do not use torchrun/DDP/DeepSpeed")
    for key in ("student", "teacher_ref"):
        path = Path(getattr(a, key)).expanduser()
        path = (path if path.is_absolute() else a.models_root / path).resolve()
        if not (path / "config.json").is_file():
            p.error(f"Missing local model config: {path}")
        setattr(a, key, path)
    if a.prompt_key == a.solution_key:
        p.error("Prompt and solution fields must be different")
    a.template_kwargs = json.loads(a.chat_template_kwargs)
    if not isinstance(a.template_kwargs, dict) or any(k in a.template_kwargs for k in (
            "tokenize", "add_generation_prompt", "continue_final_message")):
        p.error("Invalid chat-template-kwargs")
    return a


def tokenizer_for(path, a):
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True,
                                        trust_remote_code=a.trust_remote_code)


def load_model(path, device, a):
    model = AutoModelForCausalLM.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=a.trust_remote_code,
        dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    model.requires_grad_(False)
    # Disable dropout so rollout, coefficient pass and gradient pass use the same policy.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model.eval()
    return model


def load_student(a):
    # Always use the original frozen base, including when resuming an adapter.
    base = load_model(a.student, a.student_device, a)
    if a.resume:
        student = PeftModel.from_pretrained(base, str(a.resume), is_trainable=True,
                                             local_files_only=True)
    else:
        config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=a.lora_r,
                            lora_alpha=a.lora_alpha, target_modules=a.lora_target_modules,
                            lora_dropout=0.0, bias="none", init_lora_weights=True,
                            modules_to_save=None)
        student = get_peft_model(base, config)
    # No trainable bias, embeddings, head or other shared parameters: disabling
    # adapters must recover precisely the immutable original student reference.
    for name, parameter in student.named_parameters():
        if parameter.requires_grad and not any(part in name for part in (".lora_A.", ".lora_B.")):
            raise ValueError(f"Only LoRA A/B parameters may be trainable: {name}")
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.enable_input_require_grads()
    student.print_trainable_parameters()
    return student


def read_prompts(a, tok):
    import pyarrow.parquet as pq
    required = {a.prompt_key, a.solution_key}
    rows, file_rows = [], []
    if len({path.resolve() for path in a.data}) != len(a.data):
        raise ValueError("Training data files must be distinct")
    for path in a.data:
        if not path.is_file() or path.suffix != ".parquet":
            raise ValueError(f"Expected a local training Parquet file: {path}")
        parquet = pq.ParquetFile(path)
        missing = required - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"Missing training data fields in {path}: {sorted(missing)}")
        part = parquet.read(columns=sorted(required)).to_pylist()
        file_rows.append({"path": str(path), "rows": len(part)})
        rows.extend(part)
    encoded = []
    overlong, teacher_overlong = 0, 0
    for i, row in enumerate(rows):
        value = row.get(a.prompt_key)
        if isinstance(value, str):
            messages = [{"role": "user", "content": value}]
        elif isinstance(value, list) and value:
            messages = [dict(m) for m in value]
        else:
            raise ValueError(f"Row {i}: missing/invalid prompt field {a.prompt_key}")
        if any(m.get("role") not in {"system", "user", "assistant"}
               or not isinstance(m.get("content"), str) for m in messages):
            raise ValueError(f"Row {i}: only text role/content messages are supported")
        if messages[-1]["role"] != "user":
            raise ValueError(f"Row {i}: prompt must end with a user, not a reference answer")
        if a.prompt_format == "dapo":
            if len(messages) != 1:
                raise ValueError("DAPO format expects one user message; use --prompt-format existing for multi-turn")
            question = messages[0]["content"].strip()
            # Remove only the exact known suffix in local DAPO-style source data.
            suffix = r"Please reason step by step, and put your final answer within \boxed{}."
            if question.endswith(suffix):
                question = question[:-len(suffix)].rstrip()
            messages[0]["content"] = DAPO_PROMPT.format(question=question)
        solution = row.get(a.solution_key)
        if not isinstance(solution, str) or not solution.strip():
            raise ValueError(f"Row {i}: missing/empty solution field {a.solution_key}")
        text = apply_chat_template({"prompt": messages}, processing_class=tok, **a.template_kwargs)["prompt"]
        ids = tok.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Row {i}: empty tokenized prompt")
        if len(ids) > a.max_prompt_tokens:
            overlong += 1
            continue
        teacher_messages = [dict(m) for m in messages]
        teacher_messages[-1]["content"] += PRIVILEGED_SUFFIX.format(solution=solution.strip())
        teacher_text = apply_chat_template({"prompt": teacher_messages}, processing_class=tok,
                                           **a.template_kwargs)["prompt"]
        teacher_ids = tok.encode(teacher_text, add_special_tokens=False)
        if not teacher_ids:
            raise ValueError(f"Row {i}: empty tokenized teacher prompt")
        if len(teacher_ids) > a.max_teacher_prompt_tokens:
            teacher_overlong += 1
            continue
        encoded.append({"student_ids": ids, "teacher_ids": teacher_ids})
    if not encoded:
        raise ValueError("No usable prompts remain after length filtering")
    return encoded, {"files": file_rows, "source_rows": len(rows), "usable_rows": len(encoded),
                     "overlong_rows": overlong, "teacher_overlong_rows": teacher_overlong}


def response_logits(model, ids, prompt_len):
    device = next(model.parameters()).device
    x = ids.to(device)[None, :]
    output = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False)
    # Logits at prompt_len-1 predict the first generated token, including EOS.
    return output.logits[0, prompt_len - 1:-1, :]


@torch.no_grad()
def rollout(student, prompts, a, tok):
    student.eval()
    eos = student.generation_config.eos_token_id
    if eos is None:
        eos = tok.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
    config = GenerationConfig(max_new_tokens=a.max_new_tokens, do_sample=True,
                              temperature=1.0, top_p=1.0, top_k=0,
                              eos_token_id=eos, pad_token_id=tok.pad_token_id,
                              bos_token_id=tok.bos_token_id, use_cache=True)
    records = []
    for prompt in prompts:
        ids = prompt["student_ids"]
        x = torch.tensor([ids], device=a.student_device)
        for _ in range(a.rollouts_per_prompt):
            sequence = student.generate(input_ids=x, attention_mask=torch.ones_like(x),
                                        generation_config=config)[0].cpu()
            generated = sequence[len(ids):].tolist()
            for j, token in enumerate(generated):
                if token in eos_ids:
                    sequence = sequence[:len(ids) + j + 1]
                    break
            if len(sequence) <= len(ids):
                raise RuntimeError("Empty student rollout")
            records.append({"ids": sequence, "prompt_len": len(ids),
                            "teacher_ids": prompt["teacher_ids"]})
    return records


@torch.no_grad()
def annotate(records, student, teacher_ref):
    student.eval()
    reward_sum, token_count = 0.0, 0
    for record in records:
        ids, length = record["ids"], record["prompt_len"]
        logits = response_logits(student, ids, length)
        top1 = logits.argmax(-1).cpu()
        del logits
        privileged_ids = record["teacher_ids"]
        teacher_sequence = torch.tensor(privileged_ids + ids[length:].tolist())
        logits = response_logits(teacher_ref, teacher_sequence, len(privileged_ids))
        teacher_lp = selected_log_probs(logits, top1.to(logits.device)).cpu()
        del logits
        logits = response_logits(teacher_ref, ids, length)
        reference_lp = selected_log_probs(logits, top1.to(logits.device)).cpu()
        del logits
        # Context manager restores LoRA for subsequent rollout / student forward.
        # This no-grad base-only pass finishes before any gradient graph is built.
        with student.disable_adapter():
            logits = response_logits(student, ids, length)
            anchor_lp = selected_log_probs(logits, ids[length:].to(logits.device)).cpu()
            del logits
        shift = teacher_lp - reference_lp
        if not torch.isfinite(shift).all() or not torch.isfinite(anchor_lp).all():
            raise FloatingPointError("Non-finite teacher shift or anchor log probability")
        record.update(top1=top1, shift=shift, anchor_logp=anchor_lp)
        reward_sum += shift.sum().item()
        token_count += shift.numel()
    return reward_sum / token_count


def save_checkpoint(path, student, tok, optimizer, step, alpha, a):
    path.mkdir(parents=True, exist_ok=False)
    student.save_pretrained(path, safe_serialization=True, save_embedding_layers=False)
    tok.save_pretrained(path)
    torch.save({"optimizer": optimizer.state_dict(), "step": step, "alpha": alpha}, path / "training_state.pt")
    (path / "run_arguments.json").write_text(json.dumps(vars(a), default=str, indent=2), encoding="utf-8")
    # Written last so interrupted partial checkpoints are not resumed.
    (path / "complete.json").write_text(json.dumps({"step": step, "format": "solution-conditioned-sequential-v2"}), encoding="utf-8")


def main():
    a = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; no CPU training fallback")
    for key in ("student_device", "teacher_ref_device"):
        device = torch.device(getattr(a, key))
        if device.type not in {"cuda", "cpu"} or (key == "student_device" and device.type != "cuda"):
            raise ValueError(f"Unsupported {key}: {device}")
    set_seed(a.seed)
    tok = tokenizer_for(a.student, a)
    if not tok.chat_template:
        raise ValueError("Student tokenizer must include a local chat template")
    # Token probability subtraction requires the same token events and prefixes.
    for path in (a.teacher_ref,):
        other = tokenizer_for(path, a)
        if tok.get_vocab() != other.get_vocab():
            raise ValueError(f"Tokenizer token-to-ID mapping differs: {path}. Cross-tokenizer alignment is not implemented.")
    if tok.pad_token_id is None:
        if tok.eos_token_id is None:
            raise ValueError("Tokenizer needs pad_token or eos_token")
        tok.pad_token = tok.eos_token
    prompts, data_info = read_prompts(a, tok)
    start, alpha = 0, a.alpha
    if a.resume:
        if not (a.resume / "complete.json").is_file():
            raise ValueError("Resume checkpoint is incomplete")
        marker = json.loads((a.resume / "complete.json").read_text(encoding="utf-8"))
        if marker.get("format") != "solution-conditioned-sequential-v2":
            raise ValueError("Resume requires a solution-conditioned checkpoint with sequential sampling")
        previous = json.loads((a.resume / "run_arguments.json").read_text(encoding="utf-8"))
        # Only output, device placement, save frequency and final step can change.
        flexible = {"resume", "output", "steps", "save_every", "student_device", "teacher_ref_device"}
        for key, value in vars(a).items():
            if key not in flexible and previous.get(key) != json.loads(json.dumps(value, default=str)):
                raise ValueError(f"Resume configuration differs: {key}")
    student = load_student(a)
    teacher_ref = load_model(a.teacher_ref, a.teacher_ref_device, a)
    for model in (student, teacher_ref):
        if model.get_output_embeddings().weight.shape[0] != student.get_output_embeddings().weight.shape[0]:
            raise ValueError("Models must have identical output vocabulary dimensions")
        context = getattr(model.config, "max_position_embeddings", None)
        prompt_limit = a.max_prompt_tokens if model is student else a.max_teacher_prompt_tokens
        if context and prompt_limit + a.max_new_tokens > context:
            raise ValueError(f"Training length exceeds model context {context}")
    trainable_parameters = [p for p in student.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable LoRA parameters; check target modules")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=a.learning_rate, weight_decay=0.0)
    if a.resume:
        state = torch.load(a.resume / "training_state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        start, alpha = state["step"], state["alpha"]
        del state
    if start >= a.steps:
        raise ValueError("steps must exceed the resumed step")
    available_steps = math.ceil(len(prompts) / a.prompts_per_step)
    final_step = min(a.steps, available_steps)
    if start >= final_step:
        raise ValueError("No unused prompts remain after the resumed step")
    if final_step < a.steps:
        print(f"Only {len(prompts)} usable prompts: training ends at step {final_step} "
              f"before requested step {a.steps}, without repeating prompts.", flush=True)
    run_dir = a.output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {"arguments": vars(a), "data": data_info, "cuda": torch.version.cuda,
                "versions": {p: importlib.metadata.version(p) for p in ("torch", "accelerate", "transformers", "trl", "peft", "pyarrow")}}
    (run_dir / "config.json").write_text(json.dumps(metadata, default=str, indent=2), encoding="utf-8")
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for step in range(start, final_step):
            begin = time.perf_counter()
            # Step-indexed RNG makes uninterrupted/resumed runs use the same seeds.
            set_seed(a.seed + step)
            offset = step * a.prompts_per_step
            batch = prompts[offset:offset + a.prompts_per_step]
            records = rollout(student, batch, a, tok)
            mean_shift = annotate(records, student, teacher_ref)
            if not a.fixed_alpha:
                alpha = update_alpha(alpha, mean_shift)
            student.train()
            optimizer.zero_grad(set_to_none=True)
            policy_sum, kl_sum = 0.0, 0.0
            total_tokens = sum(len(r["ids"]) - r["prompt_len"] for r in records)
            # One optimizer step for the ENTIRE on-policy rollout batch.
            for record in records:
                ids, length = record["ids"], record["prompt_len"]
                logits = response_logits(student, ids, length)
                device = logits.device
                top_lp = selected_log_probs(logits, record["top1"].to(device))
                sampled_lp = selected_log_probs(logits, ids[length:].to(device))
                pg, kl = loss_terms(top_lp, sampled_lp, record["shift"].to(device),
                                    record["anchor_logp"].to(device))
                weight = (len(ids) - length) / total_tokens
                loss = weight * (pg + alpha * kl)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss; no optimizer update performed")
                loss.backward()
                policy_sum += pg.detach().item() * weight
                kl_sum += kl.detach().item() * weight
                del logits, top_lp, sampled_lp, pg, kl, loss
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, a.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            metrics = {"step": step + 1, "policy_loss": policy_sum, "sampled_k3_kl": kl_sum,
                       "mean_top1_shift": mean_shift, "alpha": alpha, "grad_norm": float(grad_norm),
                       "response_tokens": total_tokens, "rollouts": len(records),
                       "mean_response_length": total_tokens / len(records),
                       "seconds": time.perf_counter() - begin}
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            print(json.dumps(metrics), flush=True)
            if (step + 1) % a.save_every == 0 or step + 1 == final_step:
                save_checkpoint(run_dir / f"checkpoint-{step + 1}", student, tok, optimizer, step + 1, alpha, a)
    print(f"Training artifacts: {run_dir}")


if __name__ == "__main__":
    main()
