"""Offline math evaluation. See the project root README.md; no downloads or training."""
from __future__ import annotations
# CUDA_VISIBLE_DEVICES=3 python eval/run_eval.py --checkpoint 500
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

# Must be set before importing Hugging Face/vLLM libraries, including workers.
for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
            "HF_HUB_DISABLE_TELEMETRY", "VLLM_NO_USAGE_STATS", "DO_NOT_TRACK"):
    os.environ[key] = "1"

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = ROOT / "train/grpo_checkpoint"
LEGACY_SUFFIX = r"Please reason step by step, and put your final answer within \boxed{}."
SYSTEM_PROMPT = r"""Solve the following math problem step by step.
The last line of your response should contain only the final answer enclosed in \boxed{}."""

def checkpoint_paths(directory, selection="all"):
    """Discover complete training checkpoints in numeric step order."""
    if not directory.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {directory}")
    candidates = sorted(
        (path for path in directory.iterdir()
         if path.is_dir() and re.fullmatch(r"checkpoint-\d+", path.name)),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    if selection not in {"all", "latest"}:
        if not re.fullmatch(r"\d+", selection):
            raise ValueError("--checkpoint must be all, latest, or a step number (e.g. 100)")
        candidates = [path for path in candidates if int(path.name.split("-")[-1]) == int(selection)]
    weights = ("model.safetensors", "model.safetensors.index.json",
               "pytorch_model.bin", "pytorch_model.bin.index.json")
    complete = []
    for path in candidates:
        if (path / "config.json").is_file() and (path / "trainer_state.json").is_file() and any(
            (path / name).is_file() for name in weights
        ):
            complete.append(path)
        else:
            print(f"Skipping incomplete checkpoint: {path}", flush=True)
    if not complete:
        raise ValueError(f"No complete checkpoint found in {directory} for selection {selection!r}")
    return complete[-1:] if selection == "latest" else complete


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--model", type=Path, help="Evaluate one complete local model directory")
    source.add_argument("--checkpoint-dir", type=Path,
                        help=f"Scan checkpoint-* directories (default: {CHECKPOINT_ROOT})")
    p.add_argument("--checkpoint", default="all", help="Checkpoint selection: all (default), latest, or a step number")
    p.add_argument("--lora-path", type=Path, help="Local LoRA adapter directory, including a training checkpoint")
    p.add_argument("--data", type=Path, default=ROOT / "dataset/eval")
    p.add_argument("--output", type=Path, default=ROOT / "eval/results")
    p.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-model-len", type=int, default=40960)
    p.add_argument("--max-new-tokens", type=int, default=38912)
    p.add_argument("--samples", type=int, choices=[16,32], default=16, help="avg@n requires n completions per question")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, help="First N rows per file")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-num-seqs", type=int, default=128, help="vLLM scheduler concurrency")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--warmup", type=int, default=1, help="Untimed warmup calls")
    p.add_argument("--prompt-key", help="Optional dotted field path")
    p.add_argument("--answer-key", help="Optional dotted field path")
    p.add_argument("--plain-prompt", action="store_true", help="Base model without chat template")
    p.add_argument("--chat-template-kwargs", default='{"enable_thinking":false}', help='JSON, e.g. {"enable_thinking":true}')
    p.add_argument("--trust-remote-code", action="store_true", help="Allow custom code in local model")
    a = p.parse_args(argv)
    for name in ("batch_size", "max_model_len", "max_new_tokens", "samples", "tensor_parallel_size", "max_num_seqs"):
        if getattr(a, name) < 1:
            p.error(f"{name} must be positive")
    if a.max_new_tokens >= a.max_model_len:
        p.error("max-new-tokens must be smaller than max-model-len")
    if a.warmup < 0 or (a.limit is not None and a.limit < 1):
        p.error("warmup must be nonnegative; limit must be positive")
    if not math.isfinite(a.temperature) or a.temperature < 0 or not 0 < a.top_p <= 1:
        p.error("temperature must be finite and >= 0; top-p must be in (0, 1]")
    if not 0 < a.gpu_memory_utilization <= 1:
        p.error("gpu-memory-utilization must be in (0, 1]")
    if a.samples > 1 and a.temperature == 0:
        p.error("Use positive temperature for multiple independent samples")
    if a.backend == "transformers" and a.tensor_parallel_size != 1:
        p.error("Transformers backend uses one GPU; tensor parallelism is vLLM-only")
    a.data = a.data.expanduser().resolve()
    a.output = a.output.expanduser().resolve()
    a.checkpoint_paths = []
    if a.model is None:
        if a.lora_path is not None:
            p.error("--lora-path requires an explicit --model base-model path")
        a.checkpoint_dir = (a.checkpoint_dir or CHECKPOINT_ROOT).expanduser().resolve()
        try:
            a.checkpoint_paths = checkpoint_paths(a.checkpoint_dir, a.checkpoint)
        except ValueError as exc:
            p.error(str(exc))
    else:
        if a.checkpoint != "all":
            p.error("--checkpoint is only used when scanning a checkpoint directory")
        a.model = a.model.expanduser().resolve()
        if not a.model.is_dir() or not (a.model / "config.json").is_file():
            p.error(f"Expected complete local model directory with config.json: {a.model}")
    a.tokenizer_path = a.model
    a.lora_rank = None
    if a.lora_path is not None:
        a.lora_path = a.lora_path.expanduser().resolve()
        adapter_config = json.loads((a.lora_path / "adapter_config.json").read_text(encoding="utf-8"))
        a.lora_rank = max([adapter_config["r"], *adapter_config.get("rank_pattern", {}).values()])
        if (a.lora_path / "tokenizer_config.json").is_file():
            a.tokenizer_path = a.lora_path
    if not a.data.exists():
        p.error(f"Local dataset does not exist: {a.data}")
    a.template_kwargs = json.loads(a.chat_template_kwargs)
    if not isinstance(a.template_kwargs, dict):
        p.error("chat-template-kwargs must be a JSON object")
    if {"tokenize", "add_generation_prompt", "continue_final_message"} & a.template_kwargs.keys():
        p.error("Do not override tokenization or generation-prompt control arguments")
    return a


def field(row, path):
    for key in path.split("."):
        if not isinstance(row, dict) or key not in row:
            return None
        row = row[key]
    return row


def choose(row, explicit, defaults):
    for key in [explicit] if explicit else defaults:
        value = field(row, key)
        if value is not None:
            return value
    raise ValueError(f"Missing field {explicit or defaults}; available keys: {list(row)}")


def load_rows(path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: JSON must contain a list of records")
    return rows


def boxed(text):
    """Extract last balanced boxed expression, including nested braces."""
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for i in range(start, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return text[start:i]
    return None


def extract(text):
    # Ignore completed reasoning blocks before looking for the final answer.
    text = text.rsplit("</think>", 1)[-1]
    answer = boxed(text)
    if answer is not None:
        return answer
    matches = re.findall(r"^[ \t]*(?:Answer|Final answer|答案)[ \t]*[:：][ \t]*([^\r\n]+)",
                         text, re.I | re.M)
    if matches:
        answer = matches[-1].strip().rstrip(".。")
        return boxed(answer) if boxed(answer) is not None else answer
    # Never take an arbitrary last number from a chain of thought.
    if re.fullmatch(r"\s*[+-]?\d+(?:\.\d+|/\d+)?\s*", text):
        return text.strip()
    return None


def prepare(row, a, tokenizer):
    from trl.data_utils import apply_chat_template

    prompt = choose(row, a.prompt_key, ["prompt", "messages", "problem", "question"])
    gold = choose(row, a.answer_key, ["reward_model.ground_truth", "answer", "ground_truth", "target"])
    if not isinstance(gold, (str, int, float)) or isinstance(gold, bool):
        raise ValueError("Ground truth must be a scalar string or number")
    if isinstance(gold, float) and not math.isfinite(gold):
        raise ValueError("Ground truth cannot be NaN or infinity")
    gold = boxed(str(gold)) or str(gold)
    if not gold.strip():
        raise ValueError("Empty ground truth")
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list) and prompt:
        messages = [dict(m) for m in prompt]
    else:
        raise ValueError("Prompt must be text or a nonempty role/content message list")
    if any(m.get("role") not in {"system", "user", "assistant"}
           or not isinstance(m.get("content"), str) for m in messages):
        raise ValueError("Only text system/user/assistant messages are supported")
    if messages[-1]["role"] != "user":
        raise ValueError("Prompt must end with user, not a reference assistant answer")
    # All 90 local AIME/HMMT rows contain this exact instruction suffix.
    # Remove only that suffix, never arbitrary mathematical boxed expressions.
    question = messages[-1]["content"].strip()
    if question.endswith(LEGACY_SUFFIX):
        question = question[:-len(LEGACY_SUFFIX)].rstrip()
    if not question:
        raise ValueError("Empty question after removing the legacy instruction")
    messages[-1]["content"] = question
    if a.plain_prompt:
        if len(messages) != 1:
            raise ValueError("plain-prompt supports only one user message")
        rendered = SYSTEM_PROMPT + "\n\n" + question
        ids = tokenizer.encode(rendered, add_special_tokens=True)
    else:
        if messages[0]["role"] == "system":
            messages[0]["content"] = SYSTEM_PROMPT
        else:
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        if not tokenizer.chat_template:
            raise ValueError("Local tokenizer has no chat template; use --plain-prompt for base models")
        rendered = apply_chat_template({"prompt": messages}, processing_class=tokenizer, **a.template_kwargs)["prompt"]
        ids = tokenizer.encode(rendered, add_special_tokens=False)
    if not ids or len(ids) + a.max_new_tokens > a.max_model_len:
        raise ValueError(f"Prompt length {len(ids)} + generation budget exceeds {a.max_model_len}; increase context or reduce budget")
    return {"prompt": rendered, "ids": ids, "gold": gold}


class VLLMBackend:
    def __init__(self, a, tokenizer):
        from vllm import LLM, SamplingParams
        self.a = a
        self.generate_kwargs = {}
        lora_kwargs = {}
        if a.lora_path is not None:
            from vllm.lora.request import LoRARequest
            # Reserve a supported rank capacity for this adapter, including rank patterns.
            max_lora_rank = max(8, 1 << (a.lora_rank - 1).bit_length())
            lora_kwargs = {"enable_lora": True, "max_lora_rank": max_lora_rank}
            self.generate_kwargs["lora_request"] = LoRARequest(a.lora_path.name, 1, str(a.lora_path))
        self.params = SamplingParams(n=a.samples, temperature=a.temperature,
                                     top_p=a.top_p, max_tokens=a.max_new_tokens)
        self.model = LLM(model=str(a.model), tokenizer=str(a.tokenizer_path), dtype=a.dtype,
                         trust_remote_code=a.trust_remote_code,
                         tensor_parallel_size=a.tensor_parallel_size,
                         max_num_seqs=a.max_num_seqs,
                         max_num_batched_tokens = 65536,
                         gpu_memory_utilization=a.gpu_memory_utilization,
                         max_model_len=a.max_model_len, seed=a.seed,
                         enable_prefix_caching=False, generation_config="vllm", **lora_kwargs)
        scheduler = self.model.llm_engine.vllm_config.scheduler_config
        self.engine_settings = {"max_num_seqs": scheduler.max_num_seqs,
                                "max_num_batched_tokens": scheduler.max_num_batched_tokens}
        print(f"vLLM scheduler: {self.engine_settings}", flush=True)

    def generate(self, batch):
        outputs = self.model.generate([{"prompt_token_ids": r["ids"]} for r in batch],
                                      self.params, use_tqdm=False, **self.generate_kwargs)
        return [[{"text": c.text, "tokens": len(c.token_ids),
                  "finish_reason": c.finish_reason} for c in o.outputs] for o in outputs]


class TransformersBackend:
    def __init__(self, a, tokenizer):
        import torch
        from transformers import AutoModelForCausalLM, GenerationConfig
        self.a, self.tokenizer, self.torch = a, tokenizer, torch
        dtype = "auto" if a.dtype == "auto" else getattr(torch, a.dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(a.model), local_files_only=True, trust_remote_code=a.trust_remote_code,
            dtype=dtype).to("cuda").eval()
        if a.lora_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model, str(a.lora_path), local_files_only=True, is_trainable=False).eval()
        # Fresh config avoids model-supplied sampling/beam defaults.
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = tokenizer.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos]) - {None}
        self.config = GenerationConfig(
            max_new_tokens=a.max_new_tokens, do_sample=a.temperature > 0,
            num_return_sequences=a.samples, eos_token_id=eos,
            pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
            **({"temperature": a.temperature, "top_p": a.top_p, "top_k": 0}
               if a.temperature > 0 else {}))

    def generate(self, batch):
        torch, tok = self.torch, self.tokenizer
        inputs = tok.pad([{"input_ids": r["ids"], "attention_mask": [1] * len(r["ids"])}
                          for r in batch], padding=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output = self.model.generate(**inputs, generation_config=self.config)
        output = output[:, inputs["input_ids"].shape[1]:].tolist()
        completions = []
        for ids in output:
            reason = "length"
            for i, token in enumerate(ids):
                if token in self.eos:
                    ids, reason = ids[:i + 1], "stop"
                    break
            completions.append({"text": tok.decode(ids, skip_special_tokens=True),
                                "tokens": len(ids), "finish_reason": reason})
        return [completions[i:i + self.a.samples]
                for i in range(0, len(completions), self.a.samples)]


def evaluate(a):
    from answer_grader import AnswerGrader
    import torch
    from transformers import AutoTokenizer, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    versions = {}
    for package in ("torch", "accelerate", "vllm", "transformers", "trl", "peft", "pyarrow", "math-verify", "sympy", "latex2sympy2_extended"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    targets = {"torch": "2.10.0+cu128", "accelerate": "1.15.0", "transformers": "5.3.0", "trl": "1.13.0"}
    if a.backend == "vllm":
        targets["vllm"] = "0.19.1"
    for package, expected in targets.items():
        if versions[package] != expected:
            print(f"WARNING: target {package}={expected}; installed {versions[package]}", flush=True)
    if torch.version.cuda != "12.8":
        print(f"WARNING: expected CUDA runtime 12.8; got {torch.version.cuda}", flush=True)
    if a.backend == "vllm" and str(versions["transformers"]).startswith("5.3."):
        print("WARNING: official vLLM 0.19.1 excludes Transformers 5.3.x. "
              "Use the compatible custom build described in the project root README.md.", flush=True)
    set_seed(a.seed)
    tokenizer = AutoTokenizer.from_pretrained(str(a.tokenizer_path), local_files_only=True,
                                               trust_remote_code=a.trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a pad_token or eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    files = sorted(p for p in a.data.iterdir() if p.suffix in {".parquet", ".json", ".jsonl"}) if a.data.is_dir() else [a.data]
    if not files or any(p.suffix not in {".parquet", ".json", ".jsonl"} for p in files):
        raise ValueError("Provide local parquet, JSON-array or JSONL files")
    datasets = []
    for path in files:
        rows = load_rows(path)
        if a.limit:
            rows = rows[:a.limit]
        if not rows:
            raise ValueError(f"Empty dataset: {path}")
        prepared = []
        for index, row in enumerate(rows):
            try:
                item = prepare(row, a, tokenizer)
                item["grader"] = AnswerGrader(item["gold"])
            except Exception as exc:
                raise ValueError(f"{path.name}, row {index}: {exc}") from exc
            item["row_index"] = index
            prepared.append(item)
        datasets.append((path.name, prepared))
    run_dir = a.output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {"arguments": vars(a), "versions": versions, "cuda_runtime": torch.version.cuda,
                "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    (run_dir / "config.json").write_text(json.dumps(metadata, default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    started = time.perf_counter()
    backend = VLLMBackend(a, tokenizer) if a.backend == "vllm" else TransformersBackend(a, tokenizer)
    load_seconds = time.perf_counter() - started
    if a.backend == "vllm":
        metadata["vllm_scheduler"] = backend.engine_settings
        (run_dir / "config.json").write_text(json.dumps(metadata, default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    for _ in range(a.warmup):
        backend.generate(datasets[0][1][:1])
    summary = {"primary_metric": "avg@32", "samples_per_question": a.samples,
               "model_load_seconds": load_seconds, "datasets": {}}
    for name, rows in datasets:
        elapsed, tokens, counts, latencies = 0.0, 0, [], []
        truncated, failed = 0, 0
        with (run_dir / f"{name}.predictions.jsonl").open("w", encoding="utf-8") as f:
            for start in range(0, len(rows), a.batch_size):
                batch = rows[start:start + a.batch_size]
                torch.cuda.synchronize()
                begin = time.perf_counter()
                outputs = backend.generate(batch)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - begin
                elapsed += seconds
                latencies.append(seconds)
                if len(outputs) != len(batch) or any(len(o) != a.samples for o in outputs):
                    raise RuntimeError("Backend returned an unexpected number of completions")
                for row, completions in zip(batch, outputs):
                    correct = 0
                    for c in completions:
                        answer = extract(c["text"])
                        c.update(extracted_answer=answer, **row["grader"].grade(answer))
                        correct += int(c["correct"])
                        failed += int(answer is None)
                        tokens += c["tokens"]
                        truncated += int(c["finish_reason"] == "length")
                    counts.append(correct)
                    record = {"row_index": row["row_index"], "prompt": row["prompt"],
                              "prompt_tokens": len(row["ids"]), "reference": row["gold"],
                              "correct_samples": correct, "avg@32": correct / a.samples,
                              "batch_seconds": seconds, "completions": completions}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{name}: {min(start + a.batch_size, len(rows))}/{len(rows)}", flush=True)
        total = len(rows) * a.samples
        metrics = {"questions": len(rows), "completions": total,
                   "avg@32": statistics.mean(c / a.samples for c in counts),
                   "generation_seconds": elapsed, "output_tokens": tokens,
                   "output_tokens_per_second": tokens / elapsed,
                   "completions_per_second": total / elapsed,
                   "mean_batch_seconds": statistics.mean(latencies),
                   "median_batch_seconds": statistics.median(latencies),
                   "length_finish_rate": truncated / total, "answer_extraction_failure_rate": failed / total}
        summary["datasets"][name] = metrics
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({name: metrics}, ensure_ascii=False, indent=2), flush=True)
    print(f"Results: {run_dir}")


def checkpoint_command(checkpoint, output, argv):
    """Forward evaluation options while replacing checkpoint selection and output."""
    forwarded = []
    tokens = iter(argv)
    for token in tokens:
        if token.split("=", 1)[0] in {"--checkpoint-dir", "--checkpoint", "--output"}:
            if "=" not in token:
                next(tokens)
        else:
            forwarded.append(token)
    # GRPO training explicitly disables thinking by default. An explicit eval
    # override is forwarded unchanged for experiments trained with thinking on.
    if not any(token.split("=", 1)[0] == "--chat-template-kwargs" for token in forwarded):
        forwarded.extend(["--chat-template-kwargs", '{"enable_thinking":false}'])
    return [sys.executable, str(Path(__file__).resolve()), *forwarded,
            "--model", str(checkpoint), "--output", str(output)]


def evaluate_checkpoints(a, argv):
    """Use a fresh process per checkpoint so vLLM/CUDA memory is fully released."""
    run_dir = a.output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = {"checkpoint_dir": str(a.checkpoint_dir), "checkpoints": {}}
    failed = []
    for checkpoint in a.checkpoint_paths:
        output = run_dir / checkpoint.name
        print(f"Evaluating {checkpoint.name} ({checkpoint})", flush=True)
        result = subprocess.run(checkpoint_command(checkpoint, output, argv), check=False)
        summaries = sorted(output.glob("*/summary.json"))
        entry = {"model": str(checkpoint), "step": int(checkpoint.name.split("-")[-1]),
                 "returncode": result.returncode}
        if result.returncode == 0 and summaries:
            entry["result_dir"] = str(summaries[-1].parent)
            entry["metrics"] = json.loads(summaries[-1].read_text(encoding="utf-8"))
        else:
            entry["error"] = "Evaluation failed or did not produce summary.json"
            failed.append(checkpoint.name)
        summary["checkpoints"][checkpoint.name] = entry
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Checkpoint comparison: {run_dir / 'summary.json'}", flush=True)
    if failed:
        raise RuntimeError(f"Failed checkpoints: {', '.join(failed)}; see {run_dir / 'summary.json'}")


def main():
    argv = sys.argv[1:]
    a = arguments(argv)
    if a.checkpoint_paths:
        evaluate_checkpoints(a, argv)
    else:
        evaluate(a)


if __name__ == "__main__":
    main()
