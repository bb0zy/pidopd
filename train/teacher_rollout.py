r"""Stage 1: generate reusable teacher responses with vLLM (no student loaded).

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m train.teacher_rollout
    # Independent teacher replica per GPU (data parallel):
    CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m train.teacher_rollout --data-parallel-size 2
    # Multiple responses per question:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m train.teacher_rollout --num-rollouts 4

Default: one response/question, max_new_tokens=16384, natural EOS stopping.
JSONL stores exact prompt/completion IDs and text; metadata records truncation.
Use --resume to continue an interrupted generation with matching settings.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

from .kd_data import (DEFAULT_ROLLOUT_PATH, PROJECT_ROOT, file_sha256, metadata_path,
                      prepare_example, tokenizer_fingerprint)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher-model-path", type=Path, default=Path("/models/qwen3-4b-1"))
    parser.add_argument("--data-path", type=Path, default=PROJECT_ROOT / "dataset/train/dapo-17k.parquet")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_ROLLOUT_PATH)
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=32, help="Questions per generation call")
    parser.add_argument("--max-num-seqs", type=int, default=16, help="Concurrent vLLM sequences")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1,
                        help="Independent teacher replicas on this node; use Python, not torchrun")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Only process first N source questions for a smoke run")
    parser.add_argument("--resume", action="store_true")
    cli = parser.parse_args(argv)
    for name in ("num_rollouts", "max_new_tokens", "max_prompt_length", "max_model_len", "batch_size",
                 "max_num_seqs", "tensor_parallel_size", "data_parallel_size"):
        if getattr(cli, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if cli.limit is not None and cli.limit < 1:
        parser.error("--limit must be positive")
    if cli.max_prompt_length + cli.max_new_tokens > cli.max_model_len:
        parser.error("max-model-len must fit max-prompt-length + max-new-tokens")
    if not math.isfinite(cli.temperature) or cli.temperature <= 0:
        parser.error("temperature must be finite and positive")
    if not 0 < cli.top_p <= 1 or cli.top_k < 0 or not 0 < cli.gpu_memory_utilization < 1:
        parser.error("Require top-p in (0,1], top-k >= 0 and gpu-memory-utilization in (0,1)")
    return cli


def scan_existing(path, num_rollouts, repair_tail=False, exclude_keys=()):
    """Resume complete rows; repair only a torn last JSONL line after a crash."""
    seen, stopped, truncated = set(), 0, 0
    if not path.exists():
        return seen, stopped, truncated
    with path.open("r+b" if repair_tail else "rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if repair_tail and not line.endswith(b"\n") and not handle.read(1):
                    handle.seek(offset)
                    handle.truncate()
                    break
                raise ValueError(f"Invalid cached rollout at byte {offset}") from None
            if not line.endswith(b"\n"):
                if not repair_tail:
                    raise ValueError("Last cached row needs a newline; run with --resume")
                handle.seek(0, 2)
                handle.write(b"\n")
            key = (record["question_id"], record["sample_id"])
            if key in seen or key[0] < 0 or not 0 <= key[1] < num_rollouts:
                raise ValueError(f"Duplicate or invalid cached rollout key: {key}")
            seen.add(key)
            if key not in exclude_keys:
                stopped += int(record["finish_reason"] == "stop")
                truncated += int(record["finish_reason"] == "length")
    return seen, stopped, truncated


def scan_cache_files(paths, num_rollouts, repair_tail=False):
    seen, stopped, truncated = set(), 0, 0
    for path in paths:
        keys, finished, limited = scan_existing(path, num_rollouts, repair_tail, exclude_keys=seen)
        seen.update(keys)
        stopped += finished
        truncated += limited
    return seen, stopped, truncated


def merge_cache_files(output_path, paths, expected_responses):
    """Stream distinct rows into one atomic output; preserve old output on failure."""
    merged = set()
    temporary = output_path.with_suffix(".merge.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for path in paths:
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    key = (row["question_id"], row["sample_id"])
                    if key not in merged:
                        target.write(line)
                        merged.add(key)
        if len(merged) != expected_responses:
            raise ValueError(f"Incomplete rollout merge: {len(merged)}/{expected_responses}")
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(output_path)


def visible_device_groups(data_parallel_size, tensor_parallel_size):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        import torch
        devices = [str(index) for index in range(torch.cuda.device_count())]
    else:
        devices = [device.strip() for device in visible.split(",") if device.strip()]
    required = data_parallel_size * tensor_parallel_size
    if len(devices) < required or len(set(devices)) != len(devices) or "-1" in devices:
        raise ValueError(f"DP={data_parallel_size}, TP={tensor_parallel_size} needs {required} distinct visible GPUs")
    return [",".join(devices[index * tensor_parallel_size:(index + 1) * tensor_parallel_size])
            for index in range(data_parallel_size)]


def output_records(question, output, sample_ids):
    if len(output.outputs) != len(sample_ids):
        raise ValueError("vLLM returned an unexpected number of responses")
    if list(output.prompt_token_ids) != question["prompt_token_ids"]:
        raise ValueError("vLLM changed the cached prompt token IDs")
    return [{"question_id": question["question_id"], "sample_id": sample_id,
             "prompt": question["prompt"], "completion": completion.text,
             "prompt_token_ids": question["prompt_token_ids"],
             "completion_token_ids": list(completion.token_ids),
             "finish_reason": completion.finish_reason, "stop_reason": completion.stop_reason}
            for sample_id, completion in zip(sample_ids, sorted(output.outputs, key=lambda item: item.index), strict=True)]


def generate_responses(cli, teacher_path, pending, output_path, on_batch=None):
    from transformers import GenerationConfig
    from vllm import LLM, SamplingParams

    generation_path = teacher_path / "generation_config.json"
    eos = (GenerationConfig.from_pretrained(str(teacher_path), local_files_only=True).eos_token_id
           if generation_path.is_file() else json.loads((teacher_path / "config.json").read_text(encoding="utf-8")).get("eos_token_id"))
    if eos is None:
        from transformers import AutoTokenizer
        eos = AutoTokenizer.from_pretrained(str(teacher_path), local_files_only=True).eos_token_id
    stop_ids = eos if isinstance(eos, list) else [eos] if eos is not None else []
    engine = LLM(model=str(teacher_path), tokenizer=str(teacher_path), dtype="bfloat16",
                 tensor_parallel_size=cli.tensor_parallel_size, gpu_memory_utilization=cli.gpu_memory_utilization,
                 max_model_len=cli.max_model_len, max_num_seqs=cli.max_num_seqs,
                 enable_prefix_caching=True, generation_config="vllm", seed=cli.seed)
    written = 0
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        for start in range(0, len(pending), cli.batch_size):
            batch = pending[start:start + cli.batch_size]
            params = [SamplingParams(n=len(indices), max_tokens=cli.max_new_tokens,
                                     temperature=cli.temperature, top_p=cli.top_p, top_k=cli.top_k,
                                     stop_token_ids=stop_ids, seed=cli.seed + question["question_id"])
                      for question, indices in batch]
            outputs = engine.generate([{"prompt_token_ids": question["prompt_token_ids"]} for question, _ in batch],
                                      params, use_tqdm=True)
            if len(outputs) != len(batch):
                raise ValueError("vLLM returned an unexpected number of requests")
            records = []
            for (question, indices), output in zip(batch, outputs, strict=True):
                records.extend(output_records(question, output, indices))
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            written += len(records)
            if on_batch is not None:
                on_batch(records)
            print(f"{output_path.name}: saved {written} new responses", flush=True)


def rollout_worker(cli, teacher_path, pending, output_path, visible_devices):
    # Spawned interpreter: isolate the GPU(s) before importing Torch/vLLM.
    if os.name == "posix":
        os.setsid()
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    generate_responses(cli, teacher_path, pending, output_path)


def run_parallel(cli, teacher_path, pending, shard_dir):
    import multiprocessing

    groups = visible_device_groups(cli.data_parallel_size, cli.tensor_parallel_size)
    context = multiprocessing.get_context("spawn")
    processes = []
    completed = False
    shard_dir.mkdir(parents=True, exist_ok=True)
    try:
        for rank, devices in enumerate(groups):
            assigned = pending[rank::len(groups)]
            if not assigned:
                continue
            process = context.Process(target=rollout_worker,
                                      args=(cli, teacher_path, assigned, shard_dir / f"rank-{rank:05d}.jsonl", devices))
            process.start()
            processes.append(process)
        while any(process.is_alive() for process in processes):
            failed = [process for process in processes if process.exitcode not in (None, 0)]
            if failed:
                raise RuntimeError(f"Rollout worker exited with code {failed[0].exitcode}; rerun with --resume")
            time.sleep(0.2)
        for process in processes:
            process.join()
            if process.exitcode:
                raise RuntimeError(f"Rollout worker exited with code {process.exitcode}; rerun with --resume")
        completed = True
    finally:
        # On failure also stop GPU-engine children of a worker that already died.
        for process in processes:
            if not completed:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        if process.is_alive():
                            process.terminate()
                elif process.is_alive():
                    process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    if os.name == "posix":
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            process.kill()
                    else:
                        process.kill()
                    process.join()


def main():
    cli = parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError("Launch with Python and --data-parallel-size N; torchrun is not needed for independent replicas")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    import pyarrow.parquet as pq
    from transformers import AutoConfig, AutoTokenizer

    teacher_path = cli.teacher_model_path.expanduser().resolve()
    if not cli.data_path.is_file() or not (teacher_path / "config.json").is_file():
        raise FileNotFoundError("Local teacher config or input parquet does not exist")
    tokenizer = AutoTokenizer.from_pretrained(str(teacher_path), local_files_only=True)
    context = AutoConfig.from_pretrained(str(teacher_path), local_files_only=True).get_text_config().max_position_embeddings
    if cli.max_model_len > context:
        raise ValueError(f"max-model-len exceeds teacher context {context}")
    source_rows = pq.read_table(cli.data_path, columns=["prompt"]).to_pylist()
    questions = []
    for index, row in enumerate(source_rows[:cli.limit]):
        ids = tokenizer.apply_chat_template(prepare_example(row)["prompt"], tokenize=True,
                                            add_generation_prompt=True, enable_thinking=cli.enable_thinking)
        ids = ids if isinstance(ids, list) else ids["input_ids"]
        if len(ids) <= cli.max_prompt_length:
            questions.append({"question_id": index, "prompt": row["prompt"], "prompt_token_ids": ids})
    if not questions:
        raise ValueError("No questions remain after prompt length filtering")
    output_path = cli.output_path.expanduser().resolve()
    meta_path = metadata_path(output_path)
    signature = {"format_version": 1, "teacher_model_path": str(teacher_path),
                 "teacher_config_sha256": file_sha256(teacher_path / "config.json"),
                 "source_sha256": file_sha256(cli.data_path), "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
                 "prompt_tokens_sha256": hashlib.sha256(json.dumps(
                     [(question["question_id"], question["prompt_token_ids"]) for question in questions],
                     separators=(",", ":")).encode("utf-8")).hexdigest(),
                 "generation_config_sha256": file_sha256(teacher_path / "generation_config.json")
                     if (teacher_path / "generation_config.json").is_file() else None,
                 **{name: getattr(cli, name) for name in ("num_rollouts", "max_new_tokens", "max_prompt_length",
                    "max_model_len", "temperature", "top_p", "top_k", "enable_thinking", "seed", "limit")}}
    shard_dir = output_path.with_suffix(".parts")
    shard_paths = sorted(shard_dir.glob("rank-*.jsonl"))
    if output_path.exists() or meta_path.exists() or shard_paths:
        if not cli.resume:
            raise FileExistsError("Rollout output exists; use --resume or a different --output-path")
        if not meta_path.is_file() or json.loads(meta_path.read_text(encoding="utf-8"))["signature"] != signature:
            raise ValueError("Resume requires the same teacher, source, tokenizer and generation settings")
    elif cli.resume:
        raise FileNotFoundError("No rollout output metadata to resume")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen, stopped, truncated = scan_cache_files([output_path, *shard_paths], cli.num_rollouts, repair_tail=cli.resume)
    valid_ids = {question["question_id"] for question in questions}
    if any(qid not in valid_ids for qid, _ in seen):
        raise ValueError("Cached question IDs do not match this input selection")
    pending = [(question, [index for index in range(cli.num_rollouts)
                           if (question["question_id"], index) not in seen]) for question in questions]
    pending = [(question, indices) for question, indices in pending if indices]
    metadata = {"signature": signature, "complete": False, "source_questions": len(source_rows),
                "selected_questions": min(cli.limit or len(source_rows), len(source_rows)),
                "usable_questions": len(questions), "expected_responses": len(questions) * cli.num_rollouts,
                "data_parallel_size": cli.data_parallel_size, "tensor_parallel_size": cli.tensor_parallel_size}

    def save_metadata(complete=False):
        metadata.update(complete=complete, saved_responses=len(seen), stopped_responses=stopped,
                        truncated_responses=truncated, truncated_ratio=truncated / max(len(seen), 1))
        temp_path = meta_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp_path.replace(meta_path)

    save_metadata()
    if pending:
        if cli.data_parallel_size == 1:
            def on_batch(records):
                nonlocal stopped, truncated
                for record in records:
                    seen.add((record["question_id"], record["sample_id"]))
                    stopped += int(record["finish_reason"] == "stop")
                    truncated += int(record["finish_reason"] == "length")
                save_metadata()
            generate_responses(cli, teacher_path, pending, output_path, on_batch=on_batch)
        else:
            run_parallel(cli, teacher_path, pending, shard_dir)
            shard_paths = sorted(shard_dir.glob("rank-*.jsonl"))
            seen, stopped, truncated = scan_cache_files([output_path, *shard_paths], cli.num_rollouts)
    if len(seen) != metadata["expected_responses"]:
        raise ValueError("Generation is incomplete; rerun with --resume")
    if shard_paths:
        merge_cache_files(output_path, [output_path, *shard_paths], metadata["expected_responses"])
        for path in shard_paths:
            path.unlink()  # The atomic merged output now contains these rows.
    save_metadata(complete=True)
    print(f"Rollouts ready: {output_path}; stopped={stopped}, length-limited={truncated}. "
          "Align excludes length-limited answers without cropping them.", flush=True)


if __name__ == "__main__":
    main()
