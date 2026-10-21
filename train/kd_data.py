"""Shared token-level contract between offline teacher generation and KD."""

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROLLOUT_PATH = PROJECT_ROOT / "dataset/train/dapo-17k-teacher.jsonl"
SYSTEM_PROMPT = (
    "Solve the following math problem step by step.\n"
    r"The last line of your response should contain only the final answer enclosed in \boxed{}."
)


def prepare_example(example):
    question = example.get("prompt")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Every DAPO row needs a nonempty prompt")
    return {"prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user", "content": question.strip()}]}


def tokenizer_fingerprint(tokenizer):
    payload = {"vocab": tokenizer.get_vocab(), "bos": tokenizer.bos_token_id, "eos": tokenizer.eos_token_id}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata_path(path):
    return Path(path).with_suffix(".meta.json")


def alignment_example(example):
    """Keep exact generated IDs, including EOS; labels only define the KL mask."""
    prompt, completion = example["prompt_token_ids"], example["completion_token_ids"]
    if not prompt or not completion:
        raise ValueError("Cached rollout needs nonempty prompt and completion token IDs")
    if any(type(token) is not int or token < 0 for token in prompt + completion):
        raise ValueError("Cached token IDs must be nonnegative integers")
    return {"input_ids": prompt + completion, "labels": [-100] * len(prompt) + completion}


def usable_rollout(example, max_length):
    # Never shorten an answer or invent EOS for one that hit the generation cap.
    return (example["finish_reason"] == "stop" and bool(example["completion_token_ids"])
            and len(example["prompt_token_ids"]) + len(example["completion_token_ids"]) <= max_length)


def padded_batch(features, pad_token_id):
    """Right-pad full sequences; prompt/padding contribute no distillation loss."""
    if not features:
        raise ValueError("Empty KD batch")
    width = max(len(feature["input_ids"]) for feature in features)
    batch = {key: [] for key in ("input_ids", "attention_mask", "labels")}
    for feature in features:
        ids, labels = feature["input_ids"], feature["labels"]
        if len(ids) != len(labels):
            raise ValueError("Input IDs and labels must have the same length")
        padding = width - len(ids)
        batch["input_ids"].append(ids + [pad_token_id] * padding)
        batch["attention_mask"].append([1] * len(ids) + [0] * padding)
        batch["labels"].append(labels + [-100] * padding)
    return batch
