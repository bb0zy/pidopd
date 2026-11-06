"""Direct-OPD objective and checkpoint state, independent of the training backend."""

import json
import math
from pathlib import Path

from .kd_data import SYSTEM_PROMPT


# Escape the instruction's literal braces before inserting the question with str.format.
PROMPT_TEMPLATE = SYSTEM_PROMPT.replace("{", "{{").replace("}", "}}") + "\n\n{question}"


def low_var_kl(student_logprob, reference_logprob):
    """Sampled k3 KL to the frozen initial student, as in the other implementation."""
    delta = (reference_logprob.detach() - student_logprob).clamp(-20, 20)
    return (delta.exp() - delta - 1).clamp(max=10)


def topk_policy_loss(student_logprobs, topk_logits, reward):
    """Shared student-top-k surrogate; gradients flow only through full-vocabulary log-probs."""
    import torch

    advantage = (torch.softmax(topk_logits, dim=-1) * reward).detach()
    policy_loss = -(advantage * student_logprobs).sum(dim=-1).mean()
    return policy_loss, advantage.mean().detach()


def direct_opd_loss(student_logprobs, topk_logits, after_logprobs, before_logprobs,
                    sampled_logprobs, reference_logprobs, alpha):
    """Inputs contain only answer positions; normalize student log-probs over the full vocabulary."""
    policy_loss, dense_reward = topk_policy_loss(student_logprobs, topk_logits, after_logprobs - before_logprobs)
    kl = low_var_kl(sampled_logprobs, reference_logprobs).mean()
    return policy_loss + alpha * kl, {
        "teacher_loss": policy_loss.detach(),
        "dense_reward": dense_reward,
        "kl_to_base": kl.detach(),
    }


def update_alpha(alpha, reward, epsilon, minimum, maximum):
    sign = 0 if reward == 0 else (1 if reward > 0 else -1)
    return min(maximum, max(minimum, alpha * (1 + epsilon * sign)))


def resolve_model_source(path):
    """Accept a local full model/checkpoint or a PEFT adapter with its local base model."""
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Local model/checkpoint directory not found: {path}")
    adapter_config = path / "adapter_config.json"
    if not adapter_config.is_file():
        return str(path), None
    config = json.loads(adapter_config.read_text())
    base = Path(config["base_model_name_or_path"]).expanduser()
    if not base.is_absolute() and not base.is_dir():
        base = path / base
    if not base.is_dir():
        raise FileNotFoundError(f"Adapter {path} needs its local base model: {base}")
    return str(base.resolve()), str(path)


def tokenizer_source(path):
    base, adapter = resolve_model_source(path)
    return adapter if adapter and (Path(adapter) / "tokenizer_config.json").is_file() else base


def load_frozen_model(path):
    from transformers import AutoModelForCausalLM

    base, adapter = resolve_model_source(path)
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, dtype="auto")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=False, local_files_only=True)
        model = model.merge_and_unload()
    return model.eval().requires_grad_(False)


def build_token_map(student_tokenizer, teacher_tokenizer):
    """Map identical token strings, including special tokens whose IDs differ between models."""
    other = teacher_tokenizer.get_vocab()
    return {index: other[token] for token, index in student_tokenizer.get_vocab().items() if token in other}


def remap_tokens(token_ids, candidate_ids, mapping):
    try:
        ids = [[mapping[token] for token in row] for row in token_ids]
        candidates = [[[mapping[token] for token in position] for position in row] for row in candidate_ids]
    except KeyError as error:
        raise ValueError(
            f"Student token ID {error.args[0]} has no identical token in the teacher vocabulary; "
            "use teachers with a compatible tokenizer"
        ) from error
    return ids, candidates


class DirectOPDState:
    """One alpha update per optimizer step; accumulates token-weighted microbatch rewards."""

    def __init__(self, alpha, signature):
        self.alpha = alpha
        self.signature = signature
        self.rollout_calls = 0
        self.reward_sum = 0.0
        self.reward_count = 0

    def observe(self, dense_reward, pair_count):
        self.reward_sum += float(dense_reward) * pair_count
        self.reward_count += pair_count

    def finish_step(self, adaptive, epsilon, minimum, maximum):
        if self.reward_count and adaptive:
            self.alpha = update_alpha(self.alpha, self.reward_sum / self.reward_count, epsilon, minimum, maximum)
        self.reward_sum = 0.0
        self.reward_count = 0

    def save(self, directory, step):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "direct_opd_state.json").write_text(json.dumps({
            "version": 1, "global_step": step, "alpha": self.alpha, "signature": self.signature,
            "rollout_calls": self.rollout_calls,
        }, indent=2) + "\n")

    def restore(self, directory):
        path = Path(directory) / "direct_opd_state.json"
        if not path.is_file():
            raise ValueError("Checkpoint has no Direct-OPD state; use it as --student-model-path to start a new run")
        saved = json.loads(path.read_text())
        if saved.get("version") != 1 or saved["signature"] != self.signature:
            raise ValueError("Resume configuration differs from the saved Direct-OPD run (models, data or objective)")
        alpha = float(saved["alpha"])
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("Checkpoint alpha must be finite and nonnegative")
        trainer_state = json.loads((Path(directory) / "trainer_state.json").read_text())
        if saved["global_step"] != trainer_state["global_step"]:
            raise ValueError("Direct-OPD state and Trainer checkpoint steps differ")
        self.alpha = alpha
        self.rollout_calls = int(saved["rollout_calls"])


def resolve_resume(output_dir, resume):
    output_dir = Path(output_dir)
    checkpoints = sorted((p for p in output_dir.glob("checkpoint-*")
                          if p.is_dir() and p.name.removeprefix("checkpoint-").isdigit()),
                         key=lambda p: int(p.name.removeprefix("checkpoint-")))
    if resume == "latest":
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint-* directory in {output_dir}")
        resume = checkpoints[-1]
    elif resume is None:
        if checkpoints or (output_dir / "final").exists():
            raise FileExistsError("Output already contains a run; use --resume-from-checkpoint latest or a new --output-dir")
        return None
    path = Path(resume).expanduser().resolve()
    required = ("adapter_config.json", "trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                "direct_opd_state.json")
    missing = [name for name in required if not (path / name).is_file()]
    if not any((path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        missing.append("adapter_model.safetensors or adapter_model.bin")
    if missing:
        raise ValueError(f"Incomplete Direct-OPD training checkpoint {path}: missing {', '.join(missing)}")
    return str(path)
