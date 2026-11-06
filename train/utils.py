"""Prompt-only data preparation shared by student rollouts and both teachers."""

from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence

from .direct_opd import PROMPT_TEMPLATE


def prepare_messages(example):
    question = example.get("prompt", example.get("problem"))
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Every row needs a nonempty prompt or problem string")
    return {"message": [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question.strip())}]}


def prompt_token_ids(tokenizer, messages, enable_thinking=False):
    rendered = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking,
    )
    return tokenizer(rendered, add_special_tokens=False)["input_ids"]


@dataclass
class myDataCollator:
    tokenizer: object
    max_length: int | None = None
    enable_thinking: bool = False

    def __call__(self, features):
        sequences = [prompt_token_ids(self.tokenizer, feature["message"], self.enable_thinking)
                     for feature in features]
        if any(not ids or (self.max_length is not None and len(ids) > self.max_length) for ids in sequences):
            raise ValueError("Empty or overlong prompt; filter the dataset before training instead of truncating")
        ids = [torch.tensor(row, dtype=torch.long) for row in sequences]
        input_ids = pad_sequence(ids, batch_first=True, padding_value=self.tokenizer.pad_token_id, padding_side="left")
        mask = pad_sequence([torch.ones_like(row) for row in ids], batch_first=True, padding_value=0, padding_side="left")
        return {"input_ids": input_ids, "attention_mask": mask,
                "prompts": input_ids, "prompt_attention_mask": mask}
