"""Utilities for preparing training data."""

from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase


def prepare_messages(example, stu_sys_prompt="", teacher_sys_prompt=""):
    teacher_prompt = (
        f"{teacher_sys_prompt}\n\n"
        f"Reference solution:\n{example['solution']}"
    ).strip()

    return {
        "message": [
            {"role": "system", "content": stu_sys_prompt},
            {"role": "user", "content": example["problem"]},
        ],
        "stu_sys_prompt": stu_sys_prompt,
        "teacher_sys_prompt": teacher_prompt,
        "Answer": example["Answer"],
    }


@dataclass
class myDataCollator:
    """Build student and teacher generation prompts with left padding.

    max_length only truncates student prompts. Teacher prompts retain the
    full privileged solution. Both sides use the same tokenizer.
    """

    tokenizer: PreTrainedTokenizerBase
    max_length: int | None = None

    def __call__(self, features):
        student_messages = []
        teacher_messages = []

        for feature in features:
            student_messages.append(feature["message"])

            teacher_messages.append(
                [{"role": "system", "content": feature["teacher_sys_prompt"]}]
                + feature["message"][1:]
            )

        input_ids, attention_mask = self._tokenize_and_pad(student_messages, self.max_length)
        teacher_ids, teacher_mask = self._tokenize_and_pad(teacher_messages)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompts": input_ids,
            "prompt_attention_mask": attention_mask,
            "teacher_prompts": teacher_ids,
            "teacher_prompt_attention_mask": teacher_mask,
            "answer": [feature["Answer"] for feature in features],
        }

    def _tokenize_and_pad(self, messages_batch, max_length=None):
        input_ids = []
        attention_mask = []

        for messages in messages_batch:
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            prompt_ids = self.tokenizer(
                formatted_prompt, add_special_tokens=False
            )["input_ids"]
            if max_length is not None:
                prompt_ids = prompt_ids[-max_length :]

            input_ids.append(prompt_ids)
            attention_mask.append([1] * len(prompt_ids))

        return (
            self._pad(input_ids, self.tokenizer.pad_token_id),
            self._pad(attention_mask, 0),
        )

    def _pad(self, sequences, padding_value):
        tensors = [torch.tensor(sequence, dtype=torch.long) for sequence in sequences]
        return pad_sequence(
            tensors,
            batch_first=True,
            padding_value=padding_value,
            padding_side="left",
        )
