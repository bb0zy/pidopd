"""Client for a separately started vLLM OpenAI-compatible server."""

import torch
from openai import OpenAI


class VLLMClient:
    def __init__(self, base_url, model="teacher", api_key="EMPTY"):
        self.model = model
        self.client = OpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout=1800.0,
            max_retries=0,
        )

    def generate(self, prompts, max_new_tokens=2048, temperature=0.9):
        """Generate from token ID prompts; return the server response."""
        return self.client.completions.create(
            model=self.model,
            prompt=prompts,
            max_tokens=max_new_tokens,
            temperature=temperature,
            extra_body={"add_special_tokens": False, "return_token_ids": True},
        )

    def score(self, token_ids):
        """Return a list of CPU float32 tensors with shape (sequence_length - 1, vocab_size).

        Row t contains log probabilities for input position t + 1; columns are token IDs.
        The first input token has no preceding context, so its undefined scores are omitted.
        """
        response = self.client.completions.create(
            model=self.model,
            prompt=token_ids,
            max_tokens=0,
            echo=True,
            logprobs=0,
            extra_body={
                "prompt_logprobs": -1,
                "add_special_tokens": False,
                "return_token_ids": True,
            },
        )

        log_probabilities = []
        for choice in sorted(response.choices, key=lambda choice: choice.index):
            prompt_logprobs = choice.model_extra["prompt_logprobs"][1:]
            vocab_size = len(prompt_logprobs[0])
            log_probs = torch.empty((len(prompt_logprobs), vocab_size), dtype=torch.float32, device="cpu")
            for position, token_logprobs in enumerate(prompt_logprobs):
                values = [token_logprobs[str(token_id)]["logprob"] for token_id in range(vocab_size)]
                log_probs[position] = torch.tensor(values, dtype=torch.float32, device="cpu")
            log_probabilities.append(log_probs)
        return log_probabilities

    def close(self):
        self.client.close()
