"""vLLM rollouts on the inference GPU using the current trainable student's weights."""

from functools import partial


def reset_prefix_cache(client):
    """vLLM may acknowledge cache reset with an empty body rather than JSON."""
    response = client.session.post(f"{client.base_url}/reset_prefix_cache")
    response.raise_for_status()


class StudentGeneration:
    def __init__(self, model, accelerator, tokenizer, args, state=None):
        from trl.generation.vllm_generation import VLLMGeneration

        eos_ids = model.generation_config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        self.pad_token_id = tokenizer.pad_token_id
        self.num_generations = getattr(args, "num_generations", 1)
        self.state = state
        self.seed = getattr(args, "seed", 42)
        self.backend = VLLMGeneration(
            model=model,
            accelerator=accelerator,
            processing_class=tokenizer,
            mode="server",
            server_base_url=args.student_vllm_server_url,
            group_port=args.student_vllm_group_port,
            server_timeout=60.0,
            enable_sleep_mode=False,
            temperature=args.temperature,
            top_p=getattr(args, "top_p", 1.0),
            top_k=-1,
            max_completion_length=args.max_new_tokens,
            logprobs=None,
            generation_kwargs={"detokenize": False, "stop_token_ids": eos_ids or [], "add_special_tokens": False},
        )
        # Patch only this acknowledgement endpoint on this client; JSON generation responses
        # and weight-transfer responses still use TRL's original parsing/error handling.
        client = self.backend.vllm_client
        client.reset_prefix_cache = partial(reset_prefix_cache, client)

    def generate(self, inputs):
        import torch

        if self.num_generations > 1:
            inputs = {
                "prompts": inputs["prompts"].repeat_interleave(self.num_generations, dim=0),
                "prompt_attention_mask": inputs["prompt_attention_mask"].repeat_interleave(self.num_generations, dim=0),
            }
        prompt_ids = [
            ids[mask.bool()].tolist()
            for ids, mask in zip(inputs["prompts"], inputs["prompt_attention_mask"])
        ]
        with torch.no_grad():
            # Always copy live weights before generation, including the initial/resumed rollout.
            # TRL merges LoRA, streams GPU0 weights to GPU1 through NCCL, then unmerges
            # the trainable model and resets prefix caches before accepting new prompts.
            self.backend.sync_weights()
            if self.state is not None:
                # vLLM owns an independent RNG: a saved request counter makes resumed sampling reproducible.
                self.backend.generation_kwargs["seed"] = (self.seed + self.state.rollout_calls) % (2**31)
            _, completion_ids, _, _ = self.backend.generate(
                prompts=prompt_ids, images=None, num_generations=self.num_generations,
            )
        packed = pack_completions(inputs, completion_ids, self.pad_token_id)
        if self.state is not None:
            self.state.rollout_calls += 1
        return packed

    def close(self):
        # The shell owns the server process; training only closes its NCCL connection.
        if self.backend is not None:
            self.backend.vllm_client.close_communicator()
            self.backend = None


def pack_completions(inputs, completions, pad_token_id):
    """Keep left-padded prompts; right-pad completions and mask by returned lengths."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    prompts = inputs["prompts"]
    if len(completions) != len(prompts) or any(not ids for ids in completions):
        raise ValueError("vLLM must return one nonempty completion per repeated prompt")
    tensors = [torch.tensor(ids, dtype=prompts.dtype, device=prompts.device) for ids in completions]
    completion_ids = pad_sequence(tensors, batch_first=True, padding_value=pad_token_id)
    completion_mask = pad_sequence(
        [torch.ones_like(ids) for ids in tensors], batch_first=True, padding_value=0,
    )
    input_ids = torch.cat((prompts, completion_ids), dim=1)
    attention_mask = torch.cat((inputs["prompt_attention_mask"], completion_mask), dim=1)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    labels[:, :prompts.shape[1]] = -100
    return input_ids, attention_mask, labels, completion_ids
