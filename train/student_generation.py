"""Colocated vLLM rollouts using the current trainable student's weights."""


class StudentGeneration:
    def __init__(self, model, accelerator, tokenizer, args):
        from trl.generation.vllm_generation import VLLMGeneration

        eos_ids = model.generation_config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        self.pad_token_id = tokenizer.pad_token_id
        self.backend = VLLMGeneration(
            model=model,
            accelerator=accelerator,
            processing_class=tokenizer,
            mode="colocate",
            tensor_parallel_size=1,
            gpu_memory_utilization=args.student_vllm_gpu_memory_utilization,
            max_model_length=args.max_length + args.max_new_tokens,
            max_num_seqs=args.per_device_train_batch_size,
            enable_sleep_mode=True,
            temperature=args.temperature,
            top_p=1.0,
            top_k=-1,
            max_completion_length=args.max_new_tokens,
            logprobs=None,
            generation_kwargs={"detokenize": False, "stop_token_ids": eos_ids or []},
        )

    def generate(self, inputs):
        import torch

        prompt_ids = [
            ids[mask.bool()].tolist()
            for ids, mask in zip(inputs["prompts"], inputs["prompt_attention_mask"])
        ]
        with torch.no_grad():
            # Always copy live weights before generation, including the initial/resumed rollout.
            # TRL merges LoRA for the copy, unmerges the trainable model and resets prefix caches.
            # Sleep level 2 discards vLLM weights, so re-sync even within an accumulation step.
            self.backend.sync_weights()
            _, completion_ids, _, _ = self.backend.generate(
                prompts=prompt_ids, images=None, num_generations=1,
            )
            # TRL sleeps the engine after generation, freeing its weights/KV cache for training.
        return pack_completions(inputs, completion_ids, self.pad_token_id)


def pack_completions(inputs, completions, pad_token_id):
    """Keep left-padded prompts; right-pad completions and mask by returned lengths."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    prompts = inputs["prompts"]
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
