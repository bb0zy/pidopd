"""Frozen Transformers teacher in a GPU-isolated local process."""

import atexit
import multiprocessing as mp
import os
import traceback


def score_candidates(model, token_ids, prompt_lengths, candidate_ids, max_length, chunk_size):
    """Score student candidates with full-vocabulary normalization on the model device."""
    import torch

    scores = []
    with torch.inference_mode():
        for ids, prompt_length, candidates in zip(token_ids, prompt_lengths, candidate_ids):
            if len(ids) > max_length:
                raise ValueError(f"Teacher input has {len(ids)} tokens; teacher_max_model_len={max_length}.")
            input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
            candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=model.device)
            past_key_values = None
            chunks = []
            # Input position t predicts token t + 1. The final input token needs no forward pass.
            for start in range(0, len(ids) - 1, chunk_size):
                end = min(start + chunk_size, len(ids) - 1)
                output = model(
                    input_ids=input_ids[:, start:end],
                    attention_mask=torch.ones((1, end), dtype=torch.long, device=model.device),
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = output.past_key_values
                answer_start = max(start, prompt_length - 1)
                if answer_start < end:
                    logits = output.logits[0, answer_start - start:].float()
                    indices = candidate_tensor[answer_start + 1 - prompt_length:end + 1 - prompt_length]
                    # Normalize over the full vocabulary, not just the selected K candidates.
                    selected = logits.gather(-1, indices) - logits.logsumexp(dim=-1, keepdim=True)
                    chunks.append(selected)
                    del logits, selected
                del output
            scores.append(torch.cat(chunks, dim=0).cpu())
            del past_key_values, chunks
    return torch.cat(scores, dim=0)


def _run_teacher(connection, gpu, model_path, reference_model_path, max_length, chunk_size):
    # Set visibility before importing CUDA libraries in this fresh spawn process.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    try:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, dtype="auto",
        ).to("cuda").eval()
        model.requires_grad_(False)
        reference_model = None
        if reference_model_path is not None:
            reference_model = AutoModelForCausalLM.from_pretrained(
                reference_model_path, local_files_only=True, dtype="auto",
            ).to("cuda").eval()
            reference_model.requires_grad_(False)
        connection.send((True, None))
        while True:
            request = connection.recv()
            if request is None:
                break
            model_kind, token_ids, prompt_lengths, candidate_ids = request
            scores = score_candidates(
                reference_model if model_kind == "reference" else model,
                token_ids, prompt_lengths, candidate_ids, max_length, chunk_size,
            )
            # Send only [total_answer_tokens, K] as a compact CPU array, never full logits.
            connection.send((True, scores.numpy()))
    except EOFError:
        pass
    except Exception:
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()


class TeacherClient:
    def __init__(self, model_path, reference_model_path=None, gpu="1", max_length=8192, chunk_size=256):
        context = mp.get_context("spawn")
        self.connection, child_connection = context.Pipe()
        self.process = context.Process(
            target=_run_teacher,
            args=(child_connection, gpu, model_path, reference_model_path, max_length, chunk_size),
            name="transformers-teacher",
        )
        self.process.start()
        child_connection.close()
        atexit.register(self.close)
        self._receive()

    def _receive(self):
        success, result = self.connection.recv()
        if not success:
            raise RuntimeError(f"Transformers teacher failed:\n{result}")
        return result

    def _score(self, model_kind, token_ids, prompt_lengths, candidate_ids):
        """Return detached CPU float32 scores in batch, answer-position, candidate order."""
        import torch

        self.connection.send((model_kind, token_ids, prompt_lengths, candidate_ids))
        return torch.from_numpy(self._receive())

    def score(self, token_ids, prompt_lengths, candidate_ids):
        return self._score("teacher", token_ids, prompt_lengths, candidate_ids)

    def score_reference(self, token_ids, prompt_lengths, candidate_ids):
        return self._score("reference", token_ids, prompt_lengths, candidate_ids)

    def close(self):
        if self.connection.closed:
            return
        if self.process.is_alive():
            try:
                self.connection.send(None)
            except (BrokenPipeError, OSError):
                pass
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
        else:
            self.process.join()
        self.connection.close()
