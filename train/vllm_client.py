"""Offline vLLM teacher, isolated on its own GPU without an HTTP server."""

import atexit
import multiprocessing as mp
import os
import traceback

import numpy as np


def select_candidate_logprobs(prompt_logprobs, prompt_length, candidate_ids):
    """Read only the requested candidates from vLLM's FlatLogprobs buffers."""
    selected = []
    # FlatLogprobs retains the empty first row, so the first answer is at prompt_length.
    for position, candidates in enumerate(candidate_ids, start=prompt_length):
        start = prompt_logprobs.start_indices[position]
        end = prompt_logprobs.end_indices[position]
        token_ids = np.asarray(prompt_logprobs.token_ids[start:end], dtype=np.int64)
        # Entries are ranked, not ordered by token ID. The actual input token
        # can appear twice; both entries have the same log probability.
        offsets = np.empty(int(token_ids.max()) + 1, dtype=np.int64)
        offsets[token_ids] = np.arange(start, end)
        selected.append([prompt_logprobs.logprobs[offsets[token_id]] for token_id in candidates])
    return selected


def _run_teacher(connection, gpu, model, llm_kwargs):
    # spawn gives this process a fresh CUDA runtime; the student's device is unchanged.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    llm = None
    try:
        from vllm import LLM, SamplingParams

        llm = LLM(model=model, **llm_kwargs)
        sampling_params = SamplingParams(
            max_tokens=1,  # Offline generate requires a positive value; ignore this output token.
            temperature=0.0,
            prompt_logprobs=-1,
            detokenize=False,
            flat_logprobs=True,
        )
        connection.send((True, None))
        while True:
            request = connection.recv()
            if request is None:
                break
            token_ids, prompt_lengths, candidate_ids = request
            outputs = llm.generate(
                [{"prompt_token_ids": ids} for ids in token_ids],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            scores = [
                select_candidate_logprobs(output.prompt_logprobs, length, candidates)
                for output, length, candidates in zip(outputs, prompt_lengths, candidate_ids)
            ]
            del outputs
            # Only small [answer_length, K] lists cross the process boundary.
            connection.send((True, scores))
    except EOFError:
        pass
    except Exception:
        connection.send((False, traceback.format_exc()))
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
        connection.close()


class VLLMClient:
    def __init__(self, model, gpu="1", **llm_kwargs):
        context = mp.get_context("spawn")
        self.connection, child_connection = context.Pipe()
        self.process = context.Process(
            target=_run_teacher,
            args=(child_connection, gpu, model, llm_kwargs),
            name="vllm-teacher",
        )
        self.process.start()
        child_connection.close()
        atexit.register(self.close)
        self._receive()  # Surface model-loading errors before starting training.

    def _receive(self):
        success, result = self.connection.recv()
        if not success:
            raise RuntimeError(f"Offline vLLM teacher failed:\n{result}")
        return result

    def score(self, token_ids, prompt_lengths, candidate_ids):
        """Return CPU float32 [total_answer_tokens, K] scores in candidate order."""
        import torch

        self.connection.send((token_ids, prompt_lengths, candidate_ids))
        scores = self._receive()
        return torch.tensor(
            [row for sequence in scores for row in sequence], dtype=torch.float32,
        )

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
