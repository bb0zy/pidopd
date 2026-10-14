"""CPU checks for teacher alignment, full-vocabulary normalization and IPC payloads."""

import importlib.util
import os
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from train.teacher_client import TeacherClient, _run_teacher, score_candidates


class ArrayTensor(np.ndarray):
    """Small NumPy stand-in for checking indexing without a local Torch install."""
    def float(self):
        return self.astype(np.float32)

    def gather(self, dim, indices):
        return np.take_along_axis(self, indices, axis=dim).view(ArrayTensor)

    def logsumexp(self, dim, keepdim):
        return np.logaddexp.reduce(self, axis=dim, keepdims=keepdim).view(ArrayTensor)

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


def array_tensor(data, dtype=None, device=None):
    return np.asarray(data, dtype=dtype).view(ArrayTensor)


ARRAY_TORCH = SimpleNamespace(
    tensor=array_tensor, long=np.int64,
    ones=lambda shape, dtype, device: np.ones(shape, dtype=dtype).view(ArrayTensor),
    cat=lambda tensors, dim: np.concatenate(tensors, axis=dim).view(ArrayTensor),
    inference_mode=nullcontext,
    from_numpy=np.asarray,
)


class PrefixModel:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, attention_mask, past_key_values, use_cache, return_dict):
        previous = [] if past_key_values is None else past_key_values
        ids = previous + input_ids[0].tolist()
        assert attention_mask.shape == (1, len(ids))
        assert use_cache and return_dict
        self.calls.append(input_ids.shape[1])
        # The score depends on every previous token, so dropping/resetting the cache fails.
        prefix = np.cumsum(np.asarray(ids))[-input_ids.shape[1]:]
        logits = np.sin(prefix[:, None] * 0.3 + np.arange(7)[None, :])
        return SimpleNamespace(logits=array_tensor(logits[None]), past_key_values=ids)


class TeacherTests(unittest.TestCase):
    def test_chunking_matches_full_sequence_reference(self):
        token_ids = [[1, 2, 3, 4, 5], [4, 3, 2, 1, 0, 2], [2, 1]]
        prompt_lengths = [2, 5, 1]
        candidates = [[[6, 1], [0, 4], [3, 5]], [[2, 0]], [[1, 6]]]
        expected = []
        for ids, prompt, selected in zip(token_ids, prompt_lengths, candidates):
            logits = np.sin(np.cumsum(ids[:-1])[:, None] * 0.3 + np.arange(7)[None, :])
            logprobs = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
            expected.extend(np.take_along_axis(logprobs[prompt - 1:], np.array(selected), axis=-1))
        for chunk_size in (1, 2, 3, 16):
            with self.subTest(chunk_size=chunk_size), patch.dict("sys.modules", {"torch": ARRAY_TORCH}):
                model = PrefixModel()
                actual = score_candidates(model, token_ids, prompt_lengths, candidates, 8192, chunk_size)
                np.testing.assert_allclose(actual, expected, atol=1e-6)
                self.assertEqual(actual.shape, (5, 2))
                self.assertEqual(actual.dtype, np.float32)
                self.assertLessEqual(max(model.calls), chunk_size)
                self.assertEqual(sum(model.calls), sum(len(ids) - 1 for ids in token_ids))

    def test_context_budget_rejects_instead_of_truncating(self):
        with patch.dict("sys.modules", {"torch": ARRAY_TORCH}):
            with self.assertRaisesRegex(ValueError, "teacher_max_model_len=2"):
                score_candidates(PrefixModel(), [[1, 2, 3]], [2], [[[0, 1]]], 2, 2)

    def test_worker_loads_once_and_returns_only_candidates(self):
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        loader = SimpleNamespace(from_pretrained=Mock(return_value=model))
        request = ("teacher", [[1, 2, 3]], [2], [[[4, 0]]])
        connection = Mock()
        connection.recv.side_effect = [request, request, None]
        small_scores = array_tensor([[-1, -2]], dtype=np.float32)
        with (
            patch.dict("sys.modules", {"transformers": SimpleNamespace(AutoModelForCausalLM=loader)}),
            patch.dict(os.environ),
            patch("train.teacher_client.score_candidates", return_value=small_scores) as scorer,
        ):
            _run_teacher(connection, "1", "/models/teacher", None, 8192, 256)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1")
        loader.from_pretrained.assert_called_once_with("/models/teacher", local_files_only=True, dtype="auto")
        model.to.assert_called_once_with("cuda")
        model.requires_grad_.assert_called_once_with(False)
        self.assertEqual(scorer.call_count, 2)
        np.testing.assert_array_equal(connection.send.call_args.args[0][1], [[-1, -2]])
        connection.close.assert_called_once()

    def test_worker_propagates_load_error(self):
        loader = SimpleNamespace(from_pretrained=Mock(side_effect=ValueError("bad model")))
        connection = Mock()
        with patch.dict("sys.modules", {"transformers": SimpleNamespace(AutoModelForCausalLM=loader)}), patch.dict(os.environ):
            _run_teacher(connection, "1", "/missing", None, 8192, 256)
        success, error = connection.send.call_args.args[0]
        self.assertFalse(success)
        self.assertIn("bad model", error)

    def test_client_receives_compact_float32_array(self):
        client = TeacherClient.__new__(TeacherClient)
        client.connection = Mock()
        client.connection.recv.return_value = (True, np.array([[-1, -2]], dtype=np.float32))
        with patch.dict("sys.modules", {"torch": ARRAY_TORCH}):
            scores = client.score([[1, 2]], [1], [[[0, 3]]])
        self.assertEqual(scores.shape, (1, 2))
        self.assertEqual(scores.dtype, np.float32)

    def test_frozen_base_routes_separately_from_teacher(self):
        teacher = Mock()
        reference = Mock()
        for model in (teacher, reference):
            model.to.return_value = model
            model.eval.return_value = model
        loader = SimpleNamespace(from_pretrained=Mock(side_effect=[teacher, reference]))
        inputs = ([[1, 2, 3]], [2], [[[4, 0]]])
        connection = Mock()
        connection.recv.side_effect = [("teacher", *inputs), ("reference", *inputs), None]
        with (
            patch.dict("sys.modules", {"transformers": SimpleNamespace(AutoModelForCausalLM=loader)}),
            patch.dict(os.environ),
            patch("train.teacher_client.score_candidates", return_value=array_tensor([[-1, -2]], dtype=np.float32)) as scorer,
        ):
            _run_teacher(connection, "1", "/models/teacher", "/models/student", 8192, 256)
        self.assertEqual(
            [call.args[0] for call in loader.from_pretrained.call_args_list],
            ["/models/teacher", "/models/student"],
        )
        teacher.requires_grad_.assert_called_once_with(False)
        reference.requires_grad_.assert_called_once_with(False)
        self.assertIs(scorer.call_args_list[0].args[0], teacher)
        self.assertIs(scorer.call_args_list[1].args[0], reference)

        client = TeacherClient.__new__(TeacherClient)
        client.connection = Mock()
        client.connection.recv.return_value = (True, np.array([[-1, -2]], dtype=np.float32))
        with patch.dict("sys.modules", {"torch": ARRAY_TORCH}):
            client.score_reference(*inputs)
        self.assertEqual(client.connection.send.call_args.args[0][0], "reference")


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
    "Requires installed Torch and Transformers; runs on CPU without model downloads",
)
class QwenTeacherTests(unittest.TestCase):
    def test_cached_chunks_match_qwen_full_forward(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM

        torch.manual_seed(7)
        config = Qwen3Config(
            vocab_size=23, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
        )
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config).eval().requires_grad_(False)
        ids = [[1, 4, 2, 7, 3], [2, 5, 1, 8, 4, 6], [1, 2]]
        lengths = [2, 5, 1]
        candidates = [[[1, 7], [2, 9], [4, 0]], [[6, 3]], [[22, 0]]]
        expected = []
        with torch.inference_mode():
            for sequence, prompt, selected in zip(ids, lengths, candidates):
                logits = model(torch.tensor([sequence]), use_cache=False).logits[0, prompt - 1:-1].float()
                expected.append(logits.log_softmax(-1).gather(-1, torch.tensor(selected)))
        expected = torch.cat(expected)
        for size in (1, 3, 256):
            actual = score_candidates(model, ids, lengths, candidates, 8192, size)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            self.assertFalse(actual.requires_grad)
            self.assertEqual(actual.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
