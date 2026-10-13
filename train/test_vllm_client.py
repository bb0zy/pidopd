"""CPU-only checks for offline teacher score alignment and result selection."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from train.vllm_client import VLLMClient, _run_teacher, select_candidate_logprobs


class FlatScores:
    def __init__(self, dense):
        self.start_indices = [0]
        self.end_indices = [0]
        self.token_ids = []
        self.logprobs = []
        for row in dense[1:]:
            # Deliberately vary rank order and duplicate the actual input token.
            ranked = np.argsort(row)[::-1].tolist()
            ids = [ranked[-1], *ranked]
            self.start_indices.append(len(self.token_ids))
            self.token_ids.extend(ids)
            self.logprobs.extend(float(row[token_id]) for token_id in ids)
            self.end_indices.append(len(self.token_ids))

    def __getitem__(self, index):
        raise AssertionError("Selection must not rebuild full-vocabulary dictionaries")


class OfflineTeacherTests(unittest.TestCase):
    def test_candidate_alignment_and_two_teacher_difference(self):
        rng = np.random.default_rng(42)
        candidates = np.array([[6, 1, 4], [2, 5, 0]])
        selected = []
        expected = []
        for prompt_length in (2, 5):
            logits = rng.normal(size=(prompt_length + 2, 7))
            dense = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
            actual = select_candidate_logprobs(
                FlatScores(dense), prompt_length, candidates.tolist(),
            )
            reference = np.take_along_axis(dense[prompt_length:], candidates, axis=-1)
            np.testing.assert_allclose(actual, reference)
            selected.append(np.asarray(actual))
            expected.append(reference)
        np.testing.assert_allclose(selected[1] - selected[0], expected[1] - expected[0])

    def test_worker_reuses_model_and_sends_only_candidates(self):
        dense = np.array([[0, 0, 0], [-1, -2, -3], [-4, -5, -6], [-7, -8, -9]])
        output = SimpleNamespace(prompt_logprobs=FlatScores(dense))
        llm = Mock()
        llm.generate.return_value = [output, output]
        fake_vllm = SimpleNamespace(LLM=Mock(return_value=llm), SamplingParams=Mock())
        request = ([[0, 1, 2, 0], [2, 1, 0, 2]], [2, 3], [[[2, 0], [1, 2]], [[0, 2]]])
        connection = Mock()
        connection.recv.side_effect = [request, request, None]
        with patch.dict("sys.modules", {"vllm": fake_vllm}), patch.dict(os.environ):
            _run_teacher(connection, "1", "/models/teacher", {"skip_tokenizer_init": True})
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1")
        fake_vllm.LLM.assert_called_once_with(model="/models/teacher", skip_tokenizer_init=True)
        params = fake_vllm.SamplingParams.call_args.kwargs
        self.assertTrue(params["flat_logprobs"])
        self.assertFalse(params["detokenize"])
        self.assertEqual(llm.generate.call_count, 2)
        self.assertEqual(connection.send.call_args_list[1].args[0], (
            True, [[[-6.0, -4.0], [-8.0, -9.0]], [[-7.0, -9.0]]],
        ))
        llm.llm_engine.engine_core.shutdown.assert_called_once()
        connection.close.assert_called_once()

    def test_worker_reports_loading_errors(self):
        fake_vllm = SimpleNamespace(LLM=Mock(side_effect=ValueError("bad model")), SamplingParams=Mock())
        connection = Mock()
        with patch.dict("sys.modules", {"vllm": fake_vllm}), patch.dict(os.environ):
            _run_teacher(connection, "1", "/missing", {})
        success, message = connection.send.call_args.args[0]
        self.assertFalse(success)
        self.assertIn("bad model", message)
        connection.close.assert_called_once()

    def test_client_creates_only_candidate_tensor(self):
        client = VLLMClient.__new__(VLLMClient)
        client.connection = Mock()
        client.connection.recv.return_value = (True, [[[-1.0, -2.0]], [[-3.0, -4.0], [-5.0, -6.0]]])
        fake_torch = SimpleNamespace(tensor=Mock(side_effect=np.array), float32=np.float32)
        with patch.dict("sys.modules", {"torch": fake_torch}):
            scores = client.score([[0, 1], [0, 2, 1]], [1, 1], [[[2, 1]], [[0, 2], [1, 0]]])
        self.assertEqual(scores.shape, (3, 2))
        self.assertEqual(scores.dtype, np.float32)
        np.testing.assert_array_equal(scores, [[-1, -2], [-3, -4], [-5, -6]])


if __name__ == "__main__":
    unittest.main()
