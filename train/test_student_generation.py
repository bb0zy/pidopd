"""CPU checks for rollout weight freshness and token/mask alignment."""

import importlib.util
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from train.student_generation import StudentGeneration, pack_completions


class Tensor(np.ndarray):
    device = "cpu"

    def bool(self):
        return self.astype(bool)

    def clone(self):
        return self.copy()


def tensor(data, dtype=None, device=None):
    return np.array(data, dtype=dtype).view(Tensor)


def pad_sequence(tensors, batch_first, padding_value):
    result = np.full((len(tensors), max(map(len, tensors))), padding_value, dtype=np.int64)
    for row, ids in zip(result, tensors):
        row[:len(ids)] = ids
    return result.view(Tensor)


ARRAY_TORCH = SimpleNamespace(
    tensor=tensor, no_grad=nullcontext, ones_like=np.ones_like,
    cat=lambda arrays, dim: np.concatenate(arrays, axis=dim).view(Tensor),
)


class StudentGenerationTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            student_vllm_gpu_memory_utilization=0.5, max_length=1024,
            max_new_tokens=2048, per_device_train_batch_size=32, temperature=0.9,
        )
        self.model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[9, 10]), version=0)
        self.backend = Mock()
        self.factory = Mock(return_value=self.backend)
        self.addCleanup(patch.stopall)
        patch.dict("sys.modules", {
            "trl.generation.vllm_generation": SimpleNamespace(VLLMGeneration=self.factory),
            "torch": ARRAY_TORCH,
            "torch.nn.utils.rnn": SimpleNamespace(pad_sequence=pad_sequence),
        }).start()
        self.generator = StudentGeneration(self.model, Mock(), SimpleNamespace(pad_token_id=9), self.args)
        self.inputs = {
            # A real pad/EOS token inside the prompt must survive unpadding.
            "prompts": tensor([[9, 9, 3], [4, 9, 5]]),
            "prompt_attention_mask": tensor([[0, 1, 1], [1, 1, 1]]),
        }

    def test_batched_configuration_and_stop_tokens(self):
        options = self.factory.call_args.kwargs
        self.assertIs(options["model"], self.model)
        self.assertEqual(options["max_num_seqs"], 32)
        self.assertEqual(options["max_model_length"], 3072)
        self.assertEqual(options["mode"], "colocate")
        self.assertTrue(options["enable_sleep_mode"])
        self.assertIsNone(options["logprobs"])
        self.assertEqual(options["generation_kwargs"], {"detokenize": False, "stop_token_ids": [9, 10]})

    def test_initial_updated_and_resumed_weights_synced_before_each_rollout(self):
        loaded = []

        def sync():
            loaded.append(self.model.version)

        def generate(**kwargs):
            self.assertEqual(loaded[-1], self.model.version)
            self.assertEqual(kwargs["prompts"], [[9, 3], [4, 9, 5]])
            self.assertEqual(kwargs["num_generations"], 1)
            return kwargs["prompts"], [[7, 9], [8]], None, None

        self.backend.sync_weights.side_effect = sync
        self.backend.generate.side_effect = generate
        for version in (0, 1, 2, 50, 50):
            self.model.version = version
            self.generator.generate(self.inputs)
        self.assertEqual(loaded, [0, 1, 2, 50, 50])
        self.assertEqual(
            [call[0] for call in self.backend.mock_calls],
            ["sync_weights", "generate"] * 5,
        )

    def test_padding_preserves_real_eos_and_masks_only_padding(self):
        ids, mask, labels, completion = pack_completions(self.inputs, [[7, 9], [10]], 9)
        np.testing.assert_array_equal(ids, [[9, 9, 3, 7, 9], [4, 9, 5, 10, 9]])
        np.testing.assert_array_equal(mask, [[0, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
        np.testing.assert_array_equal(labels, [[-100, -100, -100, 7, 9], [-100, -100, -100, 10, -100]])
        np.testing.assert_array_equal(completion, [[7, 9], [10, 9]])
        self.assertEqual((labels[:, 1:] != -100).sum(), 3)

    def test_failed_sync_does_not_generate_with_stale_weights(self):
        self.backend.sync_weights.side_effect = RuntimeError("sync failed")
        with self.assertRaisesRegex(RuntimeError, "sync failed"):
            self.generator.generate(self.inputs)
        self.backend.generate.assert_not_called()


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("torch", "trl", "peft")),
    "Requires Torch, TRL and PEFT; tests weight copying on CPU without vLLM or model downloads",
)
class LoraSyncTests(unittest.TestCase):
    def test_trl_copies_updated_merged_lora_and_restores_trainable_model(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from trl.generation.vllm_generation import VLLMGeneration

        config = Qwen3Config(
            vocab_size=23, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        )
        model = get_peft_model(Qwen3ForCausalLM(config), LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"]))
        backend = VLLMGeneration.__new__(VLLMGeneration)
        backend.model = model
        backend.mode = "colocate"
        backend.enable_sleep_mode = True
        backend.accelerator = SimpleNamespace()
        backend._dist = SimpleNamespace(is_fsdp=False, gather_params=lambda _: nullcontext())
        backend.llm = Mock()
        copied = {}
        backend.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights.side_effect = (
            lambda pairs: copied.update((name, weight.clone()) for name, weight in pairs)
        )
        layer = model.base_model.model.model.layers[0].self_attn.q_proj
        original = layer.base_layer.weight.detach().clone()
        # Float32 toy weights make merge/unmerge comparisons insensitive to BF16 roundoff.
        with torch.no_grad():
            for value in (0.1, 0.2):
                layer.lora_A["default"].weight.fill_(0.1)
                layer.lora_B["default"].weight.fill_(value)
                expected = layer.base_layer.weight + layer.get_delta_weight("default")
                backend.sync_weights()
                torch.testing.assert_close(copied["model.layers.0.self_attn.q_proj.weight"], expected)
                torch.testing.assert_close(layer.base_layer.weight, original)
                self.assertFalse(layer.merged)
                self.assertFalse(backend._llm_weights_sleeping)
        self.assertEqual(backend.llm.reset_prefix_cache.call_count, 2)
        backend.llm.wake_up.assert_called_with(tags=["weights"])


if __name__ == "__main__":
    unittest.main()
