"""CPU checks: python -B -m unittest train.test_align_train -v."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from train import align_train, kd_data


class FakeTokenizer:
    eos_token_id = 99
    bos_token_id = 1
    chat_template = "test-template"

    def get_vocab(self):
        return {"a": 1, "b": 2}

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False}
        assert messages[-1] == {"role": "user", "content": "Question"}
        return [10, 11, 12]

    def __call__(self, text, add_special_tokens):
        raise AssertionError("Dataset answers must not be tokenized")


class AlignDataTests(unittest.TestCase):
    def test_only_question_is_used_for_teacher_rollout(self):
        example = {"prompt": "Question", "solution": "not a reasoning trace",
                   "reward_model": {"ground_truth": "34"}}
        prepared = kd_data.prepare_example(example)
        self.assertEqual(set(prepared), {"prompt"})
        self.assertNotIn("34", str(prepared["prompt"]))
        self.assertNotIn("not a reasoning trace", str(prepared))
        self.assertEqual(prepared, kd_data.prepare_example({"prompt": "Question"}))

    def test_missing_data_fails_early(self):
        for example in ({}, {"prompt": ""}, {"prompt": "  "}, {"prompt": None}):
            with self.subTest(example=example), self.assertRaises(ValueError):
                kd_data.prepare_example(example)

    def test_full_response_including_reasoning_and_eos_is_supervised(self):
        example = {"prompt_token_ids": [10, 11], "completion_token_ids": [20, 21, 22, 99]}
        prepared = kd_data.alignment_example(example)
        self.assertEqual(prepared["input_ids"], [10, 11, 20, 21, 22, 99])
        self.assertEqual(prepared["labels"], [-100, -100, 20, 21, 22, 99])
        # Causal shift includes the first response token predicted from the prompt.
        self.assertEqual([i for i, token in enumerate(prepared["labels"][1:]) if token != -100], [1, 2, 3, 4])

    def test_padding_has_no_loss_and_eos_is_kept_when_used_as_pad(self):
        features = [kd_data.alignment_example({"prompt_token_ids": [10, 11], "completion_token_ids": [20, 21, 99]}),
                    kd_data.alignment_example({"prompt_token_ids": [12], "completion_token_ids": [22, 99]})]
        batch = kd_data.padded_batch(features, pad_token_id=99)
        self.assertEqual(batch["input_ids"][1], [12, 22, 99, 99, 99])
        self.assertEqual(batch["attention_mask"][1], [1, 1, 1, 0, 0])
        self.assertEqual(batch["labels"][1], [-100, 22, 99, -100, -100])
        self.assertEqual(set(batch), {"input_ids", "attention_mask", "labels"})

    def test_filter_removes_incomplete_or_overlong_answers_without_cropping(self):
        example = {"prompt_token_ids": [10, 11], "completion_token_ids": [20, 21, 99], "finish_reason": "stop"}
        self.assertTrue(kd_data.usable_rollout(example, 5))
        self.assertFalse(kd_data.usable_rollout(example, 4))
        self.assertFalse(kd_data.usable_rollout({**example, "finish_reason": "length"}, 100))
        self.assertFalse(kd_data.usable_rollout({**example, "completion_token_ids": []}, 100))
        self.assertEqual(example["completion_token_ids"], [20, 21, 99])

    def test_tokenizer_id_alignment_is_required(self):
        tokenizer = FakeTokenizer()
        align_train.validate_tokenizers(tokenizer, tokenizer)
        other = SimpleNamespace(get_vocab=lambda: {"a": 2, "b": 1})
        with self.assertRaisesRegex(ValueError, "token-to-ID"):
            align_train.validate_tokenizers(tokenizer, other)
        other = SimpleNamespace(get_vocab=tokenizer.get_vocab, bos_token_id=1, eos_token_id=98)
        with self.assertRaisesRegex(ValueError, "eos_token_id"):
            align_train.validate_tokenizers(tokenizer, other)

    def test_offline_parameter_defaults(self):
        cli = align_train.parse_args([])
        self.assertEqual(cli.student_model_path, "/models/qwen3-1.7b")
        self.assertEqual(cli.teacher_model_path, "/models/qwen3-4b-1")
        self.assertEqual(cli.output_dir, align_train.PROJECT_ROOT / "train/align_checkpoint")
        self.assertEqual(cli.num_train_epochs, 1.0)
        self.assertEqual(cli.per_device_train_batch_size * cli.gradient_accumulation_steps, 32)
        self.assertEqual(cli.data_path, kd_data.DEFAULT_ROLLOUT_PATH)
        self.assertEqual(cli.max_length, 32768)
        self.assertTrue(cli.use_liger_kernel)
        self.assertFalse(hasattr(cli, "max_new_tokens"))

    def test_metadata_rejects_unfinished_or_mismatched_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text("{}", encoding="utf-8")
            cache = root / "responses.jsonl"
            metadata = {"complete": True, "signature": {"format_version": 1,
                        "tokenizer_fingerprint": kd_data.tokenizer_fingerprint(FakeTokenizer()),
                        "teacher_config_sha256": kd_data.file_sha256(config)}}
            path = kd_data.metadata_path(cache)
            path.write_text(json.dumps(metadata), encoding="utf-8")
            self.assertEqual(align_train.validate_rollout_metadata(cache, root, FakeTokenizer()), metadata)
            metadata["complete"] = False
            path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "completion"):
                align_train.validate_rollout_metadata(cache, root, FakeTokenizer())
            metadata["complete"] = True
            metadata["signature"]["tokenizer_fingerprint"] = "wrong"
            path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "vocabulary"):
                align_train.validate_rollout_metadata(cache, root, FakeTokenizer())


class GKDObjectiveTests(unittest.TestCase):
    def test_fused_full_vocab_forward_kl_matches_reference_on_cuda(self):
        try:
            import torch
            from trl.experimental.gkd import GKDTrainer
            from trl.losses import FusedLinearJSDLoss
            from trl.import_utils import is_liger_kernel_available
        except ImportError:
            self.skipTest("Torch/TRL training dependencies are not installed")
        if not torch.cuda.is_available() or not is_liger_kernel_available():
            self.skipTest("CUDA and liger-kernel are required for the fused loss check")
        torch.manual_seed(42)
        # Different teacher/student hidden sizes, matching Qwen3-4B/1.7B usage.
        student_hidden = torch.randn(1, 5, 16, device="cuda", requires_grad=True)
        teacher_hidden = torch.randn(1, 5, 32, device="cuda", dtype=torch.bfloat16)
        student_weight = torch.randn(128, 16, device="cuda", requires_grad=True)
        teacher_weight = torch.randn(128, 32, device="cuda", dtype=torch.bfloat16)
        labels = torch.tensor([[-100, -100, 20, 21, 99]], device="cuda")
        fused = FusedLinearJSDLoss(beta=0.0, ignore_index=-100, compiled=False,
                                  weight_hard_loss=0.0, weight_soft_loss=1.0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = fused(student_input=student_hidden[:, :-1], student_weight=student_weight,
                           teacher_input=teacher_hidden[:, :-1], teacher_weight=teacher_weight,
                           true_labels=labels[:, 1:], student_bias=None, teacher_bias=None)
            student_logits = torch.nn.functional.linear(student_hidden[:, :-1], student_weight).float()
            teacher_logits = torch.nn.functional.linear(teacher_hidden[:, :-1], teacher_weight).float()
        expected = GKDTrainer.generalized_jsd_loss(student_logits, teacher_logits, labels[:, 1:], beta=0.0)
        torch.testing.assert_close(actual.float(), expected, atol=0.1, rtol=0.02)
        actual.backward()
        self.assertIsNotNone(student_weight.grad)
        self.assertIsNone(teacher_weight.grad)

    def test_offline_trainer_bypasses_all_rollout(self):
        try:
            from train.offline_gkd import OfflineGKDTrainer
            from trl.trainer.sft_trainer import SFTTrainer
        except ImportError:
            self.skipTest("Torch/TRL training dependencies are not installed")
        from unittest.mock import patch
        trainer = object.__new__(OfflineGKDTrainer)
        trainer.generate_on_policy_outputs = lambda *args: self.fail("Offline alignment must never generate")
        with patch.object(SFTTrainer, "training_step", return_value="trained") as step:
            self.assertEqual(trainer.training_step(None, {"labels": [1]}, 1), "trained")
            step.assert_called_once_with(trainer, None, {"labels": [1]}, 1)

    def test_real_gkd_forward_kl_mask_shift_and_teacher_gradients(self):
        try:
            import torch
            from trl.experimental.gkd import GKDTrainer
        except ImportError:
            self.skipTest("Torch/TRL training dependencies are not installed")

        class Model(torch.nn.Module):
            def __init__(self, logits):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.tensor(logits, dtype=torch.float32))

            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(logits=self.logits)

        student = Model([[[0., 1., 2.], [2., 0., 1.], [0., 2., 1.], [1., 0., 2.]]])
        teacher = Model([[[2., 0., 1.], [0., 1., 2.], [1., 2., 0.], [0., 1., 2.]]])
        teacher.requires_grad_(False)
        trainer = object.__new__(GKDTrainer)
        trainer.teacher_model = teacher
        trainer.beta = 0.0
        trainer.use_liger_gkd_loss = False
        inputs = {"input_ids": torch.tensor([[0, 1, 2, 0]]),
                  "attention_mask": torch.tensor([[1, 1, 1, 0]]),
                  "labels": torch.tensor([[-100, -100, 2, -100]])}
        loss = trainer.compute_loss(student, inputs, num_items_in_batch=1)
        log_t = teacher.logits[0, 1].log_softmax(-1)
        log_s = student.logits[0, 1].log_softmax(-1)
        expected = (log_t.exp() * (log_t - log_s)).sum()
        torch.testing.assert_close(loss, expected)
        # The token ID selects no hard target: changing it leaves soft KD intact.
        inputs["labels"][0, 2] = 0
        torch.testing.assert_close(trainer.compute_loss(student, inputs, num_items_in_batch=1), expected)
        loss.backward()
        self.assertIsNone(teacher.logits.grad)
        self.assertGreater(student.logits.grad[0, 1].abs().sum().item(), 0)
        self.assertEqual(student.logits.grad[0, [0, 2, 3]].abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
