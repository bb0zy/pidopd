"""CPU checks: python -B -m unittest tests.train.test_align_train -v."""

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
        self.assertEqual(cli.loss_top_k, 16)
        self.assertEqual(cli.logit_chunk_size, 128)
        self.assertFalse(hasattr(cli, "max_new_tokens"))

    def test_all_positions_chunk_size_and_invalid_values(self):
        from unittest.mock import patch
        from train.offline_gkd import ChunkedTopKLoss

        self.assertEqual(align_train.parse_args(['--logit-chunk-size', '-1']).logit_chunk_size, -1)
        for size in (0, -2):
            with self.subTest(size=size), self.assertRaises(SystemExit), patch('sys.stderr'):
                align_train.parse_args(['--logit-chunk-size', str(size)])
            with self.assertRaisesRegex(ValueError, '-1 or positive'):
                ChunkedTopKLoss(chunk_size=size)

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


class TopKObjectiveTests(unittest.TestCase):
    def test_student_topk_surrogate_and_detached_teacher(self):
        import torch
        from train.offline_gkd import topk_distillation_loss

        student = torch.tensor([[2., 0., 1., -1.], [0., 1., -1., 2.]], requires_grad=True)
        teacher = torch.tensor([[-1., 3., 0., 1.], [3., -1., 2., 0.]], requires_grad=True)
        # Student and teacher prefer different tokens; support must be student-selected.
        ids = torch.tensor([[0, 2], [3, 1]])
        log_s = student.log_softmax(-1).gather(-1, ids)
        log_t = teacher.log_softmax(-1).gather(-1, ids)
        weights = student.gather(-1, ids).softmax(-1)
        expected = -(weights * (log_t - log_s)).detach().mul(log_s).sum(-1).mean()
        actual = topk_distillation_loss(student, teacher, 2)
        torch.testing.assert_close(actual, expected)
        actual_grad = torch.autograd.grad(actual, student, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, student)[0]
        torch.testing.assert_close(actual_grad, expected_grad)
        actual.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(student.grad).all())
        # Normalization remains full-vocabulary: tokens outside top-k can receive gradients.
        self.assertGreater(student.grad[0, 1].abs().item(), 0)

    def test_full_support_gradient_matches_reverse_kl(self):
        import torch
        from train.offline_gkd import topk_distillation_loss

        student = torch.tensor([[1., 0., -1.]], requires_grad=True)
        teacher = torch.tensor([[0., 1., -0.5]])
        log_s, log_t = student.log_softmax(-1), teacher.log_softmax(-1)
        full_kl = (log_s.exp() * (log_s - log_t)).sum(-1).mean()
        for k in (3, 16):
            loss = topk_distillation_loss(student, teacher, k)
            gradient = torch.autograd.grad(loss, student, retain_graph=True)[0]
            expected = torch.autograd.grad(full_kl, student, retain_graph=True)[0]
            torch.testing.assert_close(gradient, expected)
        identical = topk_distillation_loss(student, student.detach(), 1)
        torch.testing.assert_close(torch.autograd.grad(identical, student)[0], torch.zeros_like(student))

    def test_chunked_head_matches_dense_loss_and_gradients_with_mask_and_bias(self):
        import torch
        from train.offline_gkd import ChunkedTopKLoss, topk_distillation_loss

        torch.manual_seed(42)
        labels = torch.tensor([[-100, 1, 3, -100], [-100, -100, 2, 1]])
        mask = labels != -100
        teacher = torch.randn(2, 4, 6, requires_grad=True)
        teacher_weight = torch.randn(9, 6, requires_grad=True)
        teacher_bias = torch.randn(9, requires_grad=True)
        initial = [torch.randn(2, 4, 5), torch.randn(9, 5), torch.randn(9)]
        for size in (-1, 1, 3, 128):
            student, weight, bias = [x.clone().requires_grad_() for x in initial]
            expected = topk_distillation_loss(
                torch.nn.functional.linear(student, weight, bias)[mask],
                torch.nn.functional.linear(teacher, teacher_weight, teacher_bias)[mask], 3)
            expected_grads = torch.autograd.grad(expected, (student, weight, bias))
            actual = ChunkedTopKLoss(3, size)(student, weight, teacher, teacher_weight, labels, bias, teacher_bias)
            torch.testing.assert_close(actual, expected)
            actual.backward()
            for value, gradient in zip((student, weight, bias), expected_grads):
                torch.testing.assert_close(value.grad, gradient, atol=1e-6, rtol=1e-5)
            self.assertTrue(torch.equal(student.grad[~mask], torch.zeros_like(student.grad[~mask])))
            self.assertIsNone(teacher.grad)
            self.assertIsNone(teacher_weight.grad)
            self.assertIsNone(teacher_bias.grad)
        student, weight, bias = [x.clone().requires_grad_() for x in initial]
        zero = ChunkedTopKLoss()(student, weight, teacher, teacher_weight, torch.full_like(labels, -100), bias)
        zero.backward()
        self.assertEqual(zero.item(), 0)
        self.assertTrue(torch.equal(weight.grad, torch.zeros_like(weight)))

    def test_actual_decoder_forward_causal_mask_and_global_token_normalization(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from trl.models.utils import _ForwardRedirection
        from train.offline_gkd import ChunkedTopKLoss, OfflineGKDTrainer, topk_distillation_loss

        torch.manual_seed(7)
        def model(hidden):
            config = Qwen3Config(vocab_size=11, hidden_size=hidden, intermediate_size=32,
                                 num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                 head_dim=8, max_position_embeddings=64)
            config._attn_implementation = 'eager'
            return Qwen3ForCausalLM(config)
        student, teacher = model(16), model(24).eval().requires_grad_(False)
        trainer = object.__new__(OfflineGKDTrainer)
        trainer.accelerator = SimpleNamespace(unwrap_model=lambda m: m)
        trainer.teacher_model = teacher
        trainer.use_liger_gkd_loss = True
        trainer._forward_redirection = _ForwardRedirection()
        trainer.liger_loss = ChunkedTopKLoss(3, 1)
        inputs = {'input_ids': torch.tensor([[1, 2, 3, 4, 0]]),
                  'attention_mask': torch.tensor([[1, 1, 1, 1, 0]]),
                  'labels': torch.tensor([[-100, -100, 3, 4, -100]])}
        valid = inputs['labels'][:, 1:] != -100
        s_logits = student(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).logits[:, :-1][valid]
        with torch.no_grad():
            t_logits = teacher(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).logits[:, :-1][valid]
        expected = topk_distillation_loss(s_logits, t_logits, 3)
        # Normal model lm_head.forward must not materialize full-sequence logits.
        from unittest.mock import patch
        with patch.object(student.lm_head, 'forward', side_effect=AssertionError('Full-sequence logits are forbidden')):
            loss, outputs = trainer.compute_loss(student, inputs, return_outputs=True, num_items_in_batch=4)
        torch.testing.assert_close(loss, expected * 2 / 4)
        self.assertIsNone(outputs.logits)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertGreater(student.lm_head.weight.grad.abs().sum().item(), 0)
        # Labels define positions only; there is no hard-label CE component.
        inputs['labels'][0, 2] = 10
        torch.testing.assert_close(trainer.compute_loss(student, inputs, num_items_in_batch=4), expected * 2 / 4)

    def test_real_training_and_full_checkpoint_resume_without_rollouts(self):
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM
        from trl.experimental.gkd import GKDConfig
        from train.offline_gkd import OfflineGKDTrainer
        from tests.train.test_direct_opd import TrainerResumeTests
        from train.rl_train import resolve_full_resume

        with tempfile.TemporaryDirectory() as directory:
            base, tokenizer = TrainerResumeTests.make_assets(directory)
            output = Path(directory) / 'checkpoints'
            features = [kd_data.alignment_example({'prompt_token_ids': [3, 4], 'completion_token_ids': [5, 1]})] * 4
            dataset = Dataset.from_list(features)
            def make_trainer(steps):
                args = GKDConfig(output_dir=str(output), use_cpu=True, bf16=False, fp16=False,
                                 use_liger_kernel=False, lmbda=0, seq_kd=False, beta=1,
                                 max_steps=steps, per_device_train_batch_size=1, gradient_accumulation_steps=2,
                                 max_length=32, report_to='none', disable_tqdm=True, dataloader_pin_memory=False,
                                 save_steps=1, learning_rate=0.01, lr_scheduler_type='constant',
                                 gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False})
                student = AutoModelForCausalLM.from_pretrained(base)
                teacher = AutoModelForCausalLM.from_pretrained(base).eval().requires_grad_(False)
                with torch.no_grad():
                    teacher.lm_head.weight[5] += 0.2
                trainer = OfflineGKDTrainer(model=student, teacher_model=teacher, args=args, processing_class=tokenizer,
                                            train_dataset=dataset, data_collator=align_train.OfflineKDCollator(0),
                                            loss_top_k=3, logit_chunk_size=1)
                trainer.model_accepts_loss_kwargs = True
                trainer.generate_on_policy_outputs = lambda *a, **kw: self.fail('Offline alignment must never generate')
                return trainer
            first = make_trainer(1)
            first.train()
            checkpoint = resolve_full_resume(output, 'latest')
            self.assertTrue((Path(checkpoint) / 'model.safetensors').is_file())
            saved = {k: v.detach().clone() for k, v in first.model.state_dict().items()}
            second = make_trainer(2)
            second.train(resume_from_checkpoint=checkpoint)
            self.assertEqual(second.state.global_step, 2)
            self.assertTrue(any(not torch.equal(saved[k], v) for k, v in second.model.state_dict().items()))
            self.assertTrue(all(int(value['step']) == 2 for value in second.optimizer.state.values()))
            self.assertTrue(all(p.grad is None for p in second.teacher_model.parameters()))


if __name__ == "__main__":
    unittest.main()
