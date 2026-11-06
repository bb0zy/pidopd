"""CPU checks for prompt isolation, checkpoint recovery, and actual Trainer/PEFT updates."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from train.direct_opd import (DirectOPDState, build_token_map, load_frozen_model, remap_tokens,
                              resolve_model_source, resolve_resume, update_alpha)
from train.main import parse_args
from train.utils import myDataCollator, prepare_messages


class DirectOPDControlTests(unittest.TestCase):
    def test_alpha_uses_all_microbatches_and_updates_once_per_optimizer_step(self):
        state = DirectOPDState(1.0, {})
        state.observe(1.0, 2)
        state.observe(-0.5, 10)
        state.finish_step(True, 0.01, 0.5, 2.5)
        self.assertAlmostEqual(state.alpha, 0.99)
        self.assertEqual(state.reward_count, 0)
        self.assertEqual(update_alpha(2.5, 1, 0.01, 0.5, 2.5), 2.5)
        self.assertEqual(update_alpha(0.5, -1, 0.01, 0.5, 2.5), 0.5)

    def test_resume_restores_alpha_counter_and_rejects_wrong_teachers(self):
        with TemporaryDirectory() as directory:
            state = DirectOPDState(0.859, {'before': '/before', 'after': '/after', 'anchor': '/original'})
            state.rollout_calls = 90
            state.save(directory, 45)
            (Path(directory) / 'trainer_state.json').write_text(json.dumps({'global_step': 45}))
            restored = DirectOPDState(1.0, state.signature)
            restored.restore(directory)
            self.assertEqual(restored.alpha, state.alpha)
            self.assertEqual(restored.rollout_calls, 90)
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                DirectOPDState(1.0, {'after': '/wrong'}).restore(directory)
            (Path(directory) / 'trainer_state.json').write_text(json.dumps({'global_step': 44}))
            with self.assertRaisesRegex(ValueError, 'steps differ'):
                restored.restore(directory)

    def test_latest_resume_rejects_incomplete_checkpoint(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint-45'
            path.mkdir()
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                resolve_resume(directory, 'latest')
            with self.assertRaises(FileExistsError):
                resolve_resume(directory, None)
            with self.assertRaises(FileNotFoundError):
                resolve_resume(path, 'latest')

    def test_teacher_token_ids_are_remapped_by_token_string(self):
        student = SimpleNamespace(get_vocab=lambda: {'hello': 0, '<think>': 1, 'world': 2, 'missing': 3})
        teacher = SimpleNamespace(get_vocab=lambda: {'hello': 0, '<think>': 8, 'world': 2})
        mapping = build_token_map(student, teacher)
        ids, candidates = remap_tokens([[0, 1, 2]], [[[1, 0], [2, 1]]], mapping)
        self.assertEqual(ids, [[0, 8, 2]])
        self.assertEqual(candidates, [[[8, 0], [2, 8]]])
        with self.assertRaisesRegex(ValueError, 'token ID 3'):
            remap_tokens([[3]], [[[0]]], mapping)

    def test_prompt_does_not_include_reference_solution_or_answer(self):
        prepared = prepare_messages({'prompt': '2 + 2?', 'solution': 'SECRET SOLUTION', 'Answer': 'SECRET ANSWER'})
        text = prepared['message'][0]['content']
        self.assertIn('2 + 2?', text)
        self.assertIn(r'\boxed{}', text)
        self.assertNotIn('Answer:', text)
        self.assertNotIn('SECRET', text)
        self.assertEqual(list(prepared), ['message'])

    def test_collator_masks_left_padding_and_rejects_truncation(self):
        tokenizer = Mock(pad_token_id=0)
        tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: messages[0]['content']
        tokenizer.side_effect = lambda text, **kwargs: {'input_ids': [1] * len(text)}
        collator = myDataCollator(tokenizer, max_length=3)
        batch = collator([{'message': [{'content': 'ab'}]}, {'message': [{'content': 'abc'}]}])
        self.assertEqual(batch['prompt_attention_mask'].tolist(), [[0, 1, 1], [1, 1, 1]])
        self.assertNotIn('teacher_prompts', batch)
        self.assertFalse(tokenizer.apply_chat_template.call_args.kwargs['enable_thinking'])
        with self.assertRaisesRegex(ValueError, 'overlong'):
            collator([{'message': [{'content': 'abcd'}]}])

    def test_cli_accepts_model_checkpoints_and_resume(self):
        args = parse_args(['--model-before', '/before/checkpoint-100', '--model-after', '/after/checkpoint-200',
                           '--student-model-path', '/student/checkpoint-300', '--resume-from-checkpoint'])
        self.assertEqual(args.resume_from_checkpoint, 'latest')
        self.assertEqual(args.num_generations, 4)
        self.assertEqual(args.model_before, '/before/checkpoint-100')
        with self.assertRaises(SystemExit), patch('sys.stderr'):
            parse_args(['--model-before', '/before', '--model-after', '/after', '--alpha', '0'])


class TrainerResumeTests(unittest.TestCase):
    @staticmethod
    def make_assets(directory):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

        base = Path(directory) / 'base'
        vocab = {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'hello': 3, 'question': 4, 'answer': 5, 'yes': 6, 'no': 7}
        raw = Tokenizer(WordLevel(vocab, unk_token='<unk>'))
        raw.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='<pad>', eos_token='<eos>', unk_token='<unk>')
        tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}hello"
        tokenizer.padding_side = 'left'
        tokenizer.save_pretrained(base)
        torch.manual_seed(7)
        config = Qwen3Config(vocab_size=8, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                             max_position_embeddings=128, eos_token_id=1, pad_token_id=0)
        config._attn_implementation = 'eager'
        Qwen3ForCausalLM(config).save_pretrained(base)
        return base, tokenizer

    def test_full_models_and_lora_checkpoints_load_as_frozen_teachers(self):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        with TemporaryDirectory() as directory:
            base, _ = self.make_assets(directory)
            adapter = Path(directory) / 'checkpoint-45'
            model = get_peft_model(AutoModelForCausalLM.from_pretrained(base), LoraConfig(
                task_type='CAUSAL_LM', r=2, lora_alpha=4, target_modules=['q_proj']))
            for name, value in model.named_parameters():
                if 'lora_' in name:
                    value.data.fill_(0.1)
            model.save_pretrained(adapter)
            self.assertEqual(resolve_model_source(adapter), (str(base.resolve()), str(adapter.resolve())))
            inputs = torch.tensor([[3, 4, 5]])
            model.eval()
            expected = model(inputs).logits.detach()
            loaded = load_frozen_model(adapter)
            torch.testing.assert_close(loaded(inputs).logits, expected, atol=1e-5, rtol=1e-5)
            self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
            self.assertTrue(all(not p.requires_grad for p in load_frozen_model(base).parameters()))

    def test_actual_trainer_checkpoint_restores_optimizer_alpha_and_first_rollout_weights(self):
        from datasets import Dataset
        from peft import LoraConfig, PeftModel
        from transformers import AutoModelForCausalLM
        from train.myconfig import MyConfig
        from train.mytrainer import MyTrainer
        from train.student_generation import pack_completions

        with TemporaryDirectory() as directory:
            base, tokenizer = self.make_assets(directory)
            output = Path(directory) / 'output'
            dataset = Dataset.from_list([{'message': [{'role': 'user', 'content': 'question'}]}] * 4)
            seen_weights = []

            def generation_factory(model, accelerator, tokenizer, args, state=None):
                generator = Mock()
                def generate(inputs):
                    seen_weights.append({k: v.detach().clone() for k, v in model.state_dict().items() if 'lora_' in k})
                    state.rollout_calls += 1
                    return pack_completions(inputs, [[5, 1]] * len(inputs['prompts']), 0)
                generator.generate.side_effect = generate
                return generator

            teacher = Mock()
            teacher.score_before.side_effect = lambda ids, lengths, candidates: torch.full((sum(map(len, candidates)), len(candidates[0][0])), -2.0)
            teacher.score_after.side_effect = lambda ids, lengths, candidates: torch.full((sum(map(len, candidates)), len(candidates[0][0])), -1.0)
            teacher.score_reference.side_effect = lambda ids, lengths, candidates: torch.full((sum(map(len, candidates)), 1), -2.0)

            def make_trainer(steps, adapter=None):
                args = MyConfig(output_dir=str(output), student_model_path=str(base), model_before=str(base), model_after=str(base),
                                use_cpu=True, bf16=False, fp16=False, max_steps=steps, per_device_train_batch_size=1,
                                gradient_accumulation_steps=2, num_generations=1, max_length=32, loss_top_k=3,
                                save_steps=1, logging_steps=1, report_to='none', disable_tqdm=True,
                                learning_rate=0.01, lr_scheduler_type='constant', dataloader_pin_memory=False,
                                gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False})
                model = AutoModelForCausalLM.from_pretrained(base)
                model.config.use_cache = False
                config = LoraConfig(task_type='CAUSAL_LM', r=2, lora_alpha=4, target_modules=['q_proj'])
                if adapter:
                    model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
                    config = None
                return MyTrainer(model=model, args=args, train_dataset=dataset,
                                 processing_class=tokenizer, opd_state=DirectOPDState(1.0, {'anchor': str(base)}),
                                 peft_config=config)

            with patch('train.mytrainer.TeacherClient', return_value=teacher), patch('train.mytrainer.StudentGeneration', side_effect=generation_factory):
                first = make_trainer(1)
                first.train()
                checkpoint = resolve_resume(output, 'latest')
                saved_alpha = first.opd_state.alpha
                saved = {k: v.detach().clone() for k, v in first.model.state_dict().items() if 'lora_' in k}
                first.close()
                self.assertAlmostEqual(saved_alpha, 1.01)
                self.assertEqual(first.opd_state.rollout_calls, 2)
                seen_weights.clear()
                second = make_trainer(2, adapter=checkpoint)
                second.train(resume_from_checkpoint=checkpoint)
                self.assertEqual(second.state.global_step, 2)
                self.assertAlmostEqual(second.opd_state.alpha, 1.0201)
                self.assertEqual(second.opd_state.rollout_calls, 4)
                for key in saved:
                    torch.testing.assert_close(seen_weights[0][key], saved[key])
                self.assertTrue(any(not torch.equal(saved[k], second.model.state_dict()[k]) for k in saved))
                self.assertTrue(all(int(value['step']) == 2 for value in second.optimizer.state.values()))
                self.assertEqual(teacher.score_before.call_args_list[0].args[:2], teacher.score_after.call_args_list[0].args[:2])
                second.close()


if __name__ == '__main__':
    unittest.main()
