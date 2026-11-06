"""CPU tests for dynamic filtering, cached rollouts and rank synchronization."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr
from io import StringIO
import random
from threading import Barrier
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from train import dapo_train


def group(question, size=2):
    return [{"prompt": question, "answer": str(question)} for _ in range(size)]


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 98

    def batch_decode(self, ids, **kwargs):
        return [str(row[0]) for row in ids]


class FakeTrainerBase:
    """Only stand in for TRL/vLLM; run the real dynamic-sampling implementation."""

    def __init__(self, rounds, rank=0, world_size=1, collective=None):
        self.rounds = rounds
        self.calls = []
        self.collective = collective
        self.accelerator = SimpleNamespace(process_index=rank, num_processes=world_size)
        self.args = SimpleNamespace(seed=42)
        self.state = SimpleNamespace(global_step=100)
        self.model = SimpleNamespace(training=True)
        self._tokenizer = FakeTokenizer()
        self._metrics = {"train": defaultdict(list)}
        self.num_generations = 2
        self.rollout_func = None
        self.train_dataset = [{"prompt": i, "answer": str(i)} for i in range(20, 40)]
        self.fail_scoring = False

    def _tokenize_prompts(self, prompts):
        return [[prompt] for prompt in prompts], None, {}

    def _generate_single_turn(self, prompt_ids, images, fields):
        scores = self.rounds[len(self.calls)]
        if len(scores) != len(prompt_ids):
            raise AssertionError("Wrong refill batch size")
        self.calls.append(prompt_ids)
        ids = [[score, 99] for score in scores]
        logprobs = [[-float(prompt[0]) - 0.25, -0.75] for prompt in prompt_ids]
        return ids, logprobs

    def _generate_and_score_completions(self, inputs):
        if self.fail_scoring:
            raise RuntimeError("Scoring failed")
        output = self.rollout_func([row["prompt"] for row in inputs], self)
        return inputs, output


class FakeTrainer(dapo_train.DynamicSamplingMixin, FakeTrainerBase):
    def _sampling_totals(self, values):
        if self.collective is None:
            return values
        return self.collective(self.accelerator.process_index, values)


class DapoTests(unittest.TestCase):
    def test_defaults_and_resume_options(self):
        cli = dapo_train.parse_args(["--resume-from-checkpoint", "latest"])
        self.assertFalse(cli.enable_thinking)
        self.assertEqual(cli.max_completion_length, 10240)
        self.assertEqual(cli.output_dir, dapo_train.PROJECT_ROOT / "train/dapo_checkpoint")
        self.assertEqual(cli.resume_from_checkpoint, "latest")
        self.assertEqual(cli.per_device_train_batch_size * cli.gradient_accumulation_steps, 64)
        self.assertEqual((cli.beta, cli.epsilon, cli.epsilon_high), (0.0, 0.2, 0.28))
        cli = dapo_train.parse_args(["--max-new-tokens", "12000", "--enable-thinking"])
        self.assertEqual(cli.max_completion_length, 12000)
        self.assertTrue(cli.enable_thinking)

    def test_invalid_group_and_length_settings(self):
        cases = [
            ["--gradient-accumulation-steps", "3"],
            ["--num-generations", "1"],
            ["--max-sampling-rounds", "0"],
            ["--max-new-tokens", "1024"],
            ["--overlong-penalty", "1"],
            ["--epsilon-high", "0.1"],
        ]
        for args in cases:
            with self.subTest(args=args), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                dapo_train.parse_args(args)

    def test_penalty_boundaries_and_disabled_penalty(self):
        reward = dapo_train.make_overlong_reward(10240, 2048, 0.5)
        lengths = [10, 8192, 9216, 10240, 12000]
        ids = [[0] * length for length in lengths]
        self.assertEqual(reward(ids), [0.0, 0.0, -0.25, -0.5, -0.5])
        self.assertEqual(dapo_train.make_overlong_reward(10240, 2048, 0)(ids), [0.0] * 5)

    def test_filter_keeps_complete_mixed_groups_and_rejects_all_truncated(self):
        rows = group(1) + group(2) + group(3) + group(4)
        scores = [1, 1, 0, 0, 1, 0, 1, 0]
        ids = [[0, 99]] * 6 + [[0, 1]] * 2
        self.assertEqual(dapo_train.mixed_accuracy_group_starts(rows, scores, ids, 2, {99}), [4])
        with self.assertRaisesRegex(ValueError, "different questions"):
            dapo_train.mixed_accuracy_group_starts(group(1, 1) + group(2, 1), [1, 0], ids[:2], 2, {99})
        with self.assertRaisesRegex(ValueError, "complete local"):
            dapo_train.mixed_accuracy_group_starts(rows[:-1], scores, ids, 2, {99})

    @patch.object(dapo_train, "accuracy_reward", side_effect=lambda texts, answers: list(map(float, texts)))
    def test_refill_preserves_questions_token_ids_and_sampling_logprobs(self, reward):
        trainer = FakeTrainer([[1, 1, 1, 0], [0, 1]])
        inputs, replay = trainer._generate_and_score_completions(group(1) + group(2))
        self.assertEqual(len(trainer.calls), 2)
        self.assertEqual(len(inputs), 4)
        self.assertEqual(inputs[:2], group(2))
        self.assertGreaterEqual(inputs[2]["prompt"], 20)  # New prompt, not a retry of question 1.
        self.assertEqual(inputs[2], inputs[3])
        self.assertEqual(replay["completion_ids"], [[1, 99], [0, 99], [0, 99], [1, 99]])
        self.assertEqual(replay["prompt_ids"], [[row["prompt"]] for row in inputs])
        self.assertEqual(replay["logprobs"], [[-row["prompt"] - 0.25, -0.75] for row in inputs])
        self.assertEqual(trainer._metrics["train"]["dapo/raw_accuracy"], [4 / 6])
        self.assertIsNone(trainer.rollout_func)

    @patch.object(dapo_train, "accuracy_reward", side_effect=lambda texts, answers: list(map(float, texts)))
    def test_round_limit_never_trains_a_partial_batch(self, reward):
        trainer = FakeTrainer([[0, 0], [1, 1]], max_sampling_rounds=2)
        with self.assertRaisesRegex(RuntimeError, "no partial optimizer step"):
            trainer._generate_and_score_completions(group(1))
        self.assertEqual(len(trainer.calls), 2)
        self.assertIsNone(trainer.rollout_func)

    @patch.object(dapo_train, "accuracy_reward", side_effect=lambda texts, answers: list(map(float, texts)))
    def test_replay_is_cleared_if_parent_scoring_fails(self, reward):
        trainer = FakeTrainer([[1, 0]])
        trainer.fail_scoring = True
        with self.assertRaisesRegex(RuntimeError, "Scoring failed"):
            trainer._generate_and_score_completions(group(1))
        self.assertIsNone(trainer.rollout_func)

    def test_refill_choices_repeat_after_restoring_step_and_rank(self):
        trainer = FakeTrainer([])
        seed = f"{trainer.args.seed}:{trainer.state.global_step}:{trainer.accelerator.process_index}"
        first = trainer._refill_inputs(8, random.Random(seed))
        restored = trainer._refill_inputs(8, random.Random(seed))
        self.assertEqual(first, restored)
        for index in range(0, 8, 2):
            self.assertEqual(first[index], first[index + 1])

    @patch.object(dapo_train, "accuracy_reward", side_effect=lambda texts, answers: list(map(float, texts)))
    def test_two_ranks_finish_together_even_when_one_fills_early(self, reward):
        barrier = Barrier(2, timeout=5)
        slots = [None, None]

        def collective(rank, values):
            slots[rank] = values
            barrier.wait()
            totals = [sum(column) for column in zip(*slots)]
            barrier.wait()
            return totals

        # Rank 0 fills immediately; rank 1 needs another round. Rank 0 must keep
        # participating without replacing its already-accepted completions.
        trainers = [FakeTrainer([[1, 0], [0, 0]], rank=0, world_size=2, collective=collective),
                    FakeTrainer([[0, 0], [1, 0]], rank=1, world_size=2, collective=collective)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(trainer._generate_and_score_completions, group(rank + 1))
                       for rank, trainer in enumerate(trainers)]
            results = [future.result(timeout=10) for future in futures]
        self.assertEqual(results[0][0], group(1))
        self.assertGreaterEqual(results[1][0][0]["prompt"], 20)
        for trainer in trainers:
            self.assertEqual(len(trainer.calls), 2)
            self.assertEqual(trainer._metrics["train"]["dapo/sampling_rounds"], [2])
            self.assertEqual(trainer._metrics["train"]["dapo/raw_accuracy"], [2 / 8])
            self.assertIsNone(trainer.rollout_func)


if __name__ == "__main__":
    unittest.main()
