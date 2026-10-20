"""CPU checks: python -B -m unittest train.test_grpo_train -v."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from train import grpo_train


class GrpoLaunchCheckpointTests(unittest.TestCase):
    def test_plain_python_uses_one_gpu_without_distributed_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(grpo_train.configure_launch(), (1, 0))
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3,5"}, clear=True):
            self.assertEqual(grpo_train.configure_launch(), (1, 0))
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3")

    def test_torchrun_supports_arbitrary_gpu_counts_and_keeps_device_mapping(self):
        for world_size in (1, 2, 4):
            for rank in range(world_size):
                visible = ",".join(str(i + 2) for i in range(world_size))
                with self.subTest(world_size=world_size, rank=rank), patch.dict(
                    os.environ, {"WORLD_SIZE": str(world_size), "LOCAL_RANK": str(rank),
                                 "CUDA_VISIBLE_DEVICES": visible}, clear=True
                ):
                    self.assertEqual(grpo_train.configure_launch(), (world_size, rank))
                    self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], visible)
                    cli = grpo_train.parse_args([])
                    batch = cli.per_device_train_batch_size * world_size * cli.gradient_accumulation_steps
                    self.assertEqual(batch % cli.num_generations, 0)

    def test_incomplete_distributed_environment_fails(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "2"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "torchrun"):
                grpo_train.configure_launch()

    def test_full_model_and_sharded_checkpoints_are_resumable(self):
        for name in ("model.safetensors", "model.safetensors.index.json"):
            with self.subTest(weights=name), TemporaryDirectory() as directory:
                checkpoint = Path(directory)
                (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
                (checkpoint / name).write_text("", encoding="utf-8")
                grpo_train.validate_full_checkpoint(checkpoint)

    def test_incomplete_checkpoints_are_rejected(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "full-model"):
                grpo_train.validate_full_checkpoint(checkpoint)


class GrpoDataRewardTests(unittest.TestCase):
    def test_question_and_reference_are_separate(self):
        row = {"prompt": "Compute 2 + 2.", "solution": "DO NOT LEAK",
               "source_prompt": [{"role": "user", "content": "Old Answer: format"}],
               "reward_model": {"ground_truth": "4"}}
        prepared = grpo_train.prepare_example(row)
        self.assertEqual(prepared["answer"], "4")
        self.assertEqual(prepared["prompt"][1], {"role": "user", "content": row["prompt"]})
        self.assertEqual(set(prepared), {"prompt", "answer"})
        self.assertNotIn("DO NOT LEAK", str(prepared["prompt"]))
        self.assertNotIn("Old Answer:", str(prepared["prompt"]))

    def test_invalid_rows_fail_instead_of_training_on_missing_references(self):
        for row in ({"prompt": "Question"}, {"prompt": "", "reward_model": {"ground_truth": "4"}}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                grpo_train.prepare_example(row)

    def test_last_box_and_nested_latex(self):
        text = r"Earlier \boxed{10}; final answer: $\boxed{\frac{8}{2}}$."
        self.assertEqual(grpo_train.extract_final_answer(text), r"\frac{8}{2}")
        self.assertEqual(grpo_train.extract_final_answer(r"\[\boxed{-2015028}\]"), "-2015028")

    def test_thinking_and_malformed_answers_get_no_credit(self):
        invalid = ["No final answer", r"\boxed{4", r"\boxed{}",
                   r"<think>guess \boxed{4}", r"<think>\boxed{4}</think>no answer",
                   r"\boxed{4} or 5", r"\boxed{4}\nActually the answer is 5"]
        for text in invalid:
            with self.subTest(text=text):
                self.assertIsNone(grpo_train.extract_final_answer(text))
        self.assertEqual(grpo_train.extract_final_answer(r"<think>\boxed{4}</think>\boxed{5}"), "5")

    def test_reward_alignment_for_text_and_conversational_completions(self):
        class ExactGrader:
            def __init__(self, gold):
                self.gold = gold

            def grade(self, prediction):
                return {"correct": prediction == self.gold}

        # Stub only the mathematical equivalence engine; check actual extraction,
        # ordering and per-completion reward plumbing without training dependencies.
        with patch.object(grpo_train, "get_answer_grader", side_effect=ExactGrader):
            completions = [r"\boxed{4}", [{"role": "assistant", "content": r"\boxed{5}"}],
                           r"<think>\boxed{4}</think>No final answer"]
            self.assertEqual(grpo_train.accuracy_reward(completions, ["4", "4", "4"]), [1.0, 0.0, 0.0])
            with self.assertRaises(ValueError):
                grpo_train.accuracy_reward(completions, ["4"])

    def test_real_math_equivalence_when_dependencies_are_available(self):
        try:
            import math_verify  # noqa: F401
        except ImportError:
            self.skipTest("math-verify is not installed in this environment")
        self.assertEqual(grpo_train.accuracy_reward(
            [r"\boxed{\frac{8}{2}}", r"\boxed{22099999999999998951424}", r"\boxed{5}"],
            ["4", "22099999999999998951424", "4"]), [1.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
