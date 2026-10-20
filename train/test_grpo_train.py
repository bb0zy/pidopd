"""CPU checks: python -B -m unittest train.test_grpo_train -v."""

import unittest
from unittest.mock import patch

from train import grpo_train


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
