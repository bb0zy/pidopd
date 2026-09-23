"""CPU-only grading regression checks; intentionally not run on the dev machine.

Server command: python -m unittest discover -s eval -p test_answer_grader.py
"""
import unittest
from pathlib import Path

from answer_grader import AnswerGrader


class GradingTests(unittest.TestCase):
    def test_equivalent_answers(self):
        pairs = [
            ("023", "23"),
            (r"336^\circ", "336"),
            (r"336^\circ", "336°"),
            (r"\frac{1}{576}", "1/576"),
            (r"8\sqrt{10}", r"\sqrt{640}"),
            (r"1-\frac{2}{\pi}", r"\frac{\pi-2}{\pi}"),
            (r"2^{25}\cdot26!", r"26!\times33554432"),
            (r"\frac{4}{9}", r"\frac{8}{18}"),
            (r"\frac{-1+\sqrt{17}}{2}, \frac{-1-\sqrt{17}}{2}",
             r"\frac{-1-\sqrt{17}}{2}, \frac{\sqrt{17}-1}{2}"),
        ]
        for gold, pred in pairs:
            with self.subTest(gold=gold, prediction=pred):
                result = AnswerGrader(gold).grade(pred)
                self.assertTrue(result["correct"], result)

    def test_wrong_or_incomplete(self):
        for gold, pred in [("204", "205"), ("204", None), ("204", "204 or 205"),
                           (r"8\sqrt{10}", r"8\sqrt{11}"),
                           ("1, 2", "1"), ("1, 2", "1, 1"),
                           ("1, 2", "1, 2, 3"), (r"\frac{4}{9}", "0.44")]:
            with self.subTest(gold=gold, prediction=pred):
                self.assertFalse(AnswerGrader(gold).grade(pred)["correct"])

    def test_every_local_reference_is_parseable(self):
        import pyarrow.parquet as pq
        directory = Path(__file__).resolve().parents[1] / "dataset/eval"
        for name in ("aime24.parquet", "aime25.parquet", "hmmt_feb.parquet"):
            for index, row in enumerate(pq.read_table(directory / name).to_pylist()):
                with self.subTest(file=name, row=index):
                    gold = row["reward_model"]["ground_truth"]
                    self.assertTrue(AnswerGrader(gold).grade(gold)["correct"])


if __name__ == "__main__":
    unittest.main()
