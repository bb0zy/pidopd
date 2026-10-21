"""CPU data pipeline checks; vLLM itself requires the GPU server."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from train import kd_data, teacher_rollout
from train.test_align_train import FakeTokenizer


class TeacherRolloutTests(unittest.TestCase):
    def test_generation_budget_and_sample_defaults(self):
        cli = teacher_rollout.parse_args([])
        self.assertEqual(cli.num_rollouts, 1)
        self.assertEqual(cli.max_new_tokens, 16384)
        self.assertGreaterEqual(cli.max_model_len, cli.max_prompt_length + cli.max_new_tokens)
        self.assertEqual(teacher_rollout.parse_args(["--num-rollouts", "4"]).num_rollouts, 4)

    def test_invalid_budget_and_rollout_count_fail(self):
        for args in (["--num-rollouts", "0"], ["--max-new-tokens", "32768"], ["--temperature", "nan"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                teacher_rollout.parse_args(args)

    def test_output_keeps_exact_ids_all_samples_and_finish_reasons(self):
        question = {"question_id": 4, "prompt": "Question", "prompt_token_ids": [10, 11, 12]}
        output = SimpleNamespace(prompt_token_ids=[10, 11, 12], outputs=[
            SimpleNamespace(index=1, text="unfinished", token_ids=[30, 31], finish_reason="length", stop_reason=None),
            SimpleNamespace(index=0, text="reasoning and answer", token_ids=[20, 21, 99], finish_reason="stop", stop_reason=99)])
        rows = teacher_rollout.output_records(question, output, [0, 1])
        self.assertEqual([row["sample_id"] for row in rows], [0, 1])
        self.assertEqual(rows[0]["completion_token_ids"], [20, 21, 99])
        self.assertEqual(rows[1]["completion_token_ids"], [30, 31])
        self.assertEqual([row["finish_reason"] for row in rows], ["stop", "length"])
        self.assertEqual(kd_data.alignment_example(rows[0])["labels"], [-100, -100, -100, 20, 21, 99])

    def test_resume_recovers_only_torn_tail_and_rejects_duplicates(self):
        row = {"question_id": 0, "sample_id": 0, "finish_reason": "stop"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            path.write_bytes((json.dumps(row) + '\n{"question_id":').encode())
            self.assertEqual(teacher_rollout.scan_existing(path, 2, repair_tail=True), ({(0, 0)}, 1, 0))
            self.assertEqual(len(path.read_text().splitlines()), 1)
            with path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                teacher_rollout.scan_existing(path, 2, repair_tail=True)

    def test_stage_one_to_offline_data_and_idempotent_resume(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("PyArrow is not installed")
        calls = []

        class Engine:
            def __init__(self, **kwargs):
                calls.append(kwargs)

            def generate(self, requests, params, use_tqdm):
                return [SimpleNamespace(prompt_token_ids=request["prompt_token_ids"], outputs=[
                    SimpleNamespace(index=i, text="reasoning and answer", token_ids=[20, 21, 99],
                                    finish_reason="stop", stop_reason=99) for i in range(param.n)])
                        for request, param in zip(requests, params, strict=True)]

        transformers = ModuleType("transformers")
        transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeTokenizer())
        transformers.AutoConfig = SimpleNamespace(from_pretrained=lambda *args, **kwargs:
            SimpleNamespace(get_text_config=lambda: SimpleNamespace(max_position_embeddings=40960)))
        transformers.GenerationConfig = SimpleNamespace()
        vllm = ModuleType("vllm")
        vllm.LLM = Engine
        vllm.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher = root / "teacher"
            teacher.mkdir()
            (teacher / "config.json").write_text("{}", encoding="utf-8")
            source, output = root / "source.parquet", root / "responses.jsonl"
            pq.write_table(pa.Table.from_pylist([{"prompt": "Question"}, {"prompt": "Question"}]), source)
            argv = ["teacher_rollout", "--teacher-model-path", str(teacher), "--data-path", str(source),
                    "--output-path", str(output), "--num-rollouts", "2", "--batch-size", "1"]
            with patch.dict(sys.modules, {"transformers": transformers, "vllm": vllm}), \
                    patch.dict("os.environ", {"WORLD_SIZE": "1"}), contextlib.redirect_stdout(io.StringIO()):
                with patch.object(sys, "argv", argv):
                    teacher_rollout.main()
                rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
                self.assertEqual({(row["question_id"], row["sample_id"]) for row in rows}, {(0, 0), (0, 1), (1, 0), (1, 1)})
                self.assertTrue(all(kd_data.usable_rollout(row, 32768) for row in rows))
                self.assertTrue(all(kd_data.alignment_example(row)["labels"][-3:] == [20, 21, 99] for row in rows))
                before = output.read_bytes()
                with patch.object(sys, "argv", argv + ["--resume"]):
                    teacher_rollout.main()
                self.assertEqual(output.read_bytes(), before)
                self.assertEqual(len(calls), 1, "Completed caches must not load the vLLM engine again")
                # Simulate a crash after three of four responses were durably written.
                output.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]), encoding="utf-8")
                with patch.object(sys, "argv", argv + ["--resume"]):
                    teacher_rollout.main()
                resumed = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(resumed), 4)
                self.assertEqual({(row["question_id"], row["sample_id"]) for row in resumed},
                                 {(0, 0), (0, 1), (1, 0), (1, 1)})
                self.assertEqual(len(calls), 2)
                with patch.object(sys, "argv", argv + ["--resume", "--num-rollouts", "3"]):
                    with self.assertRaisesRegex(ValueError, "same teacher"):
                        teacher_rollout.main()
            metadata = json.loads(kd_data.metadata_path(output).read_text(encoding="utf-8"))
            self.assertTrue(metadata["complete"])
            self.assertEqual(metadata["expected_responses"], 4)
            self.assertEqual(metadata["saved_responses"], 4)


if __name__ == "__main__":
    unittest.main()
