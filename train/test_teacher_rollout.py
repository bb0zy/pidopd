"""CPU data pipeline checks; vLLM itself requires the GPU server."""

import contextlib
from collections import UserDict
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from train import kd_data, teacher_rollout
from train.test_align_train import FakeTokenizer


def write_cpu_rollouts(cli, teacher_path, pending, output_path, visible_devices):
    """Spawn-safe stand-in for GPU work; exercises real process/file isolation."""
    if os.name == "posix":
        os.setsid()
    with output_path.open("a", encoding="utf-8") as handle:
        for question, sample_ids in pending:
            for sample_id in sample_ids:
                handle.write(json.dumps({"question_id": question["question_id"], "sample_id": sample_id,
                    "prompt": question["prompt"], "prompt_token_ids": question["prompt_token_ids"],
                    "completion_token_ids": [20, 99], "completion": visible_devices,
                    "finish_reason": "stop", "stop_reason": 99}) + "\n")


class TeacherRolloutTests(unittest.TestCase):
    def test_visible_gpus_are_partitioned_between_replicas(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,5,7,9"}):
            self.assertEqual(teacher_rollout.visible_device_groups(4, 1), ["2", "5", "7", "9"])
            self.assertEqual(teacher_rollout.visible_device_groups(2, 2), ["2,5", "7,9"])
            with self.assertRaisesRegex(ValueError, "needs 5"):
                teacher_rollout.visible_device_groups(5, 1)

    def test_real_spawned_workers_cover_all_questions_once_and_merge(self):
        cli = teacher_rollout.parse_args(["--data-parallel-size", "2", "--num-rollouts", "2"])
        pending = [({"question_id": qid, "prompt": "Question", "prompt_token_ids": [10, 11]}, [0, 1])
                   for qid in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards, output = root / "responses.parts", root / "responses.jsonl"
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,5"}), \
                    patch.object(teacher_rollout, "rollout_worker", write_cpu_rollouts):
                teacher_rollout.run_parallel(cli, root, pending, shards)
            paths = sorted(shards.glob("rank-*.jsonl"))
            self.assertEqual(len(paths), 2)
            seen, stopped, truncated = teacher_rollout.scan_cache_files(paths, 2)
            self.assertEqual(seen, {(qid, sid) for qid in range(5) for sid in (0, 1)})
            self.assertEqual((stopped, truncated), (10, 0))
            teacher_rollout.merge_cache_files(output, paths, 10)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 10)
            for row in rows:
                self.assertEqual(row["completion"], "2" if row["question_id"] % 2 == 0 else "5")
                self.assertEqual(kd_data.alignment_example(row)["labels"], [-100, -100, 20, 99])

    def test_merge_deduplicates_recovery_copies_and_preserves_output_on_failure(self):
        row = {"question_id": 0, "sample_id": 0, "finish_reason": "stop"}
        other = {"question_id": 1, "sample_id": 0, "finish_reason": "length"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, shard = root / "responses.jsonl", root / "rank-0.jsonl"
            output.write_text(json.dumps(row) + "\n", encoding="utf-8")
            shard.write_text(json.dumps(row) + "\n" + json.dumps(other) + "\n", encoding="utf-8")
            self.assertEqual(teacher_rollout.scan_cache_files([output, shard], 1), ({(0, 0), (1, 0)}, 1, 1))
            original = output.read_bytes()
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                teacher_rollout.merge_cache_files(output, [output, shard], 3)
            self.assertEqual(output.read_bytes(), original)
            teacher_rollout.merge_cache_files(output, [output, shard], 2)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)

    def test_generation_budget_and_sample_defaults(self):
        cli = teacher_rollout.parse_args([])
        self.assertEqual(cli.num_rollouts, 1)
        self.assertEqual(cli.data_parallel_size, 1)
        self.assertEqual(cli.max_new_tokens, 16384)
        self.assertGreaterEqual(cli.max_model_len, cli.max_prompt_length + cli.max_new_tokens)
        self.assertEqual(teacher_rollout.parse_args(["--num-rollouts", "4"]).num_rollouts, 4)

    def test_invalid_budget_and_rollout_count_fail(self):
        for args in (["--num-rollouts", "0"], ["--max-new-tokens", "32768"], ["--temperature", "nan"],
                     ["--data-parallel-size", "0"]):
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
        class MappingTokenizer(FakeTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                ids = super().apply_chat_template(messages, **kwargs)
                # Same mapping behavior as the BatchEncoding in the reported traceback.
                return UserDict(input_ids=ids, attention_mask=[1] * len(ids))

        transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs: MappingTokenizer())
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

                # Resume an interrupted DP run, then finish with a different replica count.
                output.unlink()
                shards = output.with_suffix(".parts")
                shards.mkdir()
                (shards / "rank-00000.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
                fail_after_first_worker = True

                def fake_parallel(cli, teacher_path, pending, shard_dir):
                    groups = teacher_rollout.visible_device_groups(cli.data_parallel_size, cli.tensor_parallel_size)
                    for rank in range(len(groups)):
                        assigned = pending[rank::len(groups)]
                        if assigned:
                            teacher_rollout.generate_responses(cli, teacher_path, assigned, shard_dir / f"rank-{rank:05d}.jsonl")
                            if fail_after_first_worker:
                                raise RuntimeError("simulated worker failure")

                with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,5,7"}), \
                        patch.object(teacher_rollout, "run_parallel", side_effect=fake_parallel):
                    with patch.object(sys, "argv", argv + ["--resume", "--data-parallel-size", "2"]):
                        with self.assertRaisesRegex(RuntimeError, "simulated"):
                            teacher_rollout.main()
                    self.assertFalse(json.loads(kd_data.metadata_path(output).read_text())["complete"])
                    self.assertFalse(output.exists())
                    fail_after_first_worker = False
                    with patch.object(sys, "argv", argv + ["--resume", "--data-parallel-size", "3"]):
                        teacher_rollout.main()
                final_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
                self.assertEqual({(row["question_id"], row["sample_id"]) for row in final_rows},
                                 {(0, 0), (0, 1), (1, 0), (1, 1)})
                self.assertEqual(len(final_rows), 4)
                self.assertFalse(list(shards.glob("rank-*.jsonl")))
            metadata = json.loads(kd_data.metadata_path(output).read_text(encoding="utf-8"))
            self.assertTrue(metadata["complete"])
            self.assertEqual(metadata["expected_responses"], 4)
            self.assertEqual(metadata["saved_responses"], 4)


if __name__ == "__main__":
    unittest.main()
