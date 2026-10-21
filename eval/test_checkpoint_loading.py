"""Check complete checkpoint discovery, batch evaluation, and model routing on CPU."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


spec = importlib.util.spec_from_file_location("run_eval", Path(__file__).with_name("run_eval.py"))
run_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_eval)


class CheckpointLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.base = root / "base"
        self.base.mkdir()
        (self.base / "config.json").write_text("{}", encoding="utf-8")
        self.data = root / "questions.json"
        self.data.write_text("[]", encoding="utf-8")
        self.checkpoint_dir = root / "grpo_checkpoint"
        self.adapters = [root / "outputs" / "qwen", root / "checkpoint" / "qwen" / "checkpoint-50"]
        for adapter in self.adapters:
            adapter.mkdir(parents=True, exist_ok=True)
            (adapter / "adapter_config.json").write_text(
                json.dumps({"r": 16, "rank_pattern": {}}), encoding="utf-8")

    def args(self, adapter=None, *extra):
        argv = ["run_eval.py", "--model", str(self.base), "--data", str(self.data)]
        if adapter is not None:
            argv.extend(["--lora-path", str(adapter)])
        with patch.object(sys, "argv", argv + list(extra)):
            return run_eval.arguments()

    def create_checkpoint(self, step):
        checkpoint = self.checkpoint_dir / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        for name in ("config.json", "trainer_state.json", "tokenizer_config.json"):
            (checkpoint / name).write_text("{}", encoding="utf-8")
        (checkpoint / "model.safetensors").write_bytes(b"")
        return checkpoint.resolve()

    def test_default_scans_checkpoints_in_numeric_order(self):
        expected = [self.create_checkpoint(step) for step in (50, 200, 1000)]
        (self.checkpoint_dir / "final").mkdir()
        incomplete = self.checkpoint_dir / "checkpoint-2000"
        incomplete.mkdir()
        (incomplete / "config.json").write_text("{}", encoding="utf-8")
        with patch.object(run_eval, "CHECKPOINT_ROOT", self.checkpoint_dir), contextlib.redirect_stdout(io.StringIO()):
            args = run_eval.arguments(["--data", str(self.data)])
        self.assertEqual(args.checkpoint_paths, expected)
        self.assertIsNone(args.model)
        self.assertIsNone(args.lora_path)
        self.assertEqual(run_eval.checkpoint_paths(self.checkpoint_dir, "200"), [expected[1]])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_eval.checkpoint_paths(self.checkpoint_dir, "latest"), [expected[-1]])

    def test_checkpoint_selection_and_explicit_model(self):
        checkpoint = self.create_checkpoint(100)
        args = run_eval.arguments(["--checkpoint-dir", str(self.checkpoint_dir),
                                  "--checkpoint", "100", "--data", str(self.data)])
        self.assertEqual(args.checkpoint_paths, [checkpoint])
        direct = run_eval.arguments(["--model", str(checkpoint), "--data", str(self.data)])
        self.assertEqual(direct.model, checkpoint)
        self.assertEqual(direct.tokenizer_path, checkpoint)
        self.assertFalse(direct.checkpoint_paths)
        self.assertIsNone(direct.lora_path)
        with self.assertRaises(ValueError):
            run_eval.checkpoint_paths(self.checkpoint_dir, "999")
        with self.assertRaises(ValueError):
            run_eval.checkpoint_paths(self.checkpoint_dir, "invalid")

    def test_checkpoint_child_command_preserves_options_and_replaces_selection(self):
        checkpoint = self.create_checkpoint(100)
        output = Path(self.temp.name) / "results"
        command = run_eval.checkpoint_command(checkpoint, output, [
            "--checkpoint-dir=" + str(self.checkpoint_dir), "--checkpoint", "latest",
            "--output", "old-output", "--backend", "transformers", "--samples=16",
            "--data", str(self.data), "--limit", "5",
        ])
        self.assertEqual(command[:2], [sys.executable, str(Path(run_eval.__file__).resolve())])
        self.assertNotIn("--checkpoint", command)
        self.assertNotIn("old-output", command)
        child = run_eval.arguments(command[2:])
        self.assertEqual(child.model, checkpoint)
        self.assertEqual(child.output, output.resolve())
        self.assertEqual((child.backend, child.samples, child.limit), ("transformers", 16, 5))
        self.assertEqual(child.template_kwargs, {"enable_thinking": False})
        self.assertFalse(child.checkpoint_paths)
        override = run_eval.checkpoint_command(checkpoint, output, [
            "--data", str(self.data), "--chat-template-kwargs", '{"enable_thinking":true}',
        ])
        self.assertEqual(run_eval.arguments(override[2:]).template_kwargs, {"enable_thinking": True})

    def test_checkpoint_evaluations_are_isolated_and_aggregated(self):
        checkpoints = [self.create_checkpoint(step) for step in (100, 200)]
        args = run_eval.arguments(["--checkpoint-dir", str(self.checkpoint_dir),
                                  "--data", str(self.data), "--output", str(Path(self.temp.name) / "results")])

        def evaluate_child(command, check):
            child = run_eval.arguments(command[2:])
            result_dir = child.output / "test-run"
            result_dir.mkdir(parents=True)
            (result_dir / "summary.json").write_text(json.dumps({
                "datasets": {"aime24.parquet": {"avg@32": 0.5}},
            }), encoding="utf-8")
            return SimpleNamespace(returncode=0)

        with patch.object(run_eval.subprocess, "run", side_effect=evaluate_child) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            run_eval.evaluate_checkpoints(args, ["--checkpoint-dir", str(self.checkpoint_dir),
                                                "--data", str(self.data)])
        self.assertEqual(run.call_count, 2)
        summaries = list(args.output.glob("*/summary.json"))
        self.assertEqual(len(summaries), 1)
        summary = json.loads(summaries[0].read_text(encoding="utf-8"))["checkpoints"]
        self.assertEqual(list(summary), [path.name for path in checkpoints])
        for checkpoint in checkpoints:
            self.assertEqual(summary[checkpoint.name]["model"], str(checkpoint))
            self.assertEqual(summary[checkpoint.name]["metrics"]["datasets"]["aime24.parquet"]["avg@32"], 0.5)

    def test_checkpoint_failures_are_reported_without_skipping_later_checkpoints(self):
        self.create_checkpoint(100)
        self.create_checkpoint(200)
        args = run_eval.arguments(["--checkpoint-dir", str(self.checkpoint_dir),
                                  "--data", str(self.data), "--output", str(Path(self.temp.name) / "failed-results")])
        with patch.object(run_eval.subprocess, "run", return_value=SimpleNamespace(returncode=1)) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "checkpoint-100, checkpoint-200"):
                run_eval.evaluate_checkpoints(args, [])
        self.assertEqual(run.call_count, 2)
        summary = json.loads(next(args.output.glob("*/summary.json")).read_text(encoding="utf-8"))
        self.assertTrue(all(entry["returncode"] == 1 for entry in summary["checkpoints"].values()))

    def test_paths_and_tokenizer_fallback(self):
        base_args = self.args()
        self.assertIsNone(base_args.lora_path)
        self.assertEqual(base_args.tokenizer_path, self.base.resolve())
        self.assertEqual(base_args.max_num_seqs, 32)
        self.assertEqual(self.args(None, "--max-num-seqs", "8").max_num_seqs, 8)
        for adapter in self.adapters:
            with self.subTest(adapter=adapter):
                args = self.args(adapter)
                self.assertEqual(args.lora_path, adapter.resolve())
                self.assertEqual(args.tokenizer_path, self.base.resolve())
                self.assertEqual(args.lora_rank, 16)
                (adapter / "tokenizer_config.json").write_text("{}", encoding="utf-8")
                self.assertEqual(self.args(adapter).tokenizer_path, adapter.resolve())
                # These same fields are persisted by the evaluation entry point.
                metadata = json.loads(json.dumps(vars(self.args(adapter)), default=str))
                self.assertEqual(metadata["lora_path"], str(adapter.resolve()))

    def test_vllm_base_and_both_adapter_paths(self):
        for adapter in [None, *self.adapters]:
            with self.subTest(adapter=adapter):
                llm = MagicMock()
                engine = llm.return_value
                engine.llm_engine.vllm_config.scheduler_config = SimpleNamespace(
                    max_num_seqs=32, max_num_batched_tokens=4096)
                engine.generate.return_value = [SimpleNamespace(outputs=[SimpleNamespace(
                    text=r"\boxed{2}", token_ids=[11, 2], finish_reason="stop")])]
                request = MagicMock(side_effect=lambda name, ident, path: SimpleNamespace(
                    lora_name=name, lora_int_id=ident, lora_path=path))
                modules = {
                    "vllm": SimpleNamespace(LLM=llm, SamplingParams=SimpleNamespace),
                    "vllm.lora.request": SimpleNamespace(LoRARequest=request),
                }
                with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
                    backend = run_eval.VLLMBackend(self.args(adapter), None)
                    for _ in range(2):  # Warmup and evaluation must use the same adapter.
                        result = backend.generate([{"ids": [7, 8]}])
                        self.assertEqual(result[0][0]["text"], r"\boxed{2}")
                self.assertEqual(llm.call_args.kwargs["model"], str(self.base.resolve()))
                if adapter is None:
                    request.assert_not_called()
                    self.assertNotIn("enable_lora", llm.call_args.kwargs)
                    self.assertNotIn("lora_request", engine.generate.call_args.kwargs)
                else:
                    self.assertTrue(llm.call_args.kwargs["enable_lora"])
                    self.assertEqual(llm.call_args.kwargs["max_lora_rank"], 16)
                    for call in engine.generate.call_args_list:
                        self.assertEqual(call.kwargs["lora_request"].lora_path, str(adapter.resolve()))

    def test_vllm_rank_capacity(self):
        adapter = self.adapters[0]
        (adapter / "adapter_config.json").write_text(
            json.dumps({"r": 16, "rank_pattern": {"q_proj": 48}}), encoding="utf-8")
        llm = MagicMock()
        with patch.dict(sys.modules, {
            "vllm": SimpleNamespace(LLM=llm, SamplingParams=SimpleNamespace),
            "vllm.lora.request": SimpleNamespace(LoRARequest=MagicMock()),
        }), contextlib.redirect_stdout(io.StringIO()):
            run_eval.VLLMBackend(self.args(adapter), None)
        self.assertEqual(llm.call_args.kwargs["max_lora_rank"], 64)

    def test_transformers_base_and_both_adapter_paths(self):
        for adapter in [None, *self.adapters]:
            with self.subTest(adapter=adapter):
                base, adapted = MagicMock(), MagicMock()
                base.to.return_value = base
                base.eval.return_value = base
                adapted.eval.return_value = adapted
                for model in (base, adapted):
                    model.generation_config.eos_token_id = 2
                auto = SimpleNamespace(from_pretrained=MagicMock(return_value=base))
                peft = SimpleNamespace(from_pretrained=MagicMock(return_value=adapted))
                modules = {
                    "torch": SimpleNamespace(bfloat16="bf16"),
                    "transformers": SimpleNamespace(AutoModelForCausalLM=auto, GenerationConfig=SimpleNamespace),
                    "peft": SimpleNamespace(PeftModel=peft),
                }
                tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0, bos_token_id=1)
                with patch.dict(sys.modules, modules):
                    backend = run_eval.TransformersBackend(self.args(adapter), tokenizer)
                self.assertEqual(auto.from_pretrained.call_args.args, (str(self.base.resolve()),))
                self.assertTrue(auto.from_pretrained.call_args.kwargs["local_files_only"])
                if adapter is None:
                    peft.from_pretrained.assert_not_called()
                    self.assertIs(backend.model, base)
                else:
                    peft.from_pretrained.assert_called_once_with(
                        base, str(adapter.resolve()), local_files_only=True, is_trainable=False)
                    self.assertIs(backend.model, adapted)
                self.assertEqual(backend.config.eos_token_id, 2)
                self.assertEqual(backend.config.num_return_sequences, 32)


if __name__ == "__main__":
    unittest.main()
