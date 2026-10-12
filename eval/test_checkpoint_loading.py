"""Check base-model and adapter routing without loading GPU dependencies."""
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
                    text="Answer: 2", token_ids=[11, 2], finish_reason="stop")])]
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
                        self.assertEqual(result[0][0]["text"], "Answer: 2")
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
