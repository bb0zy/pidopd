"""Export a local LoRA checkpoint as a standalone model for the existing evaluator."""
import argparse
import json
import os
from pathlib import Path

for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
    os.environ[key] = "1"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--base-model", type=Path, help="Override original local base path after moving servers")
    p.add_argument("--device", default="cpu", help="CPU by default to avoid occupying GPU memory")
    p.add_argument("--trust-remote-code", action="store_true")
    a = p.parse_args()
    checkpoint = a.checkpoint.expanduser().resolve()
    marker = json.loads((checkpoint / "complete.json").read_text(encoding="utf-8"))
    if marker.get("format") != "direct-opd-lora-v1":
        p.error("Expected a complete Direct-OPD LoRA checkpoint")
    arguments = json.loads((checkpoint / "run_arguments.json").read_text(encoding="utf-8"))
    base = (a.base_model or Path(arguments["student"])).expanduser().resolve()
    if not (base / "config.json").is_file():
        p.error(f"Missing original local base model: {base}")
    output = a.output.expanduser().resolve()
    if output.exists():
        p.error("Output must be a new directory; existing files will not be overwritten")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        str(base), local_files_only=True, trust_remote_code=a.trust_remote_code,
        dtype=torch.bfloat16).to(a.device)
    model = PeftModel.from_pretrained(model, str(checkpoint), local_files_only=True)
    merged = model.merge_and_unload(safe_merge=True)
    output.mkdir(parents=True, exist_ok=False)
    merged.save_pretrained(output, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True,
                                        trust_remote_code=a.trust_remote_code)
    tok.save_pretrained(output)
    print(f"Merged model: {output}")


if __name__ == "__main__":
    main()
