"""Read installed distribution metadata only. No model/GPU import or downloads."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["core", "train", "vllm"], default="core")
    args = parser.parse_args()
    from packaging.requirements import Requirement
    from packaging.version import Version
    targets = {}
    for line in (Path(__file__).parent / "constraints.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, version = line.split("==", 1)
            targets[name] = version
    required = ["torch", "accelerate", "transformers", "trl", "pyarrow", "packaging"]
    if args.profile == "train":
        required.append("peft")
    if args.profile == "vllm":
        required.append("vllm")
    versions, issues = {}, []
    for name in required:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
            issues.append(f"Missing distribution: {name}")
            continue
        if name in targets and Version(versions[name]) != Version(targets[name]):
            issues.append(f"Target mismatch: {name} expected {targets[name]}, installed {versions[name]}")
        # Evaluate direct requirements in the actual installed wheel metadata.
        # Extras are deliberately not enabled; this is not a full pip-check replacement.
        for raw in metadata.requires(name) or []:
            req = Requirement(raw)
            if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                continue
            try:
                installed = metadata.version(req.name)
            except metadata.PackageNotFoundError:
                issues.append(f"{name} requires missing {req}")
                continue
            if req.specifier and not req.specifier.contains(installed, prereleases=True):
                issues.append(f"{name} requires {req}; installed {req.name}={installed}")
    notes = ["Metadata only: does not validate imports, CUDA runtime, driver, ABI, GPU kernels or model execution."]
    if args.profile == "vllm":
        notes.append("Upstream vLLM 0.19.1 requirements exclude transformers 5.3.x. Custom wheel metadata may differ; a metadata pass is not runtime certification.")
    print(json.dumps({"profile": args.profile, "python": platform.python_version(),
                      "platform": platform.platform(), "targets": targets,
                      "installed": versions, "issues": issues, "notes": notes}, indent=2))
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
