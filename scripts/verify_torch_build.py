#!/usr/bin/env python3
"""Validate the custom wheel without requiring a GPU."""

from pathlib import Path
import subprocess
import sys

import torch


EXPECTED_VERSION = "2.10.0+cu130.sm86.projectswift1"
if torch.__version__ != EXPECTED_VERSION:
    raise SystemExit(f"unexpected torch version: {torch.__version__}")
if torch.version.cuda != "13.0":
    raise SystemExit(f"unexpected CUDA version: {torch.version.cuda}")
architectures = torch.cuda.get_arch_list()
if architectures != ["sm_86"]:
    raise SystemExit(f"wheel is not SM86-only: {architectures}")

torch_root = Path(torch.__file__).resolve().parent
missing = []
for library in sorted((torch_root / "lib").glob("*.so*")):
    result = subprocess.run(["ldd", str(library)], check=False, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "not found" in line and "libcuda.so.1" not in line:
            missing.append(f"{library.name}: {line.strip()}")
if missing:
    raise SystemExit("unresolved wheel libraries:\n" + "\n".join(missing))
print(f"validated {torch.__version__}: {architectures}")
