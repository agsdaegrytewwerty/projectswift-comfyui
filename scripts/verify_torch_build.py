#!/usr/bin/env python3
"""Validate the custom wheel without requiring a GPU."""

from pathlib import Path
import subprocess
import sys

import torch
import torch.nn.functional as F


EXPECTED_VERSION = "2.10.0+cu130.sm86.projectswift1"
if torch.__version__ != EXPECTED_VERSION:
    raise SystemExit(f"unexpected torch version: {torch.__version__}")
if torch.version.cuda != "13.0":
    raise SystemExit(f"unexpected CUDA version: {torch.version.cuda}")
compiled_arch_flags = torch._C._cuda_getArchFlags()
architectures = compiled_arch_flags.split() if compiled_arch_flags else []
if architectures != ["sm_86"]:
    raise SystemExit(f"wheel is not SM86-only: {architectures}")
if not torch.backends.mkldnn.is_available():
    raise SystemExit("MKLDNN CPU backend is unavailable")
if not torch.backends.cuda.is_flash_attention_available():
    raise SystemExit("CUDA flash-attention backend is unavailable")
if torch.backends.cudnn.version() is None:
    raise SystemExit("cuDNN is unavailable")

# Exercise CPU fallback and the framework SDPA path without requiring a GPU.
left = torch.randn(8, 8)
right = torch.randn(8, 8)
torch.mm(left, right)
query = torch.randn(1, 1, 4, 8)
F.scaled_dot_product_attention(query, query, query)

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
