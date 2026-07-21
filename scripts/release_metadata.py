#!/usr/bin/env python3
"""Generate dependency inventory and immutable release metadata."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(output):
    output.mkdir(parents=True, exist_ok=True)
    components = []
    licenses = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if not name:
            continue
        license_name = (
            distribution.metadata.get("License-Expression")
            or distribution.metadata.get("License")
            or "UNKNOWN"
        ).strip()
        component = {
            "type": "library",
            "name": name,
            "version": version,
            "purl": f"pkg:pypi/{name.lower().replace('_', '-')}@{version}",
        }
        if license_name and license_name != "UNKNOWN":
            component["licenses"] = [{"license": {"name": license_name}}]
        components.append(component)
        licenses.append({"name": name, "version": version, "license": license_name})
    components.sort(key=lambda item: item["name"].lower())
    licenses.sort(key=lambda item: item["name"].lower())
    (output / "sbom.cdx.json").write_text(
        json.dumps({
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "version": 1,
            "components": components,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "license-inventory.json").write_text(
        json.dumps(licenses, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def finalize(args):
    dist = args.dist
    assets = {}
    excluded = {"SHA256SUMS", "manifest.json", "release-notes.md"}
    for path in sorted(dist.iterdir()):
        if path.is_file() and path.name not in excluded:
            assets[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}

    lock_path = args.lock
    manifest = {
        "release": args.release,
        "validation": "CPU/import validated; GPU canary pending",
        "upstream": {
            "repository": "https://github.com/comfyanonymous/ComfyUI.git",
            "commit": args.upstream_commit,
        },
        "fork_commit": args.fork_commit,
        "pytorch": {
            "repository": "https://github.com/pytorch/pytorch.git",
            "commit": args.pytorch_commit,
            "version": args.pytorch_version,
            "python_abi": "cp312-cp312",
            "cuda": "13.0",
            "architectures": ["sm_86"],
            "ptx_fallback": False,
            "builder_image": args.builder_image,
        },
        "dependency_lock": {"sha256": sha256(lock_path), "file": lock_path.name},
        "baseline_bytes": args.baseline_bytes,
        "runtime_bundle_bytes": args.runtime_bytes,
        "saved_bytes": args.baseline_bytes - args.runtime_bytes,
        "assets": assets,
    }
    manifest_path = dist / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checksum_paths = [path for path in sorted(dist.iterdir()) if path.is_file() and path.name != "SHA256SUMS"]
    (dist / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )

    mib = 1024 * 1024
    saved = args.baseline_bytes - args.runtime_bytes
    percent = saved / args.baseline_bytes * 100 if args.baseline_bytes else 0
    (dist / "release-notes.md").write_text(
        "\n".join([
            "ProjectSwift image-only ComfyUI runtime.",
            "",
            "**CPU/import validated; GPU canary pending.** Production S1-S15 remain on the old runtime.",
            "",
            f"- Baseline runtime: {args.baseline_bytes:,} bytes ({args.baseline_bytes / mib:.1f} MiB)",
            f"- Lean runtime bundle: {args.runtime_bytes:,} bytes ({args.runtime_bytes / mib:.1f} MiB)",
            f"- Cold-download saving excluding the 6.91 GiB checkpoint: {saved:,} bytes ({saved / mib:.1f} MiB, {percent:.2f}%)",
            f"- PyTorch: `{args.pytorch_version}`, CUDA 13.0, CPython 3.12, `sm_86` only, no PTX fallback",
            f"- ComfyUI upstream commit: `{args.upstream_commit}`",
            "",
        ]),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument("output", type=Path)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--dist", type=Path, required=True)
    finalize_parser.add_argument("--lock", type=Path, required=True)
    finalize_parser.add_argument("--release", required=True)
    finalize_parser.add_argument("--upstream-commit", required=True)
    finalize_parser.add_argument("--fork-commit", required=True)
    finalize_parser.add_argument("--pytorch-commit", required=True)
    finalize_parser.add_argument("--pytorch-version", required=True)
    finalize_parser.add_argument("--builder-image", required=True)
    finalize_parser.add_argument("--baseline-bytes", type=int, required=True)
    finalize_parser.add_argument("--runtime-bytes", type=int, required=True)
    args = parser.parse_args()
    if args.command == "inventory":
        inventory(args.output)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
