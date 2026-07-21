#!/usr/bin/env python3
"""CPU/import validation for the ProjectSwift release bundle."""

import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.argv = [
    str(ROOT / "main.py"),
    "--cpu",
    "--disable-api-nodes",
    "--disable-metadata",
    "--front-end-root",
    str(ROOT / "web_empty"),
]

import comfy.options
comfy.options.enable_args_parsing()

import folder_paths
import nodes
import server
import torch


def require_optional_dependencies_absent():
    for package in ("av", "kornia", "scipy", "sqlalchemy", "alembic", "sentencepiece"):
        if importlib.util.find_spec(package) is not None:
            raise AssertionError(f"optional dependency unexpectedly installed: {package}")


async def initialize_nodes():
    loop = asyncio.get_running_loop()
    server.PromptServer(loop)
    failures = await nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False)
    if failures:
        raise AssertionError(f"built-in node import failures: {failures}")


def validate_required_nodes():
    workflow_nodes = json.loads((ROOT / "projectswift/workflows/required_nodes.json").read_text())
    required = set().union(*map(set, workflow_nodes.values()))
    missing = required.difference(nodes.NODE_CLASS_MAPPINGS)
    if missing:
        raise AssertionError(f"missing ProjectSwift nodes: {sorted(missing)}")
    allowed = {
        "CFGGuider", "CLIPTextEncode", "CheckpointLoaderSimple", "DisableNoise",
        "EmptyLatentImage", "ImageBlur", "ImageScale", "ImageToMask",
        "KSamplerSelect", "KarrasScheduler", "LatentLoaderAdvanced",
        "LatentUpscaleBy", "LoadImage", "LoadLatent", "MaskToImage",
        "PreviewImage", "PrimitiveFloat", "PrimitiveInt", "PrimitiveString",
        "PrimitiveStringMultiline", "ProjectSwiftLatentLoader", "RandomNoise",
        "SamplerCustomAdvanced", "SaveAnimatedWEBP", "SaveImage", "SaveLatent",
        "SplitSigmas", "StringConcatenate", "VAEDecode", "VAEDecodeTiled",
        "VAEEncode", "VAEEncodeForInpaint",
    }
    unexpected = set(nodes.NODE_CLASS_MAPPINGS).difference(allowed)
    if unexpected:
        raise AssertionError(f"unexpected nodes in lean registry: {sorted(unexpected)}")


def validate_image_paths():
    from comfy.k_diffusion.sampling import get_sigmas_karras
    sigmas = get_sigmas_karras(4, 0.01, 14.0)
    if tuple(sigmas.shape) != (5,):
        raise AssertionError("Karras schedule shape changed")

    image = torch.rand((1, 16, 16, 3), dtype=torch.float32)
    blurred = nodes.NODE_CLASS_MAPPINGS["ImageBlur"].execute(image, 1, 1.0)[0]
    if blurred.shape != image.shape:
        raise AssertionError("ImageBlur changed image shape")

    from comfy_api.latest._io import FolderType
    from comfy_api.latest._ui import ImageSaveHelper
    with tempfile.TemporaryDirectory() as output_dir:
        folder_paths.set_output_directory(output_dir)
        result = ImageSaveHelper.save_animated_webp(
            torch.rand((2, 8, 8, 3)), "verify", FolderType.output, None,
            fps=6.0, lossless=True, quality=80, method=0,
        )
        if not (Path(output_dir) / result.filename).is_file():
            raise AssertionError("animated WebP was not written")


def validate_latent_zstd():
    import safetensors.torch
    import zstandard as zstd

    with tempfile.TemporaryDirectory() as temp_dir:
        folder_paths.set_temp_directory(temp_dir)
        payload = safetensors.torch.save({"samples": torch.ones((1, 4, 2, 2))})
        compressed = zstd.ZstdCompressor().compress(payload)
        path = Path(temp_dir) / "verify.latent.zst"
        path.write_bytes(compressed)
        loaded = nodes.NODE_CLASS_MAPPINGS["ProjectSwiftLatentLoader"]().load_latent("temp/verify.latent.zst")[0]
        if tuple(loaded["samples"].shape) != (1, 4, 2, 2):
            raise AssertionError("compressed latent shape changed")


def validate_sdxl_detection():
    import comfy.model_detection
    import comfy.supported_models

    config = dict(comfy.supported_models.SDXL.unet_config)
    config["in_channels"] = 4
    detected = comfy.model_detection.model_config_from_unet_config(config)
    if not isinstance(detected, comfy.supported_models.SDXL):
        raise AssertionError(f"synthetic SDXL config detected as {type(detected).__name__}")


async def main():
    require_optional_dependencies_absent()
    await initialize_nodes()
    validate_required_nodes()
    validate_image_paths()
    validate_latent_zstd()
    validate_sdxl_detection()
    print("ProjectSwift lean runtime verification passed")


if __name__ == "__main__":
    asyncio.run(main())
