#!/usr/bin/env python3
"""Create the API-only ProjectSwift source tree used in release bundles."""

import argparse
from pathlib import Path
import shutil


REMOVE_TOP_LEVEL = {
    ".ci",
    ".github",
    ".git",
    ".pytest_cache",
    "alembic_db",
    "blueprints",
    "comfy_api_nodes",
    "script_examples",
    "tests",
    "tests-unit",
    "tests-projectswift",
}
KEEP_EXTRAS = {
    "__init__.py",
    "nodes_custom_sampler.py",
    "nodes_images.py",
    "nodes_latent.py",
    "nodes_mask.py",
    "nodes_post_processing.py",
    "nodes_primitive.py",
    "nodes_string.py",
}


def ignored(directory, names):
    directory = Path(directory)
    ignored_names = set()
    if directory.name == "projectswift-comfyui" or (directory / "main.py").is_file():
        ignored_names.update(name for name in names if name in REMOVE_TOP_LEVEL)
    ignored_names.update(name for name in names if name == "__pycache__" or name.endswith((".pyc", ".pyo")))
    return ignored_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=ignored)

    for relative in ("app/assets", "app/database"):
        shutil.rmtree(destination / relative, ignore_errors=True)

    extras = destination / "comfy_extras"
    for child in extras.iterdir():
        if child.name not in KEEP_EXTRAS:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    tokenizers = destination / "comfy" / "text_encoders"
    for tokenizer_dir in tokenizers.glob("*tokenizer*"):
        if tokenizer_dir.is_dir():
            shutil.rmtree(tokenizer_dir)

    custom_nodes = destination / "custom_nodes"
    for child in custom_nodes.iterdir():
        if child.name not in {"projectswift_runtime.py", "__init__.py"}:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    empty_frontend = destination / "web_empty"
    empty_frontend.mkdir(exist_ok=True)
    (empty_frontend / "index.html").write_text("ProjectSwift ComfyUI API runtime\n", encoding="utf-8")


if __name__ == "__main__":
    main()
