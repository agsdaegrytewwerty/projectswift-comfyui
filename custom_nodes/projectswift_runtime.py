"""ProjectSwift-only HTTP routes and latent loader bundled with the lean fork."""

import asyncio
import csv
import io
import os
import shutil
import subprocess
import tempfile
import time
from urllib.parse import urlencode, urlparse

import aiohttp
from aiohttp import web
import folder_paths
import safetensors.torch
from server import PromptServer
import torch
import zstandard as zstd


def _parse_number(value, integer=False):
    try:
        text = str(value).strip()
        if not text or text.lower() in {"n/a", "[n/a]"} or "not supported" in text.lower():
            return None
        number = float(text)
        return int(round(number)) if integer else number
    except (TypeError, ValueError):
        return None


def _nvidia_query(fields):
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode:
        return []
    return list(csv.reader(io.StringIO(result.stdout.strip())))


def _query_gpus():
    if shutil.which("nvidia-smi") is None:
        return {"ok": False, "error": "nvidia-smi not found", "gpus": [], "summary": "-", "ts": int(time.time())}

    details = {}
    for row in _nvidia_query(
        "index,clocks.current.graphics,clocks.current.sm,clocks.current.memory,"
        "clocks.max.graphics,clocks.max.sm,clocks.max.memory,pstate,temperature.gpu"
    ):
        if len(row) < 9:
            continue
        index = _parse_number(row[0], integer=True)
        if index is None:
            continue
        details[index] = {
            "clock_graphics_mhz": _parse_number(row[1], integer=True),
            "clock_sm_mhz": _parse_number(row[2], integer=True),
            "clock_memory_mhz": _parse_number(row[3], integer=True),
            "clock_max_graphics_mhz": _parse_number(row[4], integer=True),
            "clock_max_sm_mhz": _parse_number(row[5], integer=True),
            "clock_max_memory_mhz": _parse_number(row[6], integer=True),
            "pstate": row[7].strip(),
            "temperature_c": _parse_number(row[8], integer=True),
        }

    gpus = []
    for row in _nvidia_query("index,name,power.draw,power.limit,memory.used,memory.total,utilization.gpu"):
        if len(row) < 7:
            continue
        index = _parse_number(row[0], integer=True) or 0
        power_draw = _parse_number(row[2])
        power_limit = _parse_number(row[3])
        memory_used = _parse_number(row[4], integer=True)
        memory_total = _parse_number(row[5], integer=True)
        util = _parse_number(row[6], integer=True)
        detail = details.get(index, {})
        sm = detail.get("clock_sm_mhz") or detail.get("clock_graphics_mhz")
        max_sm = detail.get("clock_max_sm_mhz") or detail.get("clock_max_graphics_mhz")
        mem = detail.get("clock_memory_mhz")
        max_mem = detail.get("clock_max_memory_mhz")
        clock_parts = []
        if sm is not None:
            clock_parts.append(f"SM {sm}/{max_sm}" if max_sm is not None else f"SM {sm}")
        if mem is not None:
            clock_parts.append(f"M {mem}/{max_mem}" if max_mem is not None else f"M {mem}")
        perf_parts = [value for value in (detail.get("pstate"), f"{detail.get('temperature_c')}C" if detail.get("temperature_c") is not None else "") if value]
        gpu = {
            "index": index,
            "name": row[1].strip(),
            "power_draw_w": power_draw,
            "power_limit_w": power_limit,
            "power_text": f"{int(round(power_draw))}/{int(round(power_limit))}W" if power_draw is not None and power_limit is not None else "-",
            "memory_used_mb": memory_used,
            "memory_total_mb": memory_total,
            "vram_text": f"{memory_used}/{memory_total}MB" if memory_used is not None and memory_total is not None else "-",
            "utilization_gpu_pct": util,
            "clock_text": " ".join(clock_parts),
            "perf_text": " ".join(perf_parts),
            **detail,
        }
        gpus.append(gpu)

    summary = " | ".join(
        f"GPU{gpu['index']} {gpu['power_text']} {gpu['vram_text']} {gpu['clock_text']} {gpu['perf_text']}".strip()
        for gpu in gpus
    ) or "-"
    return {
        "ok": bool(gpus),
        "gpus": gpus,
        "summary": summary,
        "clock_text": gpus[0].get("clock_text", "") if gpus else "",
        "perf_text": gpus[0].get("perf_text", "") if gpus else "",
        "ts": int(time.time()),
    }


@PromptServer.instance.routes.get("/gpu_stats")
async def gpu_stats(request):
    data = await asyncio.to_thread(_query_gpus)
    return web.json_response(data, status=200 if data.get("ok") else 503)


@PromptServer.instance.routes.get("/gpu_stats.txt")
async def gpu_stats_text(request):
    data = await asyncio.to_thread(_query_gpus)
    return web.Response(
        text=(data.get("summary") or data.get("error") or "-") + "\n",
        status=200 if data.get("ok") else 503,
        content_type="text/plain",
    )


def _safe_storage_path(filename, subfolder, type_name):
    if filename != os.path.basename(filename) or not filename:
        raise ValueError("Invalid filename")
    base = folder_paths.get_directory_by_type(type_name or "input")
    if base is None:
        raise ValueError(f"Unsupported type: {type_name}")
    base = os.path.abspath(base)
    directory = os.path.abspath(os.path.join(base, subfolder or ""))
    if os.path.commonpath((directory, base)) != base:
        raise PermissionError("Subfolder escapes base directory")
    path = os.path.abspath(os.path.join(directory, filename))
    if os.path.commonpath((path, directory)) != directory:
        raise PermissionError("File escapes base directory")
    return path


def _allowed_peer_base(raw_url):
    parsed = urlparse(str(raw_url or "").strip())
    if parsed.scheme != "https" or not parsed.hostname or not parsed.hostname.endswith(".salad.cloud"):
        raise PermissionError("Source server host is not allowed")
    if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise ValueError("Invalid source server URL")
    return f"https://{parsed.netloc}"


@PromptServer.instance.routes.post("/projectswift/pull_input")
async def projectswift_pull_input(request):
    try:
        payload = await request.json()
        source_base = _allowed_peer_base(payload.get("source_base_url"))
        filename = str(payload.get("filename") or "")
        subfolder = str(payload.get("subfolder") or "")
        type_name = str(payload.get("type") or "input")
        destination = _safe_storage_path(filename, subfolder, type_name)
    except PermissionError:
        return web.Response(status=403)
    except Exception as error:
        return web.Response(text=str(error), status=400, content_type="text/plain")

    overwrite = str(payload.get("overwrite", "true")).lower() in {"1", "true", "yes"}
    if os.path.exists(destination) and not overwrite:
        return web.Response(text="File already exists", status=409, content_type="text/plain")
    try:
        timeout_seconds = max(10.0, min(float(payload.get("timeout_seconds", 120.0)), 600.0))
    except (TypeError, ValueError):
        timeout_seconds = 120.0
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    descriptor, temp_path = tempfile.mkstemp(prefix="projectswift-peer-input-", dir=os.path.dirname(destination))
    source_url = source_base + "/view?" + urlencode({"filename": filename, "subfolder": subfolder, "type": type_name})
    try:
        with os.fdopen(descriptor, "wb") as output:
            timeout = aiohttp.ClientTimeout(total=timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(source_url, headers={"User-Agent": "ProjectSwiftPeerPull/1.0"}) as response:
                    if response.status >= 300:
                        return web.Response(text=await response.text(), status=response.status, content_type="text/plain")
                    async for chunk in response.content.iter_chunked(1024 * 1024):
                        output.write(chunk)
        os.replace(temp_path, destination)
    except Exception as error:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass
        return web.Response(text=str(error), status=500, content_type="text/plain")
    return web.json_response({"name": os.path.basename(destination), "subfolder": subfolder, "type": type_name})


def _resolve_view_path(filename, subfolder, type_name):
    annotated_name, base = folder_paths.annotated_filepath(filename)
    if not annotated_name:
        raise FileNotFoundError("Missing filename")
    if annotated_name.startswith("/") or ".." in annotated_name:
        raise ValueError("Invalid file path")
    if base is None:
        base = folder_paths.get_directory_by_type(type_name or "output")
    if base is None:
        raise ValueError(f"Unsupported type: {type_name}")
    base = os.path.abspath(base)
    directory = os.path.abspath(os.path.join(base, subfolder or ""))
    if os.path.commonpath((directory, base)) != base:
        raise PermissionError("Subfolder escapes base directory")
    return os.path.abspath(os.path.join(directory, os.path.basename(annotated_name)))


@PromptServer.instance.routes.get("/projectswift/view_zstd")
async def projectswift_view_zstd(request):
    try:
        path = _resolve_view_path(
            request.rel_url.query.get("filename", ""),
            request.rel_url.query.get("subfolder", ""),
            request.rel_url.query.get("type", "output"),
        )
    except FileNotFoundError:
        return web.Response(status=404)
    except PermissionError:
        return web.Response(status=403)
    except Exception as error:
        return web.Response(text=str(error), status=400, content_type="text/plain")
    if not os.path.isfile(path):
        return web.Response(status=404)
    if os.path.splitext(path)[1].lower() not in {".latent", ".ckpt"}:
        return web.Response(text="Only latent and ckpt downloads support zstd route", status=400, content_type="text/plain")

    def compress():
        with open(path, "rb") as source:
            return zstd.ZstdCompressor(level=3, threads=-1).compress(source.read())

    body = await asyncio.to_thread(compress)
    return web.Response(
        body=body,
        content_type="application/zstd",
        headers={"Content-Disposition": f'filename="{os.path.basename(path)}.zst"'},
    )


def _candidate_paths(base_path):
    yield base_path
    for suffix in (".latent", ".latent.zst"):
        if not base_path.endswith(suffix):
            yield base_path + suffix


class LatentLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent_file": ("STRING", {"default": "", "multiline": False})}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load_latent"
    CATEGORY = "only/Latent"

    def resolve_latent_path(self, latent_file):
        raw_path = str(latent_file or "").strip()
        if not raw_path:
            raise FileNotFoundError("Latent filename is empty")
        if raw_path.startswith("temp/"):
            normalized = os.path.normpath(raw_path[5:])
            if normalized.startswith("..") or os.path.isabs(normalized):
                raise FileNotFoundError("Invalid latent path")
            base = os.path.abspath(folder_paths.get_temp_directory())
            candidates = _candidate_paths(os.path.join(base, normalized))
        else:
            candidates = _candidate_paths(folder_paths.get_annotated_filepath(raw_path))
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"Latent file not found: {raw_path}")

    def load_latent(self, latent_file):
        path = self.resolve_latent_path(latent_file)
        with open(path, "rb") as source:
            payload = source.read()
        if payload.startswith(b"\x28\xb5\x2f\xfd"):
            payload = zstd.ZstdDecompressor().decompress(payload)
        try:
            latent_data = safetensors.torch.load(payload)
        except Exception:
            try:
                latent_data = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
            except TypeError:
                latent_data = torch.load(io.BytesIO(payload), map_location="cpu")
        multiplier = 1.0
        samples = None
        if isinstance(latent_data, dict):
            samples = latent_data.get("samples")
            if not torch.is_tensor(samples):
                samples = latent_data.get("latent_tensor")
                if torch.is_tensor(samples) and "latent_format_version_0" not in latent_data:
                    multiplier = 1.0 / 0.18215
            if not torch.is_tensor(samples):
                samples = next((value for value in latent_data.values() if torch.is_tensor(value) and value.numel()), None)
        elif torch.is_tensor(latent_data):
            samples = latent_data
        if samples is None or not samples.numel():
            raise ValueError("Could not extract a valid latent tensor")
        if samples.ndim == 3:
            samples = samples.unsqueeze(0)
        if samples.ndim not in {4, 5}:
            raise ValueError(f"Unsupported latent shape: {tuple(samples.shape)}")
        return ({"samples": samples.float() * multiplier},)


class ProjectSwiftLatentLoader(LatentLoaderAdvanced):
    pass


NODE_CLASS_MAPPINGS = {
    "LatentLoaderAdvanced": LatentLoaderAdvanced,
    "ProjectSwiftLatentLoader": ProjectSwiftLatentLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentLoaderAdvanced": "Load Latent (Advanced)",
    "ProjectSwiftLatentLoader": "Load Latent (Advanced)",
}
