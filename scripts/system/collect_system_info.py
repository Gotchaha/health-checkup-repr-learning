"""Collect basic hardware/software info on an HPC node.

This script is intended to run inside a batch job on a GPU node.
It gathers GPU/CPU/RAM/OS information plus NVIDIA driver and cuDNN details
(with best-effort fallbacks) and writes a machine-readable JSON artifact.

The implementation is intentionally dependency-light (stdlib only).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CmdResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _read_text(path: Path, *, max_bytes: int = 2_000_000) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > max_bytes:
        data = data[:max_bytes] + b"\n<TRUNCATED>\n"
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return data.decode(errors="replace")


def _run_cmd(argv: list[str], *, timeout_s: float) -> CmdResult:
    start = time.time()
    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    end = time.time()
    return CmdResult(
        argv=argv,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        duration_s=end - start,
    )


def _parse_key_value_lines(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        if k:
            out[k] = v
    return out


def _collect_os_info() -> dict[str, Any]:
    os_release_path = Path("/etc/os-release")
    os_release_text = _read_text(os_release_path)
    os_release: dict[str, str] | None = None
    if os_release_text is not None:
        os_release = {}
        for line in os_release_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"')
            os_release[k] = v

    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "os_release": os_release,
    }


def _collect_cpu_info(*, timeout_s: float) -> dict[str, Any]:
    info: dict[str, Any] = {}

    if shutil.which("lscpu"):
        res = _run_cmd(["lscpu"], timeout_s=timeout_s)
        info["lscpu"] = {
            "ok": res.ok,
            "returncode": res.returncode,
            "duration_s": res.duration_s,
            "stderr": res.stderr.strip(),
            "raw": res.stdout,
            "parsed": _parse_key_value_lines(res.stdout) if res.ok else None,
        }

    if shutil.which("nproc"):
        res = _run_cmd(["nproc"], timeout_s=timeout_s)
        info["nproc"] = {
            "ok": res.ok,
            "returncode": res.returncode,
            "duration_s": res.duration_s,
            "stderr": res.stderr.strip(),
            "raw": res.stdout.strip(),
            "value": int(res.stdout.strip()) if res.ok and res.stdout.strip().isdigit() else None,
        }

    cpuinfo_text = _read_text(Path("/proc/cpuinfo"))
    if cpuinfo_text is not None:
        model_name: str | None = None
        for line in cpuinfo_text.splitlines():
            if line.startswith("model name") and ":" in line:
                _, v = line.split(":", 1)
                model_name = v.strip()
                break
        info["proc_cpuinfo"] = {
            "model_name": model_name,
            "available": True,
        }
    else:
        info["proc_cpuinfo"] = {"available": False}

    return info


def _collect_memory_info(*, timeout_s: float) -> dict[str, Any]:
    info: dict[str, Any] = {}

    meminfo_text = _read_text(Path("/proc/meminfo"))
    if meminfo_text is not None:
        parsed: dict[str, int] = {}
        for line in meminfo_text.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            v = v.strip()
            m = re.match(r"^(\d+)\s+kB$", v)
            if m:
                parsed[k.strip()] = int(m.group(1)) * 1024
        info["proc_meminfo"] = {
            "available": True,
            "raw": meminfo_text,
            "bytes": parsed,
        }
    else:
        info["proc_meminfo"] = {"available": False}

    if shutil.which("free"):
        res = _run_cmd(["free", "-b"], timeout_s=timeout_s)
        info["free"] = {
            "ok": res.ok,
            "returncode": res.returncode,
            "duration_s": res.duration_s,
            "stderr": res.stderr.strip(),
            "raw": res.stdout,
        }

    return info


def _parse_nvidia_smi_header(text: str) -> dict[str, str | None]:
    driver: str | None = None
    cuda: str | None = None
    m = re.search(r"Driver Version:\s*([0-9.]+)", text)
    if m:
        driver = m.group(1)
    m = re.search(r"CUDA Version:\s*([0-9.]+)", text)
    if m:
        cuda = m.group(1)
    return {"driver_version": driver, "cuda_version": cuda}


def _collect_gpu_info(*, timeout_s: float) -> dict[str, Any]:
    info: dict[str, Any] = {"available": False}

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        info["reason"] = "nvidia-smi not found"
        return info

    info["available"] = True

    res_l = _run_cmd([nvidia_smi, "-L"], timeout_s=timeout_s)
    info["nvidia_smi_L"] = {
        "ok": res_l.ok,
        "returncode": res_l.returncode,
        "duration_s": res_l.duration_s,
        "stderr": res_l.stderr.strip(),
        "raw": res_l.stdout,
    }

    res = _run_cmd([nvidia_smi], timeout_s=timeout_s)
    header = _parse_nvidia_smi_header(res.stdout) if res.ok else {"driver_version": None, "cuda_version": None}
    info["nvidia_smi"] = {
        "ok": res.ok,
        "returncode": res.returncode,
        "duration_s": res.duration_s,
        "stderr": res.stderr.strip(),
        "header": header,
    }

    query_fields = [
        "index",
        "uuid",
        "name",
        "pci.bus_id",
        "memory.total",
    ]
    res_q = _run_cmd(
        [
            nvidia_smi,
            f"--query-gpu={','.join(query_fields)}",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=timeout_s,
    )

    gpus: list[dict[str, Any]] | None = None
    if res_q.ok:
        gpus = []
        for line in res_q.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != len(query_fields):
                continue
            gpu = dict(zip(query_fields, parts, strict=True))
            try:
                gpu["index"] = int(gpu["index"])  # type: ignore[assignment]
            except Exception:
                pass
            try:
                gpu["memory.total"] = int(gpu["memory.total"])  # MiB  # type: ignore[assignment]
            except Exception:
                pass
            gpus.append(gpu)

    info["nvidia_smi_query"] = {
        "ok": res_q.ok,
        "returncode": res_q.returncode,
        "duration_s": res_q.duration_s,
        "stderr": res_q.stderr.strip(),
        "fields": query_fields,
        "gpus": gpus,
        "raw": res_q.stdout,
    }

    return info


def _safe_pkg_version(dist_name: str) -> str | None:
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _collect_python_cuda_stack() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "packages": {
            "torch": _safe_pkg_version("torch"),
            "nvidia-cudnn-cu12": _safe_pkg_version("nvidia-cudnn-cu12"),
            "nvidia-cuda-runtime-cu12": _safe_pkg_version("nvidia-cuda-runtime-cu12"),
        },
        "torch": None,
    }

    try:
        import torch  # type: ignore

        torch_info: dict[str, Any] = {
            "version": getattr(torch, "__version__", None),
            "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
            "cuda_is_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "cudnn_is_available": bool(torch.backends.cudnn.is_available()),
            "cudnn_version": int(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
        }
        if torch_info["cuda_is_available"] and torch_info["cuda_device_count"]:
            devices: list[dict[str, Any]] = []
            for i in range(torch_info["cuda_device_count"]):
                devices.append(
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "capability": tuple(torch.cuda.get_device_capability(i)),
                    }
                )
            torch_info["devices"] = devices

        info["torch"] = torch_info
    except Exception as e:
        info["torch"] = {"error": f"{type(e).__name__}: {e}"}

    return info


def _collect_context() -> dict[str, Any]:
    keys = [
        "JOB_ID",
        "JOB_NAME",
        "NSLOTS",
        "CUDA_VISIBLE_DEVICES",
        "CONDA_DEFAULT_ENV",
        "CONDA_PREFIX",
        "SGE_TASK_ID",
        "QUEUE",
        "PE_HOSTFILE",
    ]

    env: dict[str, str] = {}
    for k in keys:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v

    return {
        "timestamp_utc": _utc_now_iso(),
        "hostname": socket.gethostname(),
        "user": os.environ.get("USER") or os.environ.get("LOGNAME"),
        "cwd": os.getcwd(),
        "env": env,
    }


def collect_system_info(*, timeout_s: float) -> dict[str, Any]:
    return {
        "context": _collect_context(),
        "os": _collect_os_info(),
        "cpu": _collect_cpu_info(timeout_s=timeout_s),
        "memory": _collect_memory_info(timeout_s=timeout_s),
        "gpu": _collect_gpu_info(timeout_s=timeout_s),
        "python_cuda_stack": _collect_python_cuda_stack(),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write JSON output to this path (directories will be created).",
    )
    p.add_argument(
        "--timeout-s",
        type=float,
        default=15.0,
        help="Timeout (seconds) for external commands like nvidia-smi/lscpu/free.",
    )
    p.add_argument(
        "--also-print",
        action="store_true",
        help="Also print a short human-readable summary to stdout.",
    )
    return p.parse_args()


def _print_summary(data: dict[str, Any]) -> None:
    ctx = data.get("context", {})
    gpu = data.get("gpu", {})
    py = data.get("python_cuda_stack", {})

    print("==============================================")
    print("System Info Summary")
    print("==============================================")
    print(f"Timestamp (UTC): {ctx.get('timestamp_utc')}")
    print(f"Host: {ctx.get('hostname')}")

    header = (gpu.get("nvidia_smi", {}) or {}).get("header", {}) if isinstance(gpu, dict) else {}
    if isinstance(header, dict):
        print(f"NVIDIA Driver: {header.get('driver_version')}")
        print(f"CUDA (nvidia-smi): {header.get('cuda_version')}")

    torch_info = py.get("torch") if isinstance(py, dict) else None
    if isinstance(torch_info, dict):
        print(f"Torch: {torch_info.get('version')}")
        print(f"Torch CUDA: {torch_info.get('cuda_version')}")
        print(f"cuDNN (via torch): {torch_info.get('cudnn_version')}")

    gpus = ((gpu.get("nvidia_smi_query", {}) or {}).get("gpus") if isinstance(gpu, dict) else None) or []
    if isinstance(gpus, list) and gpus:
        print(f"GPU Count: {len(gpus)}")
        for g in gpus:
            if isinstance(g, dict):
                print(f"- GPU {g.get('index')}: {g.get('name')} ({g.get('memory.total')} MiB)")
    else:
        print("GPU Count: 0 (or unavailable)")


def main() -> int:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    LOGGER.info("Collecting system info...")
    data = collect_system_info(timeout_s=args.timeout_s)

    if args.also_print:
        _print_summary(data)

    if args.out_json is None:
        json.dump(data, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    try:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as e:
        LOGGER.exception("Failed to write JSON output: %s", args.out_json)
        return 2

    LOGGER.info("Wrote JSON: %s", args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
