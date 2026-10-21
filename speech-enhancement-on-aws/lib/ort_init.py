# lib/ort_init.py
"""
ORT initialisation helpers — import this module BEFORE importing onnxruntime.

Problem
───────
AWS Lambda removes /sys/devices/system/cpu/possible and /present from its
sandbox. ORT's pytorch/cpuinfo hard-crashes on arm64 when these files are
missing, throwing:

    terminate called after throwing an instance of 'onnxruntime::OnnxRuntimeException'
    Attempt to use DefaultLogger but none has been registered.

The Dockerfile COPYs patch.txt (containing "0-5") into both paths as the primary
fix. This module provides two helpers:

  1. ensure_cpuinfo_files() — secondary safeguard that creates those files with
     a safe default of "0-5" if they are somehow absent. Does NOT overwrite the
     Dockerfile-installed patch.txt — that file is the authoritative source.
     Must be called before `import onnxruntime`.

  2. get_num_cpus(memory_mb) — returns the vCPU count for SessionOptions.
     Derives from the Lambda memory allocation table when memory_mb is supplied
     (most accurate); falls back to reading /sys/devices/system/cpu/possible.

Lambda vCPU allocation (arm64 and x86_64):
    1792 MB → 1 vCPU    3584 MB → 2 vCPU    5376 MB → 3 vCPU
    7168 MB → 4 vCPU    8960 MB → 5 vCPU   10240 MB → 6 vCPU
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

_CPUINFO_PATHS = (
    "/sys/devices/system/cpu/possible",
    "/sys/devices/system/cpu/present",
)


def _count_cpus_from_proc() -> int:
    """Count CPUs from /proc/cpuinfo — works even when /sys is incomplete."""
    try:
        count = 0
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("processor"):
                    count += 1
        return max(count, 1)
    except Exception:
        return 1


# Lambda vCPU allocation table (arm64 and x86_64 are identical)
# Each 1792 MB increment adds 1 vCPU. Below 1792 MB is fractional.
_LAMBDA_MB_TO_VCPU = {
    1792:  1,
    3584:  2,
    5376:  3,
    7168:  4,
    8960:  5,
    10240: 6,
}


def get_num_cpus(memory_mb: int | None = None) -> int:
    """
    Return the number of vCPUs available to this Lambda function.

    If memory_mb is provided (from context.memory_limit_in_mb), derives the
    vCPU count from the Lambda allocation table — this is the most accurate
    source since /proc/cpuinfo reports the host core count and patch.txt is
    a static build-time file.

    Falls back to reading /sys/devices/system/cpu/possible (our patch.txt)
    if memory_mb is not provided, e.g. when called outside a Lambda context.

    Lambda vCPU allocation:
        1792 MB → 1 vCPU    3584 MB → 2 vCPU    5376 MB → 3 vCPU
        7168 MB → 4 vCPU    8960 MB → 5 vCPU   10240 MB → 6 vCPU
        < 1792 MB → fractional vCPU, treated as 1
    """
    if memory_mb is not None:
        mb = int(memory_mb)
        # Find the highest allocation tier that does not exceed actual memory
        vcpus = 1
        for tier_mb, tier_vcpus in sorted(_LAMBDA_MB_TO_VCPU.items()):
            if mb >= tier_mb:
                vcpus = tier_vcpus
        log.debug(f"get_num_cpus: memory_mb={mb} → {vcpus} vCPU(s)")
        return vcpus

    # Fallback — read from patch.txt baked into the image
    try:
        with open("/sys/devices/system/cpu/possible") as f:
            cpu_range = f.read().strip()   # e.g. "0-1"
        parts = cpu_range.split("-")
        return int(parts[-1]) - int(parts[0]) + 1
    except Exception:
        log.warning(
            "Could not read /sys/devices/system/cpu/possible, defaulting to 1")
        return 1


def ensure_cpuinfo_files() -> None:
    """
    Secondary safeguard: create /sys/devices/system/cpu/possible and /present
    with content "0-5" if they are absent.

    Does NOT overwrite existing files — the Dockerfile-installed patch.txt is
    the authoritative source. Call this before importing onnxruntime.
    """
    for path in _CPUINFO_PATHS:
        try:
            if os.path.exists(path):
                continue  # already present — leave it alone
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                # safe maximum — get_num_cpus() reads this back
                f.write("0-5\n")
            log.debug(f"cpuinfo fix: created missing file '{path}' with '0-5'")
        except OSError as e:
            log.debug(f"cpuinfo fix: could not write {path}: {e}")
