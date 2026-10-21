"""
ONNX Runtime session cache — one lazily-created InferenceSession per model,
reused across warm invocations.

MUST import lib.ort_init (cpuinfo fix) before onnxruntime — enforced here so
no caller can get the order wrong.

onnxruntime is PINNED to 1.18.0 for the whole container: the community FP16
LLM graphs hard-crash newer ORT versions at session-init (SimplifiedLayerNorm
fusion bug, reproduced directly this project on 1.23.2), and every other
model in the pipeline was individually verified to load AND run correctly
under 1.18.0. One runtime version, verified everywhere — not a per-model mix.
"""

from __future__ import annotations

import logging
import os

from lib import ort_init
# The cpuinfo fix is a CALL, not an import side effect — ort_init defines
# ensure_cpuinfo_files() but runs nothing at import time, so importing it
# alone buys no protection. Invoke it here, before onnxruntime is imported:
# ORT's arm64 cpuinfo probe aborts the process (terminate() with "Attempt to
# use DefaultLogger but none has been registered") when
# /sys/devices/system/cpu/possible is missing, which is unrecoverable and
# nearly undiagnosable from the Lambda log. patch.txt in the image is the
# primary fix; this is the cheap idempotent backstop it was always meant
# to be. On Lambda /sys is read-only, so if patch.txt DID land this is a
# no-op (file exists) and if it didn't the write fails and is logged at
# debug — either way it cannot make things worse.
ort_init.ensure_cpuinfo_files()

import onnxruntime as ort  # noqa: E402  (must follow the cpuinfo fix above)

log = logging.getLogger(__name__)

_SESSIONS: dict[str, ort.InferenceSession] = {}


def _default_intra_op_threads() -> int:
    """Real vCPU count for this Lambda, derived from the memory allocation
    (AWS_LAMBDA_FUNCTION_MEMORY_SIZE is set by the runtime).

    MUST be explicit: patch.txt advertises 6 CPUs ("0-5") to keep ORT's
    cpuinfo probe from crashing, so ORT's default spins 6 threads even on a
    1-3 vCPU function. Measured on the flow estimator with 3 real CPUs:
    default (6 threads) 7.33 s/call vs explicit 3 threads 2.46 s/call —
    a 3x penalty from pure thread oversubscription (the
    pthread_setaffinity_np errors in every Lambda log were this).
    """
    mem = os.environ.get("AWS_LAMBDA_FUNCTION_MEMORY_SIZE")
    if mem:
        return ort_init.get_num_cpus(int(mem))
    return ort_init.get_num_cpus()


def get_session(model_path: str, intra_op_threads: int | None = None) -> ort.InferenceSession:
    """Return a cached InferenceSession for model_path, creating it on first
    use. intra_op_threads defaults to the function's REAL vCPU count."""
    key = os.path.abspath(model_path)
    if key not in _SESSIONS:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        threads = intra_op_threads or _default_intra_op_threads()
        so.intra_op_num_threads = threads
        log.info("Creating ORT session: %s (intra_op_threads=%d)",
                 model_path, threads)
        _SESSIONS[key] = ort.InferenceSession(
            model_path, so, providers=["CPUExecutionProvider"])
    return _SESSIONS[key]


def drop_sessions() -> None:
    """Release all cached sessions (frees model memory mid-invocation when a
    stage is done with its models — e.g. LLM state before returning)."""
    _SESSIONS.clear()
