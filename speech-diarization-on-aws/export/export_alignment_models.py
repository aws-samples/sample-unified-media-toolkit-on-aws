#!/usr/bin/env python3
# export_alignment_models.py
# pyright: reportMissingImports=false, reportMissingModuleSource=false
import os
import sys
import subprocess
import json
import shutil
import tarfile
import functools
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import requests
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time
import argparse
import boto3
from botocore.exceptions import ClientError, NoCredentialsError


# =============================================================================
# MODEL REGISTRY
# =============================================================================

ALL_ALIGN_MODELS: Dict[str, str] = {
    "en": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "it": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
    # 'da' REMOVED 2026-10-06: Danish Parliament/FTSpeech terms forbid
    # derivatives (license audit, THIRD_PARTY_LICENSES.md)
    "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
    # 'vi' REMOVED 2026-10-06: CC-BY-NC-4.0 (non-commercial)
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "ur": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
    "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "ca": "softcatala/wav2vec2-large-xlsr-catala",
    "ml": "gvs/wav2vec2-large-xlsr-malayalam",
    # "no": "NbAiLab/nb-wav2vec2-1b-bokmaal-v2", # TOO HEAVY
    # "nn": "NbAiLab/nb-wav2vec2-1b-nynorsk", # TOO HEAVY
    "no": "NbAiLab/nb-wav2vec2-300m-bokmaal",
    "nn": "NbAiLab/nb-wav2vec2-300m-nynorsk",
    "sk": "comodoro/wav2vec2-xls-r-300m-sk-cv8",
    "sl": "anton-l/wav2vec2-large-xlsr-53-slovenian",
    "hr": "classla/wav2vec2-xls-r-parlaspeech-hr",
    "ro": "gigant/romanian-wav2vec2",
    "eu": "stefan-it/wav2vec2-large-xlsr-53-basque",
    "gl": "ifrz/wav2vec2-large-xlsr-galician",
    "ka": "xsway/wav2vec2-large-xlsr-georgian",
    "lv": "jimregan/wav2vec2-large-xlsr-latvian-cv",
    "tl": "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official",
    "sv": "KBLab/wav2vec2-large-voxrex-swedish",
}

# =============================================================================
# DEFAULTS
# =============================================================================

BASE_DIR = "./assets/models/align"
S3_KEY_PREFIX = "opensource_models/align"

MEM_ESTIMATE_EXPORT_MB = 1800
MEM_ESTIMATE_QUANTIZE_MB = 600
MEM_ESTIMATE_VERIFY_MB = 400
MEM_ESTIMATE_PER_WORKER_MB = MEM_ESTIMATE_EXPORT_MB
MEM_SAFETY_MARGIN_MB = 512

DEFAULT_WORKERS = 1
DEFAULT_KEEP_ONNX = False
MSE_THRESHOLD = 0.1
EXPORT_TIMEOUT = 600

ATTEMPT_STDERR_LINES = 15


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class ConversionResult:
    lang:              str
    model_id:          str
    success:           bool
    skipped:           bool = False
    error:             Optional[str] = None
    duration_sec:      float = 0.0
    onnx_size_mb:      float = 0.0
    quantized_size_mb: float = 0.0
    package_size_mb:   float = 0.0
    uploaded:          bool = False
    s3_uri:            Optional[str] = None
    output_shape:      Optional[str] = None
    mse:               Optional[float] = None
    peak_ram_mb:       float = 0.0


# =============================================================================
# HELPERS
# =============================================================================

def log(lang: str, msg: str):
    print(f"[{lang:>4}] {msg}", flush=True)


def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0.0


def available_ram_mb() -> float:
    return psutil.virtual_memory().available / (1024 * 1024)


def total_ram_mb() -> float:
    return psutil.virtual_memory().total / (1024 * 1024)


def safe_worker_count() -> int:
    usable = available_ram_mb() - MEM_SAFETY_MARGIN_MB
    workers = max(1, int(usable // MEM_ESTIMATE_PER_WORKER_MB))
    return workers


def is_already_converted(lang: str) -> bool:
    quant_dir = os.path.join(BASE_DIR, lang, "quantized")
    return (
        os.path.exists(os.path.join(quant_dir, "model_quantized.onnx"))
        and os.path.exists(os.path.join(quant_dir, "vocab.json"))
    )

def is_packaged(lang: str) -> bool:
    """Return True if model.tar.gz already exists for this language."""
    return os.path.exists(os.path.join(BASE_DIR, lang, "model.tar.gz"))

def is_uploaded(lang: str, bucket: str) -> bool:
    """Return True if model.tar.gz already exists at the S3 destination."""
    s3_key = f"{S3_KEY_PREFIX}/{lang}/model.tar.gz"
    try:
        boto3.client("s3").head_object(Bucket=bucket, Key=s3_key)
        return True
    except Exception:
        return False


def _make_progress_callback(lang: str, total_bytes: int):
    """Upload progress callback — logs at 0 %, 25 %, 50 %, 75 %, 100 %."""
    uploaded    = [0]
    last_mile   = [-1]

    def callback(chunk: int) -> None:
        uploaded[0] += chunk
        pct       = int(100 * uploaded[0] / total_bytes)
        milestone = (pct // 25) * 25
        if milestone > last_mile[0]:
            last_mile[0] = milestone
            mb_done  = uploaded[0] / (1024 * 1024)
            mb_total = total_bytes  / (1024 * 1024)
            log(lang, f"   ↑ {mb_done:.1f}/{mb_total:.1f} MB  ({pct}%)")

    return callback


def stage_upload(
    lang: str,
    package_path: str,
    bucket: str,
    skip_if_exists: bool = True,
) -> Tuple[bool, Optional[str]]:
    """
    Upload model.tar.gz to:
      s3://{bucket}/opensource_models/align/{lang}/model.tar.gz

    Returns (success, s3_uri).
    Skips silently when the object already exists and skip_if_exists=True.
    boto3 upload_file uses multipart automatically for large files.
    """
    if not os.path.exists(package_path):
        log(lang, f"❌ Package not found: {package_path}")
        return False, None

    s3_key  = f"{S3_KEY_PREFIX}/{lang}/model.tar.gz"
    s3_uri  = f"s3://{bucket}/{s3_key}"
    pkg_mb  = file_size_mb(package_path)

    if skip_if_exists and is_uploaded(lang, bucket):
        log(lang, f"⏭️  Already uploaded → {s3_uri}")
        return True, s3_uri

    log(lang, f"☁️  Uploading → {s3_uri}  ({pkg_mb:.1f} MB)...")
    try:
        s3 = boto3.client("s3")
        s3.upload_file(
            package_path,
            bucket,
            s3_key,
            ExtraArgs={"ContentType": "application/gzip"},
            Callback=_make_progress_callback(lang, os.path.getsize(package_path)),
        )
        log(lang, f"✅ Uploaded → {s3_uri}")
        return True, s3_uri

    except NoCredentialsError:
        log(lang, "❌ No AWS credentials — set AWS_ACCESS_KEY_ID / "
                  "AWS_SECRET_ACCESS_KEY or attach an IAM role")
        return False, None
    except ClientError as e:
        log(lang, f"❌ S3 ClientError: {e}")
        return False, None
    except Exception as e:
        log(lang, f"❌ Upload error: {e}")
        return False, None

def find_model_onnx(search_dir: str) -> Optional[str]:
    candidates = [
        os.path.join(search_dir, "model.onnx"),
        os.path.join(search_dir, "model", "model.onnx"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    for root, _, files in os.walk(search_dir):
        for f in files:
            if f == "model.onnx":
                return os.path.join(root, f)
    return None


# =============================================================================
# OUTPUT HELPERS
# =============================================================================
_PYTHON_EXPORT_SCRIPT = """\
import os
import sys
import warnings
import inspect

model_id = sys.argv[1]
out_dir  = sys.argv[2]
trust_rc = len(sys.argv) > 3 and sys.argv[3].lower() == "true"

# Force eager attention before any model/transformers import
os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "eager"

import torch

# Patch _check_trace to absorb TracerWarning-turned-exception.
# The export file is already on disk when _check_trace runs; we only skip the
# post-hoc consistency check that would spuriously fail on this known artifact.
try:
    import torch.onnx.utils as _ou
    if hasattr(_ou, "_check_trace"):
        _orig_ct = _ou._check_trace

        def _lenient_ct(*a, **kw):
            try:
                return _orig_ct(*a, **kw)
            except Exception as _e:
                _ename = type(_e).__name__
                _emsg  = str(_e).lower()
                # Absorb only if this is a TracerWarning promoted to exception
                if "TracerWarning" in _ename or (
                    "tracer" in _emsg and "warning" in _emsg
                ):
                    return   # safe: audio frames are always > 1 token
                raise        # re-raise everything else unchanged

        _ou._check_trace = _lenient_ct
except Exception as _pe:
    print("warn: _check_trace patch failed:", _pe, file=sys.stderr)

# Belt-and-suspenders: suppress at the warnings module level too
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings(
    "ignore", message=".*Converting a tensor to a Python boolean.*"
)

try:
    from optimum.exporters.onnx import main_export
except ImportError as _ie:
    print("optimum.exporters.onnx not available:", _ie, file=sys.stderr)
    sys.exit(1)

# Build call-kwargs that match the installed optimum version.
# inspect.signature lets us skip unsupported parameters cleanly.
_sig    = inspect.signature(main_export)
_params = _sig.parameters

_base = {
    "model_name_or_path": model_id,
    "output":             out_dir,
    "task":               "automatic-speech-recognition",
    "trust_remote_code":  trust_rc,
}
if "monolith" in _params:
    _base["monolith"] = True

# Variant 1: with attn_implementation (preferred — suppresses SDPA in the model)
# Variant 2: without (fallback if this model raises TypeError for that kwarg)
_variants = [_base]
if "model_kwargs" in _params:
    _with_attn = dict(_base, model_kwargs={"attn_implementation": "eager"})
    _variants  = [_with_attn, _base]   # try attn first, plain second

_last_err = None
for _kw in _variants:
    try:
        main_export(**_kw)
        sys.exit(0)
    except SystemExit as _se:
        sys.exit(_se.code if _se.code is not None else 0)
    except Exception as _ex:
        _last_err = _ex
        # Only retry next variant if this one failed due to attn_implementation
        if "attn_implementation" not in str(_ex).lower():
            break   # unrelated error — do not mask it with a retry

print("Export error:", _last_err, file=sys.stderr, flush=True)
import traceback
traceback.print_exc(file=sys.stderr)
sys.exit(1)
"""


def _log_attempt_stderr(lang: str, stdout: str, stderr: str,
                        n: int = ATTEMPT_STDERR_LINES) -> None:
    """Print the last n non-empty lines of a single failed attempt's output."""
    lines = [ln for ln in (stdout + stderr).splitlines() if ln.strip()]
    if not lines:
        return
    snippet = lines[-n:]
    log(lang, f"      ┌─ last {len(snippet)} output lines ─")
    for line in snippet:
        log(lang, f"      │ {line}")
    log(lang, "      └─")


def _print_full_stderr(lang: str, stdout: str, stderr: str) -> None:
    """Dump last 50 lines of stdout+stderr after ALL attempts (CLI + Python) fail."""
    if stdout.strip():
        log(lang, "--- stdout (last 50 lines) ---")
        for line in stdout.strip().splitlines()[-50:]:
            log(lang, f"  {line}")
    if stderr.strip():
        log(lang, "--- stderr (last 50 lines) ---")
        for line in stderr.strip().splitlines()[-50:]:
            log(lang, f"  {line}")


# =============================================================================
# PIPELINE STAGES
# =============================================================================

@functools.lru_cache(maxsize=1)
def get_optimum_version() -> Tuple[int, int]:
    """Return (major, minor) of installed optimum."""
    try:
        import optimum
        parts = optimum.__version__.split(".")
        return int(parts[0]), int(parts[1])
    except Exception:
        return 1, 0


@functools.lru_cache(maxsize=1)
def get_optimum_flags() -> Dict[str, bool]:
    """
    Probe optimum-cli --help to discover available flags.
    Cached — runs once at startup.
    """
    major, minor = get_optimum_version()
    print(f"🔍 Probing optimum-cli {major}.{minor} capabilities...", flush=True)

    try:
        result = subprocess.run(  # nosec B603
            ["optimum-cli", "export", "onnx", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        help_text = result.stdout + result.stderr
    except Exception as e:
        print(f"   ⚠️  Could not probe optimum-cli: {e}", flush=True)
        help_text = ""

    flags = {
        "attn-implementation": "--attn-implementation" in help_text,
        "model-kwargs":        "--model-kwargs" in help_text,
        "trust-remote-code":   "--trust-remote-code" in help_text,
        "monolith":            "--monolith" in help_text,
        "optimize":            "--optimize" in help_text,
    }

    if flags["attn-implementation"]:
        sdpa_strategy = "--attn-implementation eager  (1.x CLI flag)"
    elif flags["model-kwargs"]:
        sdpa_strategy = '--model-kwargs {"attn_implementation": "eager"}  (2.x)'
    else:
        sdpa_strategy = "TRANSFORMERS_ATTN_IMPLEMENTATION=eager  (env var fallback)"

    print(f"   optimum version       : {major}.{minor}")
    print(
        f"   --model-kwargs        : {'✅' if flags['model-kwargs']        else '❌'}")
    print(
        f"   --attn-implementation : {'✅' if flags['attn-implementation'] else '❌'}")
    print(
        f"   --trust-remote-code   : {'✅' if flags['trust-remote-code']   else '❌'}")
    print(
        f"   --monolith            : {'✅' if flags['monolith']            else '❌'}")
    print(
        f"   --optimize            : {'✅' if flags['optimize']            else '❌'}")
    print(f"   SDPA fix strategy     : {sdpa_strategy}")

    return flags


# def build_export_attempts(
#     base_cmd: List[str],
#     flags: Dict[str, bool],
# ) -> List[Tuple[List[str], str, Dict[str, str]]]:
#     """Build ordered export attempts based on detected optimum-cli capabilities."""
#     base_env = {"TRANSFORMERS_ATTN_IMPLEMENTATION": "eager"}

#     monolith_flags = ["--monolith"] if flags["monolith"] else []
#     optimize_flags = ["--optimize", "O2"] if flags["optimize"] else []
#     trust_flags = ["--trust-remote-code"] if flags["trust-remote-code"] else []

#     if flags["attn-implementation"]:
#         sdpa_flags = ["--attn-implementation", "eager"]
#     elif flags["model-kwargs"]:
#         sdpa_flags = ["--model-kwargs", '{"attn_implementation": "eager"}']
#     else:
#         sdpa_flags = []

#     base_extras = monolith_flags + optimize_flags

#     attempts = [
#         (
#             base_cmd + base_extras + sdpa_flags,
#             f"sdpa-fix{' + monolith' if monolith_flags else ''}"
#             f"{' + O2' if optimize_flags else ''}",
#             base_env,
#         ),
#         *([(
#             base_cmd + base_extras + sdpa_flags + trust_flags,
#             f"sdpa-fix + trust-remote-code"
#             f"{' + monolith' if monolith_flags else ''}"
#             f"{' + O2' if optimize_flags else ''}",
#             base_env,
#         )] if trust_flags else []),
#         (
#             base_cmd + base_extras,
#             f"plain{' + monolith' if monolith_flags else ''}"
#             f"{' + O2' if optimize_flags else ''}",
#             base_env,
#         ),
#         *([(
#             base_cmd + base_extras + trust_flags,
#             "plain + trust-remote-code",
#             base_env,
#         )] if trust_flags else []),
#         (
#             base_cmd[:],
#             "bare (env var only)",
#             base_env,
#         ),
#     ]

#     return attempts


def build_export_attempts(
    base_cmd: List[str],
    flags: Dict[str, bool],
) -> List[Tuple[List[str], str, Dict[str, str]]]:
    """
    Attempt order:
      Tier 1 — with --optimize O2   (best output quality, higher RAM cost)
                OOM on large models → falls through naturally to Tier 2
      Tier 2 — without --optimize   (memory-safe fallback for large models)
      Tier 3 — bare env-var only    (last resort)

    Small models (≤300 M params, ~1.2 GB fp32) succeed in Tier 1 and get O2.
    Large models (1 B params, ~4 GB fp32) OOM in Tier 1 and succeed in Tier 2.
    Both model sizes get the best result available given their RAM footprint.
    """
    base_env = {"TRANSFORMERS_ATTN_IMPLEMENTATION": "eager"}

    monolith_flags = ["--monolith"]          if flags["monolith"]          else []
    optimize_flags = ["--optimize", "O2"]    if flags["optimize"]          else []
    trust_flags    = ["--trust-remote-code"] if flags["trust-remote-code"] else []

    if flags["attn-implementation"]:
        sdpa_flags = ["--attn-implementation", "eager"]
    elif flags["model-kwargs"]:
        sdpa_flags = ["--model-kwargs", '{"attn_implementation": "eager"}']
    else:
        sdpa_flags = []

    def d(*parts: str) -> str:
        return " + ".join(p for p in parts if p)

    m  = "monolith" if monolith_flags else ""
    t  = "trust-rc" if trust_flags    else ""
    o2 = "O2"       if optimize_flags else ""

    # ── Tier 1: with O2 (small/medium models succeed here) ─────────────────
    # Large models hit OOM → loop continues to Tier 2 automatically.
    t1: List[Tuple[List[str], str, Dict[str, str]]] = []
    if optimize_flags:
        if sdpa_flags:
            t1.append((
                base_cmd + monolith_flags + optimize_flags + sdpa_flags,
                d("sdpa-fix", o2, m), base_env,
            ))
            if trust_flags:
                t1.append((
                    base_cmd + monolith_flags + optimize_flags + sdpa_flags + trust_flags,
                    d("sdpa-fix", o2, t, m), base_env,
                ))
        t1.append((
            base_cmd + monolith_flags + optimize_flags,
            d("plain", o2, m), base_env,
        ))
        if trust_flags:
            t1.append((
                base_cmd + monolith_flags + optimize_flags + trust_flags,
                d("plain", o2, t, m), base_env,
            ))

    # ── Tier 2: without O2 (memory-safe fallback for large models) ─────────
    t2: List[Tuple[List[str], str, Dict[str, str]]] = []
    if sdpa_flags:
        t2.append((
            base_cmd + monolith_flags + sdpa_flags,
            d("sdpa-fix", m), base_env,
        ))
        if trust_flags:
            t2.append((
                base_cmd + monolith_flags + sdpa_flags + trust_flags,
                d("sdpa-fix", t, m), base_env,
            ))
    t2.append((
        base_cmd + monolith_flags,
        d("plain", m), base_env,
    ))
    if trust_flags:
        t2.append((
            base_cmd + monolith_flags + trust_flags,
            d("plain", t, m), base_env,
        ))

    # ── Tier 3: bare env-var only (last resort) ─────────────────────────────
    t3: List[Tuple[List[str], str, Dict[str, str]]] = [
        (base_cmd[:], "bare (env var only)", base_env)
    ]

    return t1 + t2 + t3

def stage_export(lang: str, model_id: str, onnx_dir: str) -> bool:
    """
    Export HuggingFace wav2vec2 model → ONNX.

    Phase 1 — optimum-cli (5 flag combinations, fastest path).
    Phase 2 — Python API fallback (handles the wav2vec2 SDPA TracerWarning that
               optimum-cli cannot bypass via any CLI flag).

    Why Phase 2 is needed:
      torch.onnx.utils._check_trace collects torch.jit.TracerWarning instances
      and re-raises them as exceptions.  wav2vec2's SDPA integration always
      emits one such warning when exporting with dynamic axes because
        is_causal = query.shape[2] > 1 and attention_mask is None
      uses a symbolic shape dimension.  No CLI flag suppresses _check_trace.
      The Python fallback patches _check_trace directly so the known-harmless
      constant fold does not abort an otherwise valid export.
    """
    log(lang, f"📦 Exporting {model_id}  "
              f"(RAM free: {available_ram_mb():.0f}MB)...")

    flags = get_optimum_flags()
    base_cmd = [
        "optimum-cli", "export", "onnx",
        "--model", model_id,
        "--task",  "automatic-speech-recognition",
        onnx_dir,
    ]
    attempts = build_export_attempts(base_cmd, flags)

    last_stderr = ""
    last_stdout = ""

    # ── Phase 1: optimum-cli ───────────────────────────────────────────────
    for cmd, description, extra_env in attempts:
        if os.path.exists(onnx_dir):
            shutil.rmtree(onnx_dir, ignore_errors=True)
        os.makedirs(onnx_dir, exist_ok=True)

        log(lang, f"   ▶ Attempt: {description}")

        env = os.environ.copy()
        env.update(extra_env)

        try:
            subprocess.run(  # nosec B603  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=EXPORT_TIMEOUT,
                env=env,
            )
            log(lang, f"✅ Export complete [{description}]  "
                      f"(RAM free: {available_ram_mb():.0f}MB)")
            return True

        except subprocess.CalledProcessError as e:
            last_stderr = e.stderr or ""
            last_stdout = e.stdout or ""
            combined = (last_stderr + last_stdout).lower()

            is_sdpa = any(k in combined for k in [
                "tracerwarning", "sdpa_attention", "is_causal",
                "converting a tensor to a python boolean",
            ])
            is_trust = any(k in combined for k in [
                "trust_remote_code", "trust-remote-code",
            ])
            is_bad_flag = "unrecognized arguments" in combined
            is_oom = any(k in combined for k in [
                "out of memory", "oom", "killed",
            ])

            log(lang, f"   ✗ {description} failed")

            if is_bad_flag:
                for line in last_stderr.splitlines():
                    if "unrecognized" in line.lower():
                        log(lang, f"   ↳ Bad flag: {line.strip()}")
                        break
            elif is_oom:
                log(lang, "   ↳ Out of memory during export")
            elif is_sdpa:
                log(lang, "   ↳ SDPA tracing issue")
            elif is_trust:
                log(lang, "   ↳ trust-remote-code required")
            else:
                log(lang, "   ↳ (unclassified — showing last output lines)")
                _log_attempt_stderr(lang, last_stdout, last_stderr)

            continue

        except subprocess.TimeoutExpired:
            log(lang, f"❌ Export timed out (>{EXPORT_TIMEOUT}s)")
            return False

    # ── Phase 2: Python API fallback ──────────────────────────────────────
    log(lang, "⚡ All CLI attempts failed — trying Python API fallback "
              "(patches _check_trace for SDPA TracerWarning)...")

    for trust in [False, True]:
        description = "python-api" + (" + trust-remote-code" if trust else "")

        if os.path.exists(onnx_dir):
            shutil.rmtree(onnx_dir, ignore_errors=True)
        os.makedirs(onnx_dir, exist_ok=True)

        log(lang, f"   ▶ Attempt: {description}")

        env = os.environ.copy()
        env["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "eager"

        try:
            subprocess.run(  # nosec B603  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
                # sys.argv in the script: ["-c", model_id, onnx_dir, str(trust)]
                [sys.executable, "-c", _PYTHON_EXPORT_SCRIPT,
                 model_id, onnx_dir, str(trust)],
                check=True,
                capture_output=True,
                text=True,
                timeout=EXPORT_TIMEOUT,
                env=env,
            )
            log(lang, f"✅ Export complete [{description}]  "
                      f"(RAM free: {available_ram_mb():.0f}MB)")
            return True

        except subprocess.CalledProcessError as e:
            last_stderr = e.stderr or ""
            last_stdout = e.stdout or ""
            log(lang, f"   ✗ {description} failed")
            _log_attempt_stderr(lang, last_stdout, last_stderr)

        except subprocess.TimeoutExpired:
            log(lang, f"❌ Python export timed out (>{EXPORT_TIMEOUT}s)")
            return False

    log(lang, "❌ All export attempts failed (CLI + Python API)")
    _print_full_stderr(lang, last_stdout, last_stderr)
    return False


def stage_quantize(lang: str, orig_path: str, quant_path: str) -> bool:
    """
    Quantize ONNX model to QInt8.

    quant_pre_process import is inside the try block so that ImportError on
    older onnxruntime versions triggers the graceful fallback path.
    """
    log(lang, f"⚙️  Pre-processing  (RAM free: {available_ram_mb():.0f}MB)...")

    preprocessed_path = orig_path.replace(".onnx", "_preprocessed.onnx")

    try:
        from onnxruntime.quantization import quant_pre_process

        quant_pre_process(
            input_model_path=orig_path,
            output_model_path=preprocessed_path,
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=False,
            auto_merge=True,
            verbose=0,
        )
        preproc_size = file_size_mb(preprocessed_path)
        log(lang, f"✅ Pre-processing complete  "
                  f"({preproc_size:.1f} MB, RAM free: {available_ram_mb():.0f}MB)")
        gc.collect()

    except Exception as e:
        log(lang, f"⚠️  Pre-processing failed: {e}")
        log(lang, "   Falling back to quantizing original model")
        preprocessed_path = orig_path

    log(lang, f"⚖️  Quantizing  (RAM free: {available_ram_mb():.0f}MB)...")
    try:
        quantize_dynamic(
            model_input=preprocessed_path,
            model_output=quant_path,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul"],
            per_channel=False,
            reduce_range=False,
        )
        gc.collect()
        log(lang, f"✅ Quantization complete  "
                  f"(RAM free: {available_ram_mb():.0f}MB)")
        return True

    except Exception as e:
        gc.collect()
        log(lang, f"❌ Quantization failed: {e}")
        return False

    finally:
        if preprocessed_path != orig_path and os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)
            gc.collect()


def stage_verify(
    lang: str, orig_path: str, quant_path: str
) -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Verify quantized model accuracy vs original.
    Original and quantized sessions are never both in memory at the same time.
    """
    log(lang, f"🔍 Verifying  (RAM free: {available_ram_mb():.0f}MB)...")

    try:
        onnx.checker.check_model(onnx.load(quant_path))
        gc.collect()
    except Exception as e:
        log(lang, f"❌ Invalid ONNX graph: {e}")
        return False, None, None

    dummy = np.random.randn(1, 16000).astype(np.float32)

    out_orig = None
    shape_str = None
    try:
        sess = ort.InferenceSession(
            orig_path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        out_orig = sess.run(None, {inp: dummy})[0]
        shape_str = str(out_orig.shape)
        if len(out_orig.shape) != 3:
            log(lang, f"⚠️  Expected 3D output, got {len(out_orig.shape)}D")
        del sess
        gc.collect()
    except Exception as e:
        log(lang, f"❌ Original model inference failed: {e}")
        gc.collect()
        return False, None, None

    out_quant = None
    try:
        sess = ort.InferenceSession(quant_path, providers=[
                                    "CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        out_quant = sess.run(None, {inp: dummy})[0]
        del sess
        gc.collect()
    except Exception as e:
        log(lang, f"❌ Quantized model inference failed: {e}")
        gc.collect()
        return False, None, None

    mse = float(np.mean((out_orig - out_quant) ** 2))
    log(lang, f"   Output shape : {shape_str}")
    log(lang, f"   MSE          : {mse:.6f}")

    del out_orig, out_quant
    gc.collect()

    if mse < MSE_THRESHOLD:
        log(lang, "✅ Verification passed")
        return True, shape_str, mse
    else:
        log(lang, f"⚠️  MSE {mse:.4f} exceeds threshold {MSE_THRESHOLD}")
        return False, shape_str, mse


def stage_vocab(lang: str, model_id: str, onnx_dir: str, quant_dir: str):
    """Copy vocab.json + preprocessor_config.json to quantized dir."""
    for file_name in ["vocab.json", "preprocessor_config.json"]:
        dest = os.path.join(quant_dir, file_name)
        if os.path.exists(dest):
            continue

        src = os.path.join(onnx_dir, file_name)
        if os.path.exists(src):
            shutil.copy2(src, dest)
            log(lang, f"✅ {file_name} copied from export")
            continue

        url = f"https://huggingface.co/{model_id}/raw/main/{file_name}"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(dest, "wb") as f:
                f.write(r.content)
            log(lang, f"✅ {file_name} downloaded")
        except Exception as e:
            level = "❌" if file_name == "vocab.json" else "⚠️ "
            log(lang, f"{level} Could not get {file_name}: {e}")


def stage_package(lang: str, quant_dir: str, package_path: str) -> bool:
    """
    Pack all files in quant_dir into model.tar.gz (flat archive — no
    subdirectory wrapper), so the archive extracts directly as:
        model_quantized.onnx
        vocab.json
        preprocessor_config.json
    """
    log(lang, f"🗜️  Packaging  (RAM free: {available_ram_mb():.0f}MB)...")
    try:
        files = sorted(
            f for f in os.listdir(quant_dir)
            if os.path.isfile(os.path.join(quant_dir, f))
        )
        if not files:
            log(lang, "❌ No files found in quantized dir — nothing to package")
            return False

        with tarfile.open(package_path, "w:gz") as tar:
            for fname in files:
                fpath = os.path.join(quant_dir, fname)
                tar.add(fpath, arcname=fname)          # arcname = flat name only
                log(lang, f"   + {fname:<35} ({file_size_mb(fpath):.1f} MB)")

        pkg_size = file_size_mb(package_path)
        log(lang, f"✅ Package ready: {package_path}  ({pkg_size:.1f} MB)")
        return True

    except Exception as e:
        log(lang, f"❌ Packaging failed: {e}")
        if os.path.exists(package_path):        # remove partial file on error
            os.remove(package_path)
        return False

# =============================================================================
# MAIN CONVERSION FUNCTION
# =============================================================================

def convert_single(
    lang: str,
    model_id: str,
    skip_existing: bool,
    keep_onnx: bool,
    s3_bucket: Optional[str] = None,
) -> ConversionResult:
    """Full pipeline for one language."""
    t_start = time.time()
    proc = psutil.Process(os.getpid())
    ram_start = proc.memory_info().rss / (1024 * 1024)

    if skip_existing and is_already_converted(lang):
        quant_dir    = os.path.join(BASE_DIR, lang, "quantized")
        package_path = os.path.join(BASE_DIR, lang, "model.tar.gz")
        if not is_packaged(lang):
            log(lang, "⏭️  Already converted — creating missing package...")
            stage_package(lang, quant_dir, package_path)
        else:
            log(lang, "⏭️  Already converted and packaged — skipping conversion")

        uploaded = False
        s3_uri   = None
        if s3_bucket and os.path.exists(package_path):
            uploaded, s3_uri = stage_upload(
                lang, package_path, s3_bucket, skip_if_exists=True
            )

        return ConversionResult(
            lang=lang, model_id=model_id,
            success=True, skipped=True,
            package_size_mb=round(file_size_mb(package_path), 1),
            uploaded=uploaded,
            s3_uri=s3_uri,
        )

    ram_free = available_ram_mb()
    if ram_free < MEM_ESTIMATE_EXPORT_MB + MEM_SAFETY_MARGIN_MB:
        reason = (
            f"Insufficient RAM: {ram_free:.0f}MB free, "
            f"need ~{MEM_ESTIMATE_EXPORT_MB + MEM_SAFETY_MARGIN_MB:.0f}MB. "
            f"Reduce --workers or wait for other workers to finish."
        )
        log(lang, f"⏸️  {reason}")
        return ConversionResult(
            lang=lang, model_id=model_id,
            success=False, error=reason,
            duration_sec=round(time.time() - t_start, 1),
        )

    onnx_dir = os.path.join(BASE_DIR, lang, "onnx")
    quant_dir = os.path.join(BASE_DIR, lang, "quantized")
    quant_path = os.path.join(quant_dir, "model_quantized.onnx")
    os.makedirs(onnx_dir,  exist_ok=True)
    os.makedirs(quant_dir, exist_ok=True)

    def fail(reason: str) -> ConversionResult:
        gc.collect()
        return ConversionResult(
            lang=lang, model_id=model_id, success=False,
            error=reason,
            duration_sec=round(time.time() - t_start, 1),
        )

    if not stage_export(lang, model_id, onnx_dir):
        return fail("ONNX export failed")

    orig_path = find_model_onnx(onnx_dir)
    if not orig_path:
        return fail("model.onnx not found after export")
    onnx_size = file_size_mb(orig_path)
    log(lang, f"   model.onnx  : {onnx_size:.1f} MB")

    if not stage_quantize(lang, orig_path, quant_path):
        return fail("Quantization failed")
    quant_size = file_size_mb(quant_path)
    log(lang, f"   quantized   : {quant_size:.1f} MB "
              f"({100*(1 - quant_size/onnx_size):.0f}% reduction)")

    passed, shape_str, mse = stage_verify(lang, orig_path, quant_path)

    stage_vocab(lang, model_id, onnx_dir, quant_dir)

    # ── Package quantized dir → model.tar.gz ────────────────────────────────
    package_path = os.path.join(BASE_DIR, lang, "model.tar.gz")
    if passed:
        stage_package(lang, quant_dir, package_path)
    pkg_size = file_size_mb(package_path)

    # ── Upload to S3 ─────────────────────────────────────────────────────────
    uploaded = False
    s3_uri   = None
    if passed and s3_bucket and pkg_size > 0:
        uploaded, s3_uri = stage_upload(
            lang, package_path, s3_bucket, skip_if_exists=True
        )

    if not keep_onnx and passed:
        shutil.rmtree(onnx_dir, ignore_errors=True)
        log(lang, f"🗑️  Removed onnx dir (freed {onnx_size:.1f} MB disk)")

    gc.collect()

    duration = round(time.time() - t_start, 1)
    ram_end = proc.memory_info().rss / (1024 * 1024)
    peak_ram = ram_end - ram_start

    log(lang, f"{'✅' if passed else '⚠️ '} Done in {duration}s  "
              f"(RAM delta: {peak_ram:+.0f}MB)")

    return ConversionResult(
        lang=lang,
        model_id=model_id,
        success=passed,
        duration_sec=duration,
        onnx_size_mb=round(onnx_size, 1),
        quantized_size_mb=round(quant_size, 1),
        package_size_mb=round(pkg_size, 1),
        uploaded=uploaded,
        s3_uri=s3_uri,
        output_shape=shape_str,
        mse=round(mse, 6) if mse is not None else None,
        peak_ram_mb=round(peak_ram, 1),
    )


# =============================================================================
# REPORTING
# =============================================================================

def print_summary(results: List[ConversionResult], s3_bucket: Optional[str] = None):
    succeeded = sum(1 for r in results if r.success and not r.skipped)
    skipped = sum(1 for r in results if r.skipped)
    failed = sum(1 for r in results if not r.success)

    print("=" * 105)
    print("📊 CONVERSION SUMMARY")
    print("=" * 105)
    print(f"{'Lang':<6} {'Model':<45} {'Status':<10} "
          f"{'qMB':>6} {'pkgMB':>6} {'S3':>4} {'MSE':>8} {'Time':>7} {'RAM△':>7}")
    print("-" * 105)

    for r in sorted(results, key=lambda x: x.lang):
        status = "SKIPPED" if r.skipped else ("✅ OK" if r.success else "❌ FAIL")
        size   = f"{r.quantized_size_mb:.0f}" if r.quantized_size_mb else "-"
        pkg    = f"{r.package_size_mb:.0f}"   if r.package_size_mb   else "-"
        mse    = f"{r.mse:.4f}" if r.mse is not None else "-"
        secs   = f"{r.duration_sec:.0f}s"     if r.duration_sec      else "-"
        ram    = f"{r.peak_ram_mb:+.0f}"      if r.peak_ram_mb       else "-"
        # S3 column: "ok" uploaded | "--" failed/skipped | "  " not attempted
        if s3_bucket:
            s3col = "ok" if r.uploaded else "--"
        else:
            s3col = ""
        print(f"{r.lang:<6} {r.model_id[:44]:<45} {status:<10} "
              f"{size:>6} {pkg:>6} {s3col:>4} {mse:>8} {secs:>7} {ram:>7}")
        if r.error:
            print(f"       └─ {r.error}")
        if r.s3_uri:
            print(f"       └─ {r.s3_uri}")

    print("-" * 105)
    uploaded_count = sum(1 for r in results if r.uploaded)
    ...                                                     # keep existing counts
    if s3_bucket:
        print(f"   ☁️  Uploaded  : {uploaded_count}")
    print("=" * 105)


def save_report(results: List[ConversionResult], path: str = "conversion_report.json"):
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "system": {
            "total_ram_mb":     round(total_ram_mb(),     0),
            "available_ram_mb": round(available_ram_mb(), 0),
        },
        "summary": {
            "total":     len(results),
            "succeeded": sum(1 for r in results if r.success and not r.skipped),
            "skipped":   sum(1 for r in results if r.skipped),
            "failed":    sum(1 for r in results if not r.success),
        },
        "models": sorted([
            {
                "lang":              r.lang,
                "model_id":          r.model_id,
                "success":           r.success,
                "skipped":           r.skipped,
                "error":             r.error,
                "duration_sec":      r.duration_sec,
                "onnx_size_mb":      r.onnx_size_mb,
                "quantized_size_mb": r.quantized_size_mb,
                "package_size_mb":   r.package_size_mb,
                "uploaded":          r.uploaded,
                "s3_uri":            r.s3_uri,
                "output_shape":      r.output_shape,
                "mse":               r.mse,
                "peak_ram_mb":       r.peak_ram_mb,
            }
            for r in results
        ], key=lambda x: x["lang"]),
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 Report saved → {path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert wav2vec2 alignment models → quantized ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available languages: {', '.join(sorted(ALL_ALIGN_MODELS.keys()))}",
    )
    parser.add_argument(
        "--langs", nargs="+", metavar="LANG",
        help="Languages to convert (default: all)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help=(
            "Parallel workers. Default: auto-calculated from available RAM. "
            "Use 1 for safest operation."
        ),
    )
    parser.add_argument(
        "--keep-onnx", action="store_true", default=DEFAULT_KEEP_ONNX,
        help="Keep original ONNX after quantization",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-convert even if already done",
    )
    parser.add_argument(
        "--s3-bucket", metavar="BUCKET",
        help=(
            "S3 bucket to upload model.tar.gz after packaging. "
            "Destination: s3://BUCKET/opensource_models/align/{lang}/model.tar.gz. "
            "Uses boto3 default credential chain "
            "(env vars / ~/.aws / IAM role)."
        ),
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all languages and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable alignment models:")
        for lang, model_id in sorted(ALL_ALIGN_MODELS.items()):
            status = "✅" if is_already_converted(lang) else "  "
            print(f"  {status} {lang:<6} {model_id}")
        return

    if args.workers is not None:
        n_workers = args.workers
        worker_source = "specified"
    else:
        n_workers = safe_worker_count()
        worker_source = "auto"

    skip_existing = not args.force

    if args.langs:
        unknown = set(args.langs) - set(ALL_ALIGN_MODELS.keys())
        if unknown:
            print(f"❌ Unknown languages: {sorted(unknown)}")
            sys.exit(1)
        to_convert = {lang: ALL_ALIGN_MODELS[lang] for lang in args.langs}
    else:
        to_convert = ALL_ALIGN_MODELS

    ram_total = total_ram_mb()
    ram_free = available_ram_mb()
    estimated_peak = n_workers * MEM_ESTIMATE_PER_WORKER_MB
    estimated_gb = len(to_convert) * 250 / 1024

    print(
        f"\n🖥️  System RAM  : {ram_total:.0f} MB total, {ram_free:.0f} MB free")
    print(
        f"⚠️  Per-worker  : ~{MEM_ESTIMATE_PER_WORKER_MB} MB peak (during export)")
    print(f"📊 Workers     : {n_workers} ({worker_source})")
    print(f"📊 Peak RAM    : ~{estimated_peak} MB with {n_workers} workers")

    if estimated_peak > ram_free - MEM_SAFETY_MARGIN_MB:
        print(f"\n⚠️  WARNING: {n_workers} workers may exceed available RAM!")
        print(f"   Consider --workers 1 to be safe")
        if n_workers > 1 and not args.workers:
            n_workers = 1
            print(f"   Auto-reducing to 1 worker")

    print(
        f"\n🌍 Converting  : {len(to_convert)}/{len(ALL_ALIGN_MODELS)} languages")
    print(f"   Skip exist  : {skip_existing}")
    print(f"   Keep ONNX   : {args.keep_onnx}")
    print(f"   Est. disk   : ~{estimated_gb:.1f} GB (quantized only)")
    print(f"   Languages   : {sorted(to_convert.keys())}\n")

    results: List[ConversionResult] = []

    if n_workers == 1:
        print("🔄 Running serially (safest for memory)\n")
        for lang, model_id in to_convert.items():
            results.append(
                convert_single(lang, model_id, skip_existing, args.keep_onnx, args.s3_bucket)
            )
    else:
        print(f"🚀 Running {n_workers} parallel workers\n")
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_lang = {
                executor.submit(
                    convert_single, lang, model_id,
                    skip_existing, args.keep_onnx, args.s3_bucket,
                ): lang
                for lang, model_id in to_convert.items()
            }
            for future in as_completed(future_to_lang):
                lang = future_to_lang[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    log(lang, f"❌ Unexpected error: {e}")
                    results.append(ConversionResult(
                        lang=lang,
                        model_id=to_convert[lang],
                        success=False,
                        error=str(e),
                    ))

    print_summary(results, args.s3_bucket)
    save_report(results)


if __name__ == "__main__":
    main()
