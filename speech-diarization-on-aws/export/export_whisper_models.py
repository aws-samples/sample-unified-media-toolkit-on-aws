#!/usr/bin/env python3
# export_whisper_models.py
import os
import sys
import json
import shutil
import tarfile
import gc
import psutil
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from botocore.exceptions import ClientError, NoCredentialsError


# =============================================================================
# MODEL REGISTRY
# =============================================================================

WHISPER_MODELS: Dict[str, str] = {
    # key    : HuggingFace model ID
    # -------------------------------------------------------------------------
    # "turbo" : multilingual — covers ~99 languages out of the box.
    #          Use a specialized entry below when a fine-tuned checkpoint
    #          exists for a language the turbo model handles poorly.
    #
    # Key convention:
    #   "turbo"           → the shared multilingual model
    #   ISO 639-1 code   → language-specific fine-tune (e.g. "cy", "ar", "hi")
    #   descriptive slug → when a single code is ambiguous (e.g. "cy-en" for
    #                      a Welsh/English bilingual model)
    # -------------------------------------------------------------------------
    "turbo": "openai/whisper-large-v3-turbo",

    # ── Specialized / fine-tuned models ─────────────────────────────────────
    "cy":   "techiaith/whisper-large-v3-ft-verbatim-cy-en",   # Welsh / English
    # "ar": "tarteel-ai/whisper-large-v2-ar",                 # Arabic
    # "hi": "vasista22/whisper-medium-hi",                    # Hindi
    # "ko": "SungBeom/whisper-small-ko",                      # Korean
}

# Files copied alongside model weights into the ct2 output dir.
# Missing files are skipped with a warning rather than raising FileNotFoundError
# (added_tokens.json is absent on some checkpoints).
COPY_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "added_tokens.json",
]

# Skip heavy non-safetensors weight formats — only safetensors + config needed.
IGNORE_PATTERNS = ["*.msgpack", "*.h5", "*.ot", "*.bin", "*.pth"]


# =============================================================================
# DEFAULTS
# =============================================================================

BASE_DIR = "./assets/models/whisper"
S3_KEY_PREFIX = "opensource_models/whisper"
MEM_ESTIMATE_CONVERT_MB = 4000     # peak RAM during ctranslate2 conversion
MEM_SAFETY_MARGIN_MB = 512


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class ConversionResult:
    key:             str
    model_id:        str
    success:         bool
    skipped:         bool = False
    error:           Optional[str] = None
    duration_sec:    float = 0.0
    ct2_size_mb:     float = 0.0
    package_size_mb: float = 0.0
    uploaded:        bool = False
    s3_uri:          Optional[str] = None
    peak_ram_mb:     float = 0.0


# =============================================================================
# HELPERS
# =============================================================================

def log(key: str, msg: str) -> None:
    print(f"[{key:>6}] {msg}", flush=True)


def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0.0


def dir_size_mb(path: str) -> float:
    """Sum of every file under a directory tree."""
    if not os.path.exists(path):
        return 0.0
    total = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(path)
        for f in files
    )
    return total / (1024 * 1024)


def available_ram_mb() -> float:
    return psutil.virtual_memory().available / (1024 * 1024)


def total_ram_mb() -> float:
    return psutil.virtual_memory().total / (1024 * 1024)


def safe_worker_count() -> int:
    usable = available_ram_mb() - MEM_SAFETY_MARGIN_MB
    return max(1, int(usable // MEM_ESTIMATE_CONVERT_MB))


def is_downloaded(download_dir: str) -> bool:
    """
    True if a complete snapshot is already present.
    Heuristic: config.json + at least one .safetensors shard.
    Avoids re-downloading after a failed conversion step.
    """
    if not os.path.isdir(download_dir):
        return False
    files = os.listdir(download_dir)
    return (
        "config.json" in files
        and any(f.endswith(".safetensors") for f in files)
    )


def is_converted(model_dir: str) -> bool:
    """True if CTranslate2 model.bin exists under model_dir/ct2/."""
    ct2_dir = os.path.join(model_dir, "ct2")
    return (
        os.path.isdir(ct2_dir)
        and os.path.exists(os.path.join(ct2_dir, "model.bin"))
    )


def is_packaged(model_dir: str) -> bool:
    return os.path.exists(os.path.join(model_dir, "model.tar.gz"))


def is_uploaded_s3(key: str, bucket: str) -> bool:
    s3_key = f"{S3_KEY_PREFIX}/{key}/model.tar.gz"
    try:
        boto3.client("s3").head_object(Bucket=bucket, Key=s3_key)
        return True
    except Exception:
        return False


# =============================================================================
# PIPELINE STAGES
# =============================================================================

def stage_download(
    key: str,
    model_id: str,
    download_dir: str,
    hf_token: Optional[str],
) -> bool:
    """Download model snapshot from HuggingFace Hub."""
    if is_downloaded(download_dir):
        log(key, f"⏭️  Raw download already present — skipping  "
                 f"({dir_size_mb(download_dir):.1f} MB)")
        return True

    log(key, f"📥 Downloading {model_id}  "
             f"(RAM free: {available_ram_mb():.0f}MB)...")
    try:
        from huggingface_hub import snapshot_download

        os.makedirs(download_dir, exist_ok=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=download_dir,
            token=hf_token or None,
            ignore_patterns=IGNORE_PATTERNS,
        )
        dl_size = dir_size_mb(download_dir)
        log(key, f"✅ Download complete  "
                 f"({dl_size:.1f} MB, RAM free: {available_ram_mb():.0f}MB)")
        return True

    except Exception as e:
        log(key, f"❌ Download failed: {e}")
        return False


def stage_convert(
    key: str,
    download_dir: str,
    ct2_dir: str,
    quantization: str,
) -> bool:
    """
    Convert HuggingFace checkpoint → CTranslate2.

    Quantization guide for AWS Lambda:
      int8           — best size on x86_64 and arm64          (default)
      int8_float16   — better quality on ARM64 Graviton
                       (fp16 compute for activations, int8 weights)
      float16        — no size saving; useful if GPU needed
    """
    log(key, f"⚙️  Converting → CTranslate2 ({quantization})  "
             f"(RAM free: {available_ram_mb():.0f}MB)...")
    try:
        import ctranslate2

        if os.path.exists(ct2_dir):
            shutil.rmtree(ct2_dir)
        os.makedirs(ct2_dir, exist_ok=True)

        # Only copy files that actually exist in the download.
        copy_files = [
            f for f in COPY_FILES
            if os.path.exists(os.path.join(download_dir, f))
        ]
        missing = set(COPY_FILES) - set(copy_files)
        if missing:
            log(key, f"   ⚠️  Not found (skipping): {sorted(missing)}")
        log(key, f"   Copying: {copy_files}")

        converter = ctranslate2.converters.TransformersConverter(
            download_dir,
            copy_files=copy_files,
        )
        converter.convert(ct2_dir, quantization=quantization, force=True)
        gc.collect()

        ct2_size = dir_size_mb(ct2_dir)
        log(key, f"✅ Conversion complete  "
                 f"({ct2_size:.1f} MB, RAM free: {available_ram_mb():.0f}MB)")
        return True

    except Exception as e:
        gc.collect()
        log(key, f"❌ Conversion failed: {e}")
        return False


def stage_package(key: str, ct2_dir: str, package_path: str) -> bool:
    """
    Pack all files in ct2_dir into a flat model.tar.gz.
    Archive extracts directly as model.bin, vocab.json, etc. — no subfolder.
    """
    log(key, f"🗜️  Packaging  (RAM free: {available_ram_mb():.0f}MB)...")
    try:
        files = sorted(
            f for f in os.listdir(ct2_dir)
            if os.path.isfile(os.path.join(ct2_dir, f))
        )
        if not files:
            log(key, "❌ No files found in ct2 dir — nothing to package")
            return False

        with tarfile.open(package_path, "w:gz") as tar:
            for fname in files:
                fpath = os.path.join(ct2_dir, fname)
                tar.add(fpath, arcname=fname)      # arcname = flat name only
                log(key, f"   + {fname:<45} ({file_size_mb(fpath):.1f} MB)")

        pkg_size = file_size_mb(package_path)
        log(key, f"✅ Package ready: {package_path}  ({pkg_size:.1f} MB)")
        return True

    except Exception as e:
        log(key, f"❌ Packaging failed: {e}")
        if os.path.exists(package_path):
            os.remove(package_path)
        return False


def _make_progress_callback(key: str, total_bytes: int):
    """Log upload progress at 0 %, 25 %, 50 %, 75 %, 100 %."""
    uploaded = [0]
    last_mile = [-1]

    def callback(chunk: int) -> None:
        uploaded[0] += chunk
        pct = int(100 * uploaded[0] / total_bytes)
        milestone = (pct // 25) * 25
        if milestone > last_mile[0]:
            last_mile[0] = milestone
            mb_done = uploaded[0] / (1024 * 1024)
            mb_total = total_bytes / (1024 * 1024)
            log(key, f"   ↑ {mb_done:.1f}/{mb_total:.1f} MB  ({pct}%)")

    return callback


def stage_upload(
    key: str,
    package_path: str,
    bucket: str,
    skip_if_exists: bool = True,
) -> Tuple[bool, Optional[str]]:
    """
    Upload model.tar.gz to:
      s3://{bucket}/opensource_models/whisper/{key}/model.tar.gz

    Uses boto3 default credential chain (env vars / ~/.aws / IAM role).
    Multipart upload is handled automatically by upload_file for large files.
    """
    if not os.path.exists(package_path):
        log(key, f"❌ Package not found: {package_path}")
        return False, None

    s3_key = f"{S3_KEY_PREFIX}/{key}/model.tar.gz"
    s3_uri = f"s3://{bucket}/{s3_key}"
    pkg_mb = file_size_mb(package_path)

    if skip_if_exists and is_uploaded_s3(key, bucket):
        log(key, f"⏭️  Already uploaded → {s3_uri}")
        return True, s3_uri

    log(key, f"☁️  Uploading → {s3_uri}  ({pkg_mb:.1f} MB)...")
    try:
        s3 = boto3.client("s3")
        s3.upload_file(
            package_path,
            bucket,
            s3_key,
            ExtraArgs={"ContentType": "application/gzip"},
            Callback=_make_progress_callback(
                key, os.path.getsize(package_path)),
        )
        log(key, f"✅ Uploaded → {s3_uri}")
        return True, s3_uri

    except NoCredentialsError:
        log(key, "❌ No AWS credentials — set AWS_ACCESS_KEY_ID / "
                 "AWS_SECRET_ACCESS_KEY or attach an IAM role")
        return False, None
    except ClientError as e:
        log(key, f"❌ S3 ClientError: {e}")
        return False, None
    except Exception as e:
        log(key, f"❌ Upload error: {e}")
        return False, None


# =============================================================================
# MAIN CONVERSION FUNCTION
# =============================================================================

def convert_single(
    key: str,
    model_id: str,
    skip_existing: bool,
    keep_raw: bool,
    quantization: str,
    hf_token: Optional[str],
    s3_bucket: Optional[str],
) -> ConversionResult:
    """Full pipeline for one model: download → convert → package → upload."""
    t_start = time.time()
    proc = psutil.Process(os.getpid())
    ram_start = proc.memory_info().rss / (1024 * 1024)

    model_dir = os.path.join(BASE_DIR, key)
    download_dir = os.path.join(model_dir, "raw")
    ct2_dir = os.path.join(model_dir, "ct2")
    package_path = os.path.join(model_dir, "model.tar.gz")

    os.makedirs(model_dir, exist_ok=True)

    def fail(reason: str) -> ConversionResult:
        gc.collect()
        return ConversionResult(
            key=key, model_id=model_id,
            success=False, error=reason,
            duration_sec=round(time.time() - t_start, 1),
        )

    # ── RAM gate ──────────────────────────────────────────────────────────────
    ram_free = available_ram_mb()
    if ram_free < MEM_ESTIMATE_CONVERT_MB + MEM_SAFETY_MARGIN_MB:
        reason = (
            f"Insufficient RAM: {ram_free:.0f}MB free, "
            f"need ~{MEM_ESTIMATE_CONVERT_MB + MEM_SAFETY_MARGIN_MB}MB. "
            f"Reduce --workers or wait for other workers to finish."
        )
        log(key, f"⏸️  {reason}")
        return fail(reason)

    # ── Skip logic ────────────────────────────────────────────────────────────
    if skip_existing and is_converted(model_dir):
        if not is_packaged(model_dir):
            log(key, "⏭️  Already converted — creating missing package...")
            if not stage_package(key, ct2_dir, package_path):
                return fail("Packaging failed")
        else:
            log(key, "⏭️  Already converted and packaged — skipping conversion")

        uploaded, s3_uri = False, None
        if s3_bucket:
            uploaded, s3_uri = stage_upload(
                key, package_path, s3_bucket, skip_if_exists=True
            )

        return ConversionResult(
            key=key, model_id=model_id,
            success=True, skipped=True,
            ct2_size_mb=round(dir_size_mb(ct2_dir), 1),
            package_size_mb=round(file_size_mb(package_path), 1),
            uploaded=uploaded, s3_uri=s3_uri,
        )

    # ── Stage 1: Download ─────────────────────────────────────────────────────
    if not stage_download(key, model_id, download_dir, hf_token):
        return fail("Download failed")

    # ── Stage 2: Convert ──────────────────────────────────────────────────────
    if not stage_convert(key, download_dir, ct2_dir, quantization):
        return fail("CTranslate2 conversion failed")

    ct2_size = dir_size_mb(ct2_dir)
    log(key, f"   ct2 dir     : {ct2_size:.1f} MB")

    # ── Stage 3: Cleanup raw download ─────────────────────────────────────────
    if not keep_raw and os.path.exists(download_dir):
        shutil.rmtree(download_dir, ignore_errors=True)
        log(key, f"🗑️  Removed raw download  "
                 f"(RAM free: {available_ram_mb():.0f}MB)")
    gc.collect()

    # ── Stage 4: Package ──────────────────────────────────────────────────────
    if not stage_package(key, ct2_dir, package_path):
        return fail("Packaging failed")

    pkg_size = file_size_mb(package_path)
    if ct2_size > 0:
        log(key, f"   package     : {pkg_size:.1f} MB  "
                 f"({100 * (1 - pkg_size / ct2_size):.0f}% reduction)")

    # ── Stage 5: Upload ───────────────────────────────────────────────────────
    uploaded, s3_uri = False, None
    if s3_bucket:
        uploaded, s3_uri = stage_upload(
            key, package_path, s3_bucket, skip_if_exists=True
        )

    gc.collect()

    duration = round(time.time() - t_start, 1)
    ram_end = proc.memory_info().rss / (1024 * 1024)
    peak_ram = ram_end - ram_start

    log(key, f"✅ Done in {duration}s  (RAM delta: {peak_ram:+.0f}MB)")

    return ConversionResult(
        key=key, model_id=model_id,
        success=True,
        duration_sec=duration,
        ct2_size_mb=round(ct2_size, 1),
        package_size_mb=round(pkg_size, 1),
        uploaded=uploaded,
        s3_uri=s3_uri,
        peak_ram_mb=round(peak_ram, 1),
    )


# =============================================================================
# REPORTING
# =============================================================================

def print_summary(
    results: List[ConversionResult],
    s3_bucket: Optional[str] = None,
) -> None:
    succeeded = sum(1 for r in results if r.success and not r.skipped)
    skipped = sum(1 for r in results if r.skipped)
    failed = sum(1 for r in results if not r.success)
    uploaded = sum(1 for r in results if r.uploaded)

    print("\n" + "=" * 100)
    print("📊 WHISPER CONVERSION SUMMARY")
    print("=" * 100)
    print(f"{'Key':<8} {'Model':<50} {'Status':<10} "
          f"{'ct2MB':>6} {'pkgMB':>6} {'S3':>4} {'Time':>7} {'RAM△':>7}")
    print("-" * 100)

    for r in sorted(results, key=lambda x: x.key):
        status = "SKIPPED" if r.skipped else (
            "✅ OK" if r.success else "❌ FAIL")
        ct2 = f"{r.ct2_size_mb:.0f}" if r.ct2_size_mb else "-"
        pkg = f"{r.package_size_mb:.0f}" if r.package_size_mb else "-"
        secs = f"{r.duration_sec:.0f}s" if r.duration_sec else "-"
        ram = f"{r.peak_ram_mb:+.0f}" if r.peak_ram_mb else "-"
        s3col = ("ok" if r.uploaded else "--") if s3_bucket else ""
        print(f"{r.key:<8} {r.model_id[:49]:<50} {status:<10} "
              f"{ct2:>6} {pkg:>6} {s3col:>4} {secs:>7} {ram:>7}")
        if r.error:
            print(f"         └─ {r.error}")
        if r.s3_uri:
            print(f"         └─ {r.s3_uri}")

    print("-" * 100)
    print(f"Total: {len(results)}  |  "
          f"✅ Converted: {succeeded}  |  "
          f"⏭️  Skipped: {skipped}  |  "
          f"❌ Failed: {failed}"
          + (f"  |  ☁️  Uploaded: {uploaded}" if s3_bucket else ""))
    print("=" * 100)


def save_report(
    results: List[ConversionResult],
    path: str = "whisper_conversion_report.json",
) -> None:
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
                "key":             r.key,
                "model_id":        r.model_id,
                "success":         r.success,
                "skipped":         r.skipped,
                "error":           r.error,
                "duration_sec":    r.duration_sec,
                "ct2_size_mb":     r.ct2_size_mb,
                "package_size_mb": r.package_size_mb,
                "uploaded":        r.uploaded,
                "s3_uri":          r.s3_uri,
                "peak_ram_mb":     r.peak_ram_mb,
            }
            for r in results
        ], key=lambda x: x["key"]),
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 Report saved → {path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    available_keys = ", ".join(sorted(WHISPER_MODELS.keys()))

    parser = argparse.ArgumentParser(
        description="Convert Whisper models → CTranslate2 int8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Folder layout per model key:\n"
            f"  {BASE_DIR}/{{key}}/raw/          ← HuggingFace download (removed after conversion)\n"
            f"  {BASE_DIR}/{{key}}/ct2/          ← CTranslate2 model files\n"
            f"  {BASE_DIR}/{{key}}/model.tar.gz  ← flat archive ready for Lambda / S3\n"
            "\n"
            "S3 destination (if --s3-bucket provided):\n"
            f"  s3://BUCKET/{S3_KEY_PREFIX}/{{key}}/model.tar.gz\n"
            "\n"
            "Quantization notes:\n"
            "  int8            best size on x86_64 and arm64 (default)\n"
            "  int8_float16    slightly better quality on ARM64 Graviton\n"
            "  float16         no size saving; fastest if GPU Lambda needed\n"
            "\n"
            f"Available model keys: {available_keys}\n"
        ),
    )
    parser.add_argument(
        "--models", nargs="+", metavar="KEY",
        help=(
            f"Model keys to convert (default: all). "
            f"Available: {available_keys}"
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help=(
            "Parallel workers (default: auto from RAM). "
            "Use 1 for safest operation."
        ),
    )
    parser.add_argument(
        "--quantization",
        default="int8",
        choices=["int8", "int8_float16", "int16", "float16", "float32"],
        help="CTranslate2 quantization type (default: int8)",
    )
    parser.add_argument(
        "--s3-bucket", metavar="BUCKET",
        help=(
            "S3 bucket for upload. "
            f"Destination: s3://BUCKET/{S3_KEY_PREFIX}/{{key}}/model.tar.gz"
        ),
    )
    parser.add_argument(
        "--hf-token", metavar="TOKEN",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token (default: $HF_TOKEN env var)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-convert even if already done",
    )
    parser.add_argument(
        "--keep-raw", action="store_true",
        help="Keep raw HuggingFace download after conversion",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available model keys and exit",
    )
    args = parser.parse_args()

    # ── --list ────────────────────────────────────────────────────────────────
    if args.list:
        print("\nAvailable Whisper models:")
        for key, model_id in sorted(WHISPER_MODELS.items()):
            model_dir = os.path.join(BASE_DIR, key)
            conv = "✅" if is_converted(model_dir) else "  "
            pkg = "📦" if is_packaged(model_dir) else "  "
            print(f"  {conv} {pkg} {key:<8} {model_id}")
        print("\nLegend: ✅ converted  📦 packaged")
        return

    # ── Validate args ─────────────────────────────────────────────────────────
    if args.models:
        unknown = set(args.models) - set(WHISPER_MODELS.keys())
        if unknown:
            print(f"❌ Unknown model keys: {sorted(unknown)}")
            print(f"   Available: {sorted(WHISPER_MODELS.keys())}")
            sys.exit(1)
        to_convert = {k: WHISPER_MODELS[k] for k in args.models}
    else:
        to_convert = WHISPER_MODELS

    if args.workers is not None:
        n_workers = args.workers
        worker_source = "specified"
    else:
        n_workers = safe_worker_count()
        worker_source = "auto"

    skip_existing = not args.force
    ram_total = total_ram_mb()
    ram_free = available_ram_mb()

    print(
        f"\n🖥️  System RAM    : {ram_total:.0f} MB total, {ram_free:.0f} MB free")
    print(f"⚙️  Quantization  : {args.quantization}")
    print(f"📊 Workers       : {n_workers} ({worker_source})")
    print(f"   Skip existing : {skip_existing}")
    print(f"   Keep raw      : {args.keep_raw}")
    if args.s3_bucket:
        print(f"☁️  S3 bucket     : {args.s3_bucket}")
    print(
        f"\n🎤 Converting    : {len(to_convert)}/{len(WHISPER_MODELS)} models")
    print(f"   Keys          : {sorted(to_convert.keys())}\n")

    results: List[ConversionResult] = []

    if n_workers == 1:
        print("🔄 Running serially (safest for memory)\n")
        for key, model_id in to_convert.items():
            results.append(convert_single(
                key, model_id,
                skip_existing, args.keep_raw,
                args.quantization, args.hf_token, args.s3_bucket,
            ))
    else:
        print(f"🚀 Running {n_workers} parallel workers\n")
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_key = {
                executor.submit(
                    convert_single,
                    key, model_id,
                    skip_existing, args.keep_raw,
                    args.quantization, args.hf_token, args.s3_bucket,
                ): key
                for key, model_id in to_convert.items()
            }
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    log(key, f"❌ Unexpected error: {e}")
                    results.append(ConversionResult(
                        key=key,
                        model_id=to_convert[key],
                        success=False,
                        error=str(e),
                    ))

    print_summary(results, args.s3_bucket)
    save_report(results)
    sys.exit(0 if all(r.success for r in results) else 1)


if __name__ == "__main__":
    main()
