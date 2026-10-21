# run_dinov2_model_handler.py
"""
Step Function State: Extract frame features (DINOv2)
-----------------------------------------------------
Map state — one Lambda per embedding window.  Downloads the proxy
segment, extracts frames, runs DINOv2 ViT-S/14-reg for 384-dim CLS
embeddings, detects black frames, computes luma histogram entropy,
then runs shot boundary detection.

No scene clustering — shots are the terminal output of this state.
Scene clustering is deferred to a downstream pipeline that has access
to VLM/SLM classification for structural shot identification.

Writes per-window .npy (embeddings) + .json (metadata, local shots,
entropy) + _frames.tar.gz (model-input frames at proxy native resolution)
to S3.  The downstream merge state assembles global shots and
embeddings.npz, and repackages selected frames into per-shot archives.

Frame archive
─────────────
Every frame that receives an embedding (coarse + zoom-in, including
black frames) is saved at proxy native resolution (native AR, e.g.
512×288 for 16:9) as JPEG and
bundled into a per-window tar.gz archive.

Frame number accuracy
─────────────────────
Source frame numbers from pts_key JSON (showinfo integer PTS arithmetic).
No fps arithmetic fallback.  pts_key missing → hard failure.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import tarfile
import time
import traceback
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from PIL import Image

from lib.custom_exceptions import LambdaResourceTimeoutException
from lib.input_schema import (
    DEFAULT_COARSE_FPS,
    DEFAULT_COSDIST_THRESHOLD,
    DEFAULT_BLACK_COVERAGE_PCT,
    DEFAULT_BLACK_PIXEL_THRESHOLD,
    DEFAULT_BLACK_ENTROPY_MAX,
    DEFAULT_BLACK_MAX_BRIGHT_PIXEL,
    DEFAULT_MINI_BATCH_SECS,
    DEFAULT_EMBED_GEOMETRY,
    EXTRACT_BATCH_TIME_FACTOR,
    EXTRACT_CHECKPOINT_BUFFER_MS,
)
from lib.misc_utils import (
  round_half_up,
  frame_to_smpte,
)
from lib.s3_utils import (
  get_json,
  put_binary,
  put_json,
  download_file,
  upload_file,
  delete_object,
)
from lib.ort_init import ensure_cpuinfo_files, get_num_cpus
from lib.shot_pipeline import build_shots

ensure_cpuinfo_files()

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.propagate = False
_handler = logging.StreamHandler()
_handler.setLevel(logging.DEBUG)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
))
if not log.handlers:
    log.addHandler(_handler)

# ── Constants ─────────────────────────────────────────────────────────────────
_MODEL_PATH = "/opt/models/dinov2_vits14_reg_cls.onnx"
_DINO_INPUT_SIZE = 224
_DINO_EMBED_DIM = 384

_BLACK_SENTINEL: np.ndarray = np.full(
    _DINO_EMBED_DIM, 1.0 / math.sqrt(_DINO_EMBED_DIM), dtype=np.float32,
)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_cached_session: Optional[ort.InferenceSession] = None


# =============================================================================
# Checkpoint / resume (9.13)
# =============================================================================
# A window is processed in ~120 s mini-batches, and one window on very
# high-cut-rate content (film_a's action window: zoom-in densification
# samples thousands of extra frames) deterministically exceeds Lambda's HARD
# 900 s ceiling. The state therefore checkpoints BETWEEN batches: everything
# the batch loop has accumulated — embeddings, per-frame metadata, the
# persisted model-input JPEGs (needed for the final _frames.tar.gz), and the
# cross-batch seam-holdover frames — is packed into one tar on S3, and
# LambdaResourceTimeoutException is raised for the ASL to retry. The fresh
# invocation restores the tar and resumes at the recorded batch_pos, so each
# retry spends its full budget on NEW batches. The pack/unpack pair is pure
# local-filesystem work (unit-tested); only the handler touches S3.

def _should_checkpoint(remaining_ms: float, max_batch_ms: float) -> bool:
    """Checkpoint when the remaining budget cannot fit another batch (sized
    by the SLOWEST batch this invocation, with headroom) plus the checkpoint
    upload itself. On the first batch of an invocation max_batch_ms is 0 and
    only the upload buffer is reserved — a fresh Lambda always proceeds."""
    return remaining_ms < (EXTRACT_BATCH_TIME_FACTOR * max_batch_ms
                           + EXTRACT_CHECKPOINT_BUFFER_MS)


def _checkpoint_key(dest_prefix: str, window_name: str) -> str:
    return f"{dest_prefix}/embeddings/{window_name}_checkpoint.tar"


def _pack_checkpoint(tar_path, state, embeddings, persist_dir, holdover_dir):
    """One uncompressed tar (the JPEGs are already compressed):
    state.json + embeddings.npy + persist/ + holdover/. `state` must be
    JSON-serialisable; holdover frame paths are stored as basenames and
    re-rooted on restore."""
    emb_path = tar_path + ".emb.npy"
    np.save(emb_path, embeddings if embeddings is not None
            else np.zeros((0, _DINO_EMBED_DIM), dtype=np.float32))
    state_path = tar_path + ".state.json"
    with open(state_path, "w") as f:
        json.dump(state, f)
    with tarfile.open(tar_path, "w") as tar:
        tar.add(state_path, arcname="state.json")
        tar.add(emb_path, arcname="embeddings.npy")
        for name, d in (("persist", persist_dir), ("holdover", holdover_dir)):
            if d and os.path.isdir(d):
                for fn in sorted(os.listdir(d)):
                    tar.add(os.path.join(d, fn), arcname=f"{name}/{fn}")
    os.remove(emb_path)
    os.remove(state_path)


def _unpack_checkpoint(tar_path, persist_dir, holdover_dir):
    """Inverse of _pack_checkpoint: restores the two frame directories and
    returns (state, embeddings)."""
    os.makedirs(persist_dir, exist_ok=True)
    os.makedirs(holdover_dir, exist_ok=True)
    state = None
    embeddings = None
    with tarfile.open(tar_path, "r") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            fobj = tar.extractfile(m)
            if m.name == "state.json":
                state = json.load(fobj)
            elif m.name == "embeddings.npy":
                import io as _io
                embeddings = np.load(_io.BytesIO(fobj.read()))
            elif m.name.startswith("persist/"):
                with open(os.path.join(persist_dir,
                                       os.path.basename(m.name)), "wb") as f:
                    f.write(fobj.read())
            elif m.name.startswith("holdover/"):
                with open(os.path.join(holdover_dir,
                                       os.path.basename(m.name)), "wb") as f:
                    f.write(fobj.read())
    if state is None:
        raise ValueError(f"checkpoint {tar_path} has no state.json")
    return state, embeddings


# =============================================================================
# PTS data
# =============================================================================

def _load_pts_data(proxy_bucket: str, pts_key: str) -> dict:
    log.info("Loading PTS data: s3://%s/%s", proxy_bucket, pts_key)
    try:
        data = get_json(proxy_bucket, pts_key)
    except Exception as exc:
        raise ValueError(
            f"Failed to load pts_key s3://{proxy_bucket}/{pts_key}: {exc}"
        ) from exc
    if "source_frame_numbers" not in data:
        raise ValueError(
            f"pts_key missing 'source_frame_numbers'")
    if not data["source_frame_numbers"]:
        raise ValueError(
            f"pts_key 'source_frame_numbers' is empty — segment produced no frames")
    log.info("PTS: %d frames", len(data["source_frame_numbers"]))
    return data


def _get_source_frame(pts_data: dict, proxy_offset: int) -> int:
    frames = pts_data["source_frame_numbers"]
    if proxy_offset < 0 or proxy_offset >= len(frames):
        raise IndexError(
            f"proxy_offset={proxy_offset} out of range [0, {len(frames)-1}]")
    return frames[proxy_offset]


# =============================================================================
# ORT session
# =============================================================================

def _get_session(memory_mb: Optional[int]) -> ort.InferenceSession:
    global _cached_session
    if _cached_session is not None:
        return _cached_session
    num_threads = get_num_cpus(memory_mb)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = num_threads
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _cached_session = ort.InferenceSession(
        _MODEL_PATH, opts, providers=["CPUExecutionProvider"])
    log.info("ORT loaded: %s  threads=%d", _MODEL_PATH, num_threads)
    return _cached_session


# =============================================================================
# Image preprocessing / frame analysis
# =============================================================================

def _preprocess_frame(
    frame,
    geometry: str = "letterbox",
) -> np.ndarray:
    """
    Convert a native-AR proxy frame to a 224×224 model input.

    frame: a JPEG file path, or a pre-decoded uint8 HWC RGB array
    (avoids a second JPEG decode when the caller already decoded the
    frame for luma analysis).

    geometry:
      "letterbox" — fit at native AR, pad to square with black.
                    Bit-compatible with the embedding distribution of the
                    previous square-padded proxies; the default.
      "squeeze"   — anamorphic resize to 224×224, no padding.
      "crop"      — scale shortest side to 224, center-crop 224×224
                    (DINOv2 native eval protocol; discards periphery).

    Changing geometry changes the embedding distribution — every
    distance threshold in the pipeline was calibrated on "letterbox".
    """
    if isinstance(frame, np.ndarray):
        img = Image.fromarray(frame)
    else:
        img = Image.open(frame).convert("RGB")
    S = _DINO_INPUT_SIZE

    if img.size != (S, S):
        w, h = img.size
        if geometry == "squeeze":
            img = img.resize((S, S), Image.LANCZOS)
        elif geometry == "crop":
            scale = S / min(w, h)
            nw, nh = round(w * scale), round(h * scale)
            img = img.resize((nw, nh), Image.LANCZOS)
            left = (nw - S) // 2
            top = (nh - S) // 2
            img = img.crop((left, top, left + S, top + S))
        else:  # letterbox
            scale = S / max(w, h)
            nw, nh = round(w * scale), round(h * scale)
            img = img.resize((nw, nh), Image.LANCZOS)
            canvas = Image.new("RGB", (S, S), (0, 0, 0))
            canvas.paste(img, ((S - nw) // 2, (S - nh) // 2))
            img = canvas

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[np.newaxis])


def _analyze_frame(
    frame,
    blk_pix_thr: float,
    blk_cov_pct: float,
    blk_ent_max: float,
    blk_bright_max: float,
) -> tuple[bool, float]:
    """
    Compute luma histogram entropy and black frame detection.
    Returns (is_black, entropy).  Entropy always computed.

    frame: a JPEG file path, or a pre-decoded uint8 HWC RGB array.

    Black detection requires all three conditions:
      1. Coverage: >= blk_cov_pct% of pixels have luma < blk_pix_thr*255
      2. Bright pixel: max luma < blk_bright_max (rejects title cards on black)
      3. Entropy: < blk_ent_max

    1 and 2 are the decision; 3 is a backstop. Conditions 1 and 2 confine luma
    to codes 0-30, and a frame confined to that band carries no picture however
    its luma is distributed inside it, so entropy has no separating power left
    to add. It is kept because a caller may want to veto a pathological frame,
    but note that black off a real tape measures 0.4-2.8 bits of entropy, not
    ~0, so the 0.20 default (right for clean digital sources) silently rejects
    genuine black on redigitised tape; such callers pass black_entropy_max=3.5
    at request time. See the note on DEFAULT_BLACK_ENTROPY_MAX in
    lib/input_schema.py for the measurement.
    """
    if isinstance(frame, np.ndarray):
        img = frame.astype(np.float32)
    else:
        img = np.array(Image.open(frame).convert("RGB"), dtype=np.float32)
    luma = 0.2126 * img[:, :, 0] + 0.7152 * \
        img[:, :, 1] + 0.0722 * img[:, :, 2]
    total = luma.size

    # Entropy — always computed
    u8 = np.clip(luma, 0, 255).astype(np.uint8).ravel()
    counts = np.bincount(u8, minlength=256).astype(np.float64)
    probs = counts[counts > 0] / total
    entropy = round(float(-np.sum(probs * np.log2(probs))), 4)

    # Black detection — all three criteria must pass
    tol = blk_pix_thr * 255.0
    if int(np.sum(luma < tol)) < (blk_cov_pct / 100.0) * total:
        return False, entropy
    if float(np.max(luma)) >= blk_bright_max:
        return False, entropy
    is_black = entropy < blk_ent_max
    return is_black, entropy


# =============================================================================
# Frame persistence for debug archives
# =============================================================================

def _persist_frame(src_path: str, dst_path: str) -> None:
    """
    Persist one model-input frame to dst_path at native proxy resolution.

    The source is already a JPEG at the target resolution (ffmpeg -q:v 2
    output — the exact file the model input was derived from), so a byte
    copy is lossless and skips a redundant decode + re-encode per frame.

    StateStoreDebugShotImages is responsible for downscaling to 112x112
    when composing debug grid images.
    """
    shutil.copyfile(src_path, dst_path)


def _create_frames_archive(
    persist_dir: str,
    tmp_base:    str,
    window_name: str,
) -> str | None:
    frame_files = sorted(Path(persist_dir).glob("frame_*.jpg"))
    if not frame_files:
        return None
    tar_path = os.path.join(tmp_base, f"{window_name}_frames.tar.gz")
    with tarfile.open(tar_path, "w:gz", compresslevel=1) as tar:
        for jpg in frame_files:
            tar.add(str(jpg), arcname=jpg.name)
    size_mb = os.path.getsize(tar_path) / (1024 ** 2)
    log.info("Frames archive: %s  %.1f MB  %d frames",
             window_name, size_mb, len(frame_files))
    return tar_path


# =============================================================================
# DINOv2 inference
# =============================================================================

def _embed_frames(
    session: ort.InferenceSession,
    frames: list,
    batch_size: int = 16,
    geometry: str = "letterbox",
) -> np.ndarray:
    """frames items may be JPEG paths or pre-decoded uint8 HWC arrays."""
    inp = session.get_inputs()[0].name
    out = session.get_outputs()[0].name
    parts = []
    for i in range(0, len(frames), batch_size):
        batch = np.concatenate(
            [_preprocess_frame(f, geometry)
             for f in frames[i:i + batch_size]])
        parts.append(session.run([out], {inp: batch})[0])
    return np.concatenate(parts).astype(np.float32)


def _cosine_distances(embeddings: np.ndarray) -> np.ndarray:
    norms = np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
    normed = embeddings / norms
    dots = (normed[:-1] * normed[1:]).sum(axis=1)
    return (1.0 - dots).astype(np.float32)


# =============================================================================
# Frame extraction / proxy download
# =============================================================================

def _extract_frames(local_file, start_sec, duration, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(round(start_sec, 6)),
        "-accurate_seek",
        "-i", local_file,
        "-t", str(round(duration, 6)),
        "-fps_mode", "passthrough",
        "-q:v", "2",
        os.path.join(out_dir, "frame_%06d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # nosec B603  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (rc={result.returncode}): {result.stderr[-500:]}")
    return len(sorted(Path(out_dir).glob("frame_*.jpg")))


# =============================================================================
# Main processing loop
# =============================================================================

def _process_window(session, local_file, pts_data, item, cfg, tmp_base,
                    time_left_ms=None, proxy_bucket=None, ckpt_key=None):
    """
    Process one window in mini-batches.

    time_left_ms   optional callable returning the Lambda's remaining budget
                   in milliseconds (context.get_remaining_time_in_millis).
                   When given together with proxy_bucket/ckpt_key, the batch
                   loop checkpoints to S3 and raises
                   LambdaResourceTimeoutException instead of running into the
                   hard 900 s kill; the next invocation resumes from the
                   checkpoint. None (unit tests, offline replays) disables
                   checkpointing entirely.

    Returns (embeddings, frame_index, frame_types, black_frames,
             is_coarse, entropy, persist_dir).
    """
    fps_str = item["frame_rate"]
    fps = float(item["frame_rate_fps"])
    fps_frac = Fraction(fps_str.replace(":", "/"))
    window_start = float(item["window_start"])
    window_end = float(item["window_end"])
    tc_origin = item.get("timecode_first_frame")
    drop_frame = bool(item.get("drop_frame", False))
    coarse_fps = int(cfg.get("coarse_fps", DEFAULT_COARSE_FPS))
    cosdist_thr = float(cfg.get(
        "cosdist_threshold", DEFAULT_COSDIST_THRESHOLD))
    blk_pix = float(cfg.get(
        "black_pixel_threshold", DEFAULT_BLACK_PIXEL_THRESHOLD))
    blk_cov = float(cfg.get(
        "black_coverage_pct", DEFAULT_BLACK_COVERAGE_PCT))
    blk_ent = float(cfg.get(
        "black_entropy_max", DEFAULT_BLACK_ENTROPY_MAX))
    blk_bright = float(cfg.get(
        "black_max_bright_pixel", DEFAULT_BLACK_MAX_BRIGHT_PIXEL))
    mb_secs = int(cfg.get(
        "mini_batch_secs", DEFAULT_MINI_BATCH_SECS))
    embed_geometry = str(cfg.get(
        "embed_geometry", DEFAULT_EMBED_GEOMETRY))

    coarse_stride = max(1, round_half_up(fps / coarse_fps))

    all_embeddings:   list[np.ndarray] = []
    all_frame_index:  list[int] = []
    all_frame_types:  list[str] = []
    all_is_coarse:    list[bool] = []
    all_entropy:      list[float] = []
    all_black_frames: list[dict] = []

    persist_dir = os.path.join(tmp_base, "persist_frames")
    os.makedirs(persist_dir, exist_ok=True)

    # Cross-batch seam state: tail frames of the previous batch (frames
    # after its last coarse index) held over so the seam pair can be
    # zoomed if it crosses cosdist_thr.  Without this, the pair
    # (last coarse frame of batch N, first coarse frame of batch N+1)
    # is never evaluated and a cut inside that span is invisible to
    # boundary detection.
    holdover_dir = os.path.join(tmp_base, "seam_holdover")
    os.makedirs(holdover_dir, exist_ok=True)
    prev_tail: Optional[dict] = None

    pts_count = len(pts_data["source_frame_numbers"])
    batch_n = 0
    batch_pos = window_start

    # ── Resume from a checkpoint left by a timed-out predecessor ──────
    checkpointing = (time_left_ms is not None and proxy_bucket
                     and ckpt_key)
    if checkpointing:
        ckpt_local = os.path.join(tmp_base, "checkpoint.tar")
        try:
            download_file(proxy_bucket, ckpt_key, ckpt_local)
        except Exception:
            ckpt_local = None            # no checkpoint — fresh window
        if ckpt_local and os.path.exists(ckpt_local):
            state, emb = _unpack_checkpoint(
                ckpt_local, persist_dir, holdover_dir)
            os.remove(ckpt_local)
            if len(emb):
                all_embeddings.append(emb.astype(np.float32))
            all_frame_index[:] = state["frame_index"]
            all_frame_types[:] = state["frame_types"]
            all_is_coarse[:] = state["is_coarse"]
            all_entropy[:] = state["entropy"]
            all_black_frames[:] = state["black_frames"]
            batch_pos = float(state["batch_pos"])
            batch_n = int(state["batch_n"])
            if state.get("prev_tail") is not None:
                pt = state["prev_tail"]
                prev_tail = {
                    "frames": [{**f, "path": os.path.join(
                        holdover_dir, os.path.basename(f["path"]))}
                        for f in pt["frames"]],
                    "last_coarse_emb": np.array(pt["last_coarse_emb"],
                                                dtype=np.float32),
                }
            log.info(
                "RESUMED %s from checkpoint: batch_pos=%.1fs batch_n=%d "
                "frames_so_far=%d persisted=%d",
                item["window_name"], batch_pos, batch_n,
                len(all_frame_index),
                len(os.listdir(persist_dir)))

    max_batch_ms = 0.0

    while batch_pos < window_end:
        # ── Checkpoint gate: can another batch + a checkpoint fit? ────
        remaining_ms = time_left_ms() if checkpointing else None
        if checkpointing and _should_checkpoint(remaining_ms,
                                                max_batch_ms):
            state = {
                "batch_pos": batch_pos,
                "batch_n": batch_n,
                "frame_index": all_frame_index,
                "frame_types": all_frame_types,
                "is_coarse": all_is_coarse,
                "entropy": all_entropy,
                "black_frames": all_black_frames,
                "prev_tail": None if prev_tail is None else {
                    "frames": prev_tail["frames"],
                    "last_coarse_emb":
                        prev_tail["last_coarse_emb"].tolist(),
                },
            }
            emb = (np.concatenate(all_embeddings) if all_embeddings
                   else np.zeros((0, _DINO_EMBED_DIM), dtype=np.float32))
            tar_path = os.path.join(tmp_base, "checkpoint_out.tar")
            _pack_checkpoint(tar_path, state, emb,
                             persist_dir, holdover_dir)
            upload_file(proxy_bucket, ckpt_key, tar_path,
                        {"ContentType": "application/x-tar"})
            raise LambdaResourceTimeoutException(
                f"{item['window_name']} checkpointed at batch_pos="
                f"{batch_pos:.1f}s ({len(all_frame_index)} frames so far, "
                f"remaining={remaining_ms:.0f}ms, "
                f"slowest_batch={max_batch_ms:.0f}ms)")

        t_batch = time.monotonic()
        batch_end = min(batch_pos + mb_secs, window_end)
        batch_dur = batch_end - batch_pos
        batch_dir = os.path.join(tmp_base, f"batch_{batch_n:04d}")
        batch_n += 1

        try:
            n_ext = _extract_frames(
                local_file, batch_pos, batch_dur, batch_dir)
            if n_ext == 0:
                batch_pos = batch_end
                continue

            frame_paths = sorted(Path(batch_dir).glob("frame_*.jpg"))
            # Single rounding from segment start avoids split-rounding drift at
            # non-integer fps (e.g. 29.97): round(360*fps)+round(120*fps) ≠ round(480*fps).
            proxy_offset = int(math.floor(Fraction(batch_pos) * fps_frac + Fraction(1, 2)))
            available = pts_count - proxy_offset
            if available <= 0:
                batch_pos = batch_end
                continue
            if len(frame_paths) > available:
                frame_paths = frame_paths[:available]

            n_frames = len(frame_paths)
            batch_fnums = [
                _get_source_frame(pts_data, proxy_offset + i)
                for i in range(n_frames)
            ]

            # ── Frame analysis: black detection + entropy ─────────────
            # Decode each frame once; retain the decoded array for
            # coarse-grid frames (guaranteed to be embedded next) so the
            # embed step skips a second JPEG decode.  Only coarse frames
            # are cached — caching all would hold ~1.6 GB for a 120s
            # batch; coarse-only is bounded at ~1/coarse_stride of that.
            coarse_set_pre = set(range(0, n_frames, coarse_stride))
            decoded_coarse: dict[int, np.ndarray] = {}
            is_black = []
            frame_entropy = []
            for i, p in enumerate(frame_paths):
                arr = np.array(Image.open(str(p)).convert("RGB"))
                blk, ent = _analyze_frame(
                    arr, blk_pix, blk_cov, blk_ent, blk_bright)
                is_black.append(blk)
                frame_entropy.append(ent)
                if i in coarse_set_pre and not blk:
                    decoded_coarse[i] = arr

            for i, (fn, blk) in enumerate(zip(batch_fnums, is_black)):
                if blk:
                    all_black_frames.append({
                        "frame": fn,
                        "time":  round(fn / fps, 6),
                        "timecode": frame_to_smpte(
                            fn, fps_str, drop_frame, tc_origin),
                    })

            coarse_indices = list(range(0, n_frames, coarse_stride))
            non_blk_coarse = [i for i in coarse_indices if not is_black[i]]

            if non_blk_coarse:
                nbc_emb = _embed_frames(
                    session,
                    [decoded_coarse.get(i, str(frame_paths[i]))
                     for i in non_blk_coarse],
                    geometry=embed_geometry)
            else:
                nbc_emb = np.zeros((0, _DINO_EMBED_DIM), dtype=np.float32)
            decoded_coarse.clear()

            coarse_emb = np.tile(
                _BLACK_SENTINEL, (len(coarse_indices), 1)).astype(np.float32)
            nb_cur = 0
            for ci, idx in enumerate(coarse_indices):
                if not is_black[idx]:
                    coarse_emb[ci] = nbc_emb[nb_cur]
                    nb_cur += 1

            # ── Cross-batch seam zoom ─────────────────────────────────
            # The pair (last coarse frame of previous batch, first coarse
            # frame of this batch) is never seen by the in-batch zoom.
            # Evaluate it here; if it crosses cosdist_thr, embed the
            # held-over tail frames so boundary detection has gap=1
            # coverage across the seam.
            if prev_tail is not None:
                tail = prev_tail
                prev_tail = None
                # Drop tail frames that duplicate this batch's range —
                # ffmpeg -t rounding can hand the previous batch one
                # extra boundary frame that this batch also extracted.
                kept = []
                for f in tail["frames"]:
                    if f["fnum"] < batch_fnums[0]:
                        kept.append(f)
                    else:
                        try:
                            os.remove(f["path"])
                        except OSError:
                            pass
                tail["frames"] = kept
                if tail["frames"]:
                    a = tail["last_coarse_emb"]
                    b = coarse_emb[0]
                    na = a / max(float(np.linalg.norm(a)), 1e-8)
                    nb = b / max(float(np.linalg.norm(b)), 1e-8)
                    d_seam = float(1.0 - na @ nb)
                    if d_seam > cosdist_thr:
                        t_nb = [f for f in tail["frames"] if not f["is_black"]]
                        emb_map: dict[int, np.ndarray] = {}
                        if t_nb:
                            t_emb = _embed_frames(
                                session, [f["path"] for f in t_nb],
                                geometry=embed_geometry)
                            for zi, f in enumerate(t_nb):
                                emb_map[f["fnum"]] = t_emb[zi]
                        for f in tail["frames"]:
                            if f["is_black"]:
                                emb_map[f["fnum"]] = _BLACK_SENTINEL.copy()

                        tail_sorted = sorted(
                            tail["frames"], key=lambda f: f["fnum"])
                        all_embeddings.append(np.stack(
                            [emb_map[f["fnum"]] for f in tail_sorted]))
                        all_frame_index.extend(
                            f["fnum"] for f in tail_sorted)
                        all_frame_types.extend(
                            "black" if f["is_black"] else "normal"
                            for f in tail_sorted)
                        all_is_coarse.extend(False for _ in tail_sorted)
                        all_entropy.extend(
                            f["entropy"] for f in tail_sorted)
                        for f in tail_sorted:
                            dst = os.path.join(
                                persist_dir,
                                f"frame_{f['fnum']:010d}.jpg")
                            _persist_frame(f["path"], dst)
                        log.info(
                            "Seam zoom: batch seam dist=%.4f > thr=%.4f "
                            "— embedded %d tail frame(s) %d-%d",
                            d_seam, cosdist_thr, len(tail_sorted),
                            tail_sorted[0]["fnum"], tail_sorted[-1]["fnum"])
                for f in tail["frames"]:
                    try:
                        os.remove(f["path"])
                    except OSError:
                        pass

            # ── Zoom ──────────────────────────────────────────────────
            zoom_set: set[int] = set()
            if len(coarse_indices) >= 2:
                cd = _cosine_distances(coarse_emb)
                for k, d in enumerate(cd):
                    if float(d) > cosdist_thr:
                        for j in range(coarse_indices[k],
                                       coarse_indices[k + 1] + 1):
                            zoom_set.add(j)

            for i, blk in enumerate(is_black):
                if not blk:
                    continue
                k = i // coarse_stride
                if i == k * coarse_stride:
                    if k > 0:
                        for j in range((k - 1) * coarse_stride,
                                       k * coarse_stride + 1):
                            zoom_set.add(j)
                    for j in range(k * coarse_stride,
                                   min(n_frames - 1, (k + 1) * coarse_stride) + 1):
                        zoom_set.add(j)
                else:
                    lo = k * coarse_stride
                    hi = min(n_frames - 1, (k + 1) * coarse_stride)
                    for j in range(lo, hi + 1):
                        zoom_set.add(j)

            zoom_only = sorted(zoom_set - set(coarse_indices))
            coarse_set_local = set(coarse_indices)

            if zoom_only:
                z_nb = [i for i in zoom_only if not is_black[i]]
                z_bl = [i for i in zoom_only if is_black[i]]
                if z_nb:
                    z_emb = _embed_frames(
                        session, [str(frame_paths[i]) for i in z_nb],
                        geometry=embed_geometry)
                else:
                    z_emb = np.zeros((0, _DINO_EMBED_DIM), dtype=np.float32)

                z_map: dict[int, np.ndarray] = {}
                for zi, idx in enumerate(z_nb):
                    z_map[idx] = z_emb[zi]
                for idx in z_bl:
                    z_map[idx] = _BLACK_SENTINEL.copy()

                c_map = {coarse_indices[ci]: coarse_emb[ci]
                         for ci in range(len(coarse_indices))}

                all_idx = sorted(set(coarse_indices) | zoom_set)
                comb_emb = np.stack([
                    c_map[i] if i in c_map else z_map[i]
                    for i in all_idx
                ])
                comb_fn = [batch_fnums[i] for i in all_idx]
                comb_ft = ["black" if is_black[i] else "normal"
                           for i in all_idx]
                comb_ic = [i in coarse_set_local for i in all_idx]
                comb_ent = [frame_entropy[i] for i in all_idx]
                embedded_indices = all_idx
            else:
                comb_emb = coarse_emb
                comb_fn = [batch_fnums[i] for i in coarse_indices]
                comb_ft = ["black" if is_black[i] else "normal"
                           for i in coarse_indices]
                comb_ic = [True] * len(coarse_indices)
                comb_ent = [frame_entropy[i] for i in coarse_indices]
                embedded_indices = list(coarse_indices)

            all_embeddings.append(comb_emb)
            all_frame_index.extend(comb_fn)
            all_frame_types.extend(comb_ft)
            all_is_coarse.extend(comb_ic)
            all_entropy.extend(comb_ent)

            # ── Persist model-input frames at proxy native resolution ──
            for local_idx in embedded_indices:
                source_fn = batch_fnums[local_idx]
                dst = os.path.join(
                    persist_dir, f"frame_{source_fn:010d}.jpg")
                _persist_frame(str(frame_paths[local_idx]), dst)

            # ── Hold over tail frames for cross-batch seam zoom ────────
            # Frames after the last coarse index have no embedding yet
            # (in-batch zoom only covers spans between coarse pairs that
            # crossed the threshold).  Copy them out of batch_dir before
            # it is deleted so the next iteration can embed them if the
            # seam pair crosses cosdist_thr.
            if coarse_indices and batch_end < window_end:
                last_ci = coarse_indices[-1]
                already = set(embedded_indices)
                tail_frames = []
                for i in range(last_ci + 1, n_frames):
                    if i in already:
                        continue
                    hold_path = os.path.join(
                        holdover_dir, f"hold_{batch_fnums[i]:010d}.jpg")
                    shutil.copyfile(str(frame_paths[i]), hold_path)
                    tail_frames.append({
                        "fnum":     batch_fnums[i],
                        "path":     hold_path,
                        "is_black": is_black[i],
                        "entropy":  frame_entropy[i],
                    })
                prev_tail = {
                    "frames":          tail_frames,
                    "last_coarse_emb": coarse_emb[-1].copy(),
                }

        finally:
            shutil.rmtree(batch_dir, ignore_errors=True)

        batch_pos = batch_end
        # The slowest batch THIS invocation sizes the checkpoint gate's
        # estimate of "one more batch". max() only grows, so a cheap batch
        # can never talk the gate into a fatal optimism.
        max_batch_ms = max(max_batch_ms,
                           (time.monotonic() - t_batch) * 1000.0)

    n_persisted = len(list(Path(persist_dir).glob("frame_*.jpg")))
    log.info("Persisted %d model-input frames to %s", n_persisted, persist_dir)

    final_emb = (
        np.concatenate(all_embeddings) if all_embeddings
        else np.zeros((0, _DINO_EMBED_DIM), dtype=np.float32)
    )

    # Seam-zoom tail insertion can land rows out of frame order when the
    # previous batch's black-frame zoom embedded frames beyond its last
    # coarse index.  Downstream (build_shots, merge searchsorted) requires
    # frame_index strictly ascending — restore order if violated.
    fn_arr = np.array(all_frame_index, dtype=np.int64)
    if len(fn_arr) > 1 and not np.all(fn_arr[:-1] < fn_arr[1:]):
        order = np.argsort(fn_arr, kind="stable")
        final_emb = final_emb[order]
        all_frame_index = [all_frame_index[i] for i in order]
        all_frame_types = [all_frame_types[i] for i in order]
        all_is_coarse   = [all_is_coarse[i] for i in order]
        all_entropy     = [all_entropy[i] for i in order]
        log.info("Frame order restored after seam-zoom insertion")

    return (final_emb, all_frame_index, all_frame_types,
            all_black_frames, all_is_coarse, all_entropy, persist_dir)


# =============================================================================
# S3 output
# =============================================================================

def _write_outputs(
    embeddings, frame_index, frame_types, source_times,
    black_frames, is_coarse, entropy, local_shots,
    item, proxy_bucket, dest_prefix,
):
    import io as _io

    window_name = item["window_name"]
    emb_key = f"{dest_prefix}/embeddings/{window_name}.npy"
    meta_key = f"{dest_prefix}/embeddings/{window_name}.json"

    buf = _io.BytesIO()
    np.save(buf, embeddings)
    buf.seek(0)
    put_binary(proxy_bucket, emb_key, buf.getvalue(), "application/octet-stream")
    log.info("Embeddings: s3://%s/%s  %s", proxy_bucket, emb_key,
             embeddings.shape)

    shots_for_json = []
    for s in local_shots:
        d = dict(s)
        d.pop("representative_embeddings", None)
        shots_for_json.append(d)

    cfg = item.get("frames_to_shots_cfg", {})
    meta = {
        "window_name":        window_name,
        "window_id":          item["window_id"],
        "segment_id":         item["segment_id"],
        "segment_name":       item["segment_name"],
        "window_start":       item["window_start"],
        "window_end":         item["window_end"],
        "content_end":        item["content_end"],
        "source_start_frame": item["source_start_frame"],
        "source_end_frame":   item["source_end_frame"],
        "frame_rate":         item["frame_rate"],
        "time_base":          item.get("time_base", ""),
        "coarse_fps":         int(cfg.get("coarse_fps", 4)),
        "frame_count":        len(frame_index),
        "frame_index":        frame_index,
        "frame_type":         frame_types,
        "source_time":        source_times,
        "is_coarse":          is_coarse,
        "entropy":            entropy,
        "black_frames":       black_frames,
        "local_shots":        shots_for_json,
    }
    put_json(proxy_bucket, meta_key, meta)
    log.info("Metadata: s3://%s/%s", proxy_bucket, meta_key)
    return emb_key, meta_key


# =============================================================================
# Lambda handler
# =============================================================================

def lambda_handler(event: dict, context) -> dict:
    log.info("StateExtractFrameFeatures event keys = %s", sorted(event.keys()))
    t0 = time.monotonic()
    inv_uuid = str(uuid.uuid4())[:8]
    tmp_base = f"/tmp/dino_{inv_uuid}"

    try:
        item = event.get("item_data", event)
        proxy_bucket = item["proxy_bucket"]
        proxy_key = item["proxy_key"]
        pts_key = item.get("pts_key")
        dest_prefix = item.get("dest_prefix") or str(
            Path(proxy_key).parent.parent)
        cfg = item.get("frames_to_shots_cfg", {})

        if not pts_key:
            raise ValueError("pts_key is required")

        log.info("Window %s: proxy=%s  pts=%s  frames=%d→%d  %.1fs-%.1fs",
                 item["window_name"], proxy_key, pts_key,
                 item.get("source_start_frame", -1),
                 item.get("source_end_frame", -1),
                 item["window_start"], item["window_end"])

        memory_mb = getattr(context, "memory_limit_in_mb", None)
        session = _get_session(memory_mb)
        pts_data = _load_pts_data(proxy_bucket, pts_key)

        os.makedirs(tmp_base, exist_ok=True)
        local_file = os.path.join(tmp_base, "proxy.mp4")
        download_file(proxy_bucket, proxy_key, local_file)

        # ── Inference (checkpoints + resumes across the 900 s ceiling) ──
        ckpt_key = _checkpoint_key(dest_prefix, item["window_name"])
        time_left_ms = getattr(context, "get_remaining_time_in_millis", None)
        t_proc = time.monotonic()
        (embeddings, frame_index, frame_types,
         black_frames, is_coarse, entropy, persist_dir) = _process_window(
            session, local_file, pts_data, item, cfg, tmp_base,
            time_left_ms=time_left_ms, proxy_bucket=proxy_bucket,
            ckpt_key=ckpt_key)
        proc_ms = round((time.monotonic() - t_proc) * 1000, 1)

        # ── Derive source_times ───────────────────────────────────────
        fps = float(item["frame_rate_fps"])
        source_times = [round(fn / fps, 6) for fn in frame_index]
        # shot timecodes are DERIVED AT MERGE TIME from frame numbers
        # (frame_to_smpte over the container origin) — no per-frame string
        # list is computed or shipped in the window payload any more

        # ── Shot detection ────────────────────────────────────────────
        t_shots = time.monotonic()

        shot_threshold = float(
            cfg.get("shot_boundary_threshold", 0.20))
        local_shots = build_shots(
            embeddings=embeddings,
            frame_index=frame_index,
            frame_type=frame_types,
            source_time=source_times,
            shot_boundary_threshold=shot_threshold,
            frame_entropy=entropy,
        )
        shots_ms = round((time.monotonic() - t_shots) * 1000, 1)
        log.info("Shots: %d  %.0fms", len(local_shots), shots_ms)

        # ── Write outputs ─────────────────────────────────────────────
        t_write = time.monotonic()
        emb_key, meta_key = _write_outputs(
            embeddings, frame_index, frame_types, source_times,
            black_frames, is_coarse, entropy, local_shots,
            item, proxy_bucket, dest_prefix)
        write_ms = round((time.monotonic() - t_write) * 1000, 1)

        # ── Archive model-input frames ────────────────────────────────
        t_archive = time.monotonic()
        frames_key = None
        if persist_dir and os.path.isdir(persist_dir):
            tar_path = _create_frames_archive(
                persist_dir, tmp_base, item["window_name"])
            if tar_path:
                frames_key = (
                    f"{dest_prefix}/embeddings/"
                    f"{item['window_name']}_frames.tar.gz"
                )
                upload_file(proxy_bucket, frames_key, tar_path, {"ContentType": "application/gzip"})
                log.info("Frames archive: s3://%s/%s", proxy_bucket,
                         frames_key)
        archive_ms = round((time.monotonic() - t_archive) * 1000, 1)

        # ── Retire the checkpoint (best effort) ───────────────────────
        # The window's real outputs are all written; a stale checkpoint
        # would only cost storage and confuse the next debugging session.
        try:
            delete_object(proxy_bucket, ckpt_key)
        except Exception as exc:
            # Non-fatal, but never silent: a swallowed AccessDenied is how
            # the first live film_a run orphaned 8 checkpoint tars before
            # the role gained s3:DeleteObject.
            log.warning("Checkpoint delete failed (non-fatal): %s", exc)

        elapsed = round(time.monotonic() - t0, 2)
        log.info(
            "DONE %s: frames=%d  shots=%d  "
            "proc=%.0fms  shots=%.0fms  "
            "write=%.0fms  archive=%.0fms  total=%.1fs",
            item["window_name"], len(frame_index),
            len(local_shots),
            proc_ms, shots_ms, write_ms, archive_ms, elapsed)

        return {
            **item,
            "embeddings_key": emb_key,
            "metadata_key":   meta_key,
            "frames_key":     frames_key,
            "frame_count":    len(frame_index),
            "n_local_shots":  len(local_shots),
            "status":         "COMPLETED",
            "elapsed_sec":    elapsed,
        }

    except Exception as exc:
        log.error("[ERR] %s: %s", type(exc).__name__, exc)
        traceback.print_exc()
        raise
    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)
