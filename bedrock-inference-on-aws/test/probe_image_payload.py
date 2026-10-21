"""Measure a model's REAL image-payload ceiling over the Converse API.

Self-contained on purpose (stdlib + boto3 only, no repo imports) so it can be
copied alone to a remote box — e.g. a US-region EC2 instance for the models
whose geo profile is unreachable from eu (llama4). Deliberately NOT named
test_*.py: it spends real Bedrock money and `discover` must never pick it up.

What it measures, with incompressible noise PNGs (zlib level 0 — the token
plateau / limit-probe method that measured the GPT-6 entries, 2026-10-16):

  1. single-image ceiling: a coarse ladder, then binary refinement to
     --precision-mb between the last pass and the first fail;
  2. (--multi) whether the limit is per-IMAGE or per-REQUEST: k images each
     just under the single-image ceiling, k = 2, 3, 4, ...

Registry semantics the result feeds (`capabilities.image.limits.max_payload_mb`):
the limit bounds INLINE request bytes, so record the measured pass value and
pin the registry a step conservative, the way the GPT-6 entries carry 20
against a measured 22-pass/23-fail.

Examples:
  # pixtral from its home geo
  python3 probe_image_payload.py --model-id eu.mistral.pixtral-large-2502-v1:0 \
      --region eu-west-1
  # llama4 on a us-east-1 EC2 instance (geo profile only resolves in US)
  python3 probe_image_payload.py \
      --model-id us.meta.llama4-maverick-17b-instruct-v1:0 --region us-east-1
  # a thinking-required Claude 5 model needs its quiet contract alongside
  python3 probe_image_payload.py --model-id global.anthropic.claude-opus-5 \
      --region eu-west-1 \
      --extra-json '{"thinking":{"type":"adaptive"},"output_config":{"effort":"low"}}'
"""
import argparse
import json
import math
import os
import struct
import sys
import time
import zlib

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def noise_png_of_mib(target_mib: float) -> bytes:
    """A valid RGB PNG of ~target_mib raw bytes. Random pixels are
    incompressible, so compressed size tracks 3*w*h closely."""
    px = max(16, int(math.sqrt(target_mib * 1024 * 1024 / 3)))
    rows = bytearray()
    for _ in range(px):
        rows.append(0)                       # filter byte per scanline
        rows += os.urandom(3 * px)
    ihdr = struct.pack(">IIBBBBB", px, px, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(bytes(rows), 0))
            + _chunk(b"IEND", b""))


def attempt(client, model_id: str, images: list, extra: dict) -> tuple:
    """(passed: bool, detail: str) for one Converse call."""
    content = [{"image": {"format": "png", "source": {"bytes": img}}}
               for img in images]
    content.append({"text": "Reply OK."})
    request = {"modelId": model_id,
               "messages": [{"role": "user", "content": content}]}
    if extra:
        request["additionalModelRequestFields"] = extra
    for retry in (True, False):
        try:
            response = client.converse(**request)
            return True, f"in={response['usage']['inputTokens']}"
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            msg = exc.response["Error"]["Message"]
            if code in ("ExpiredTokenException", "AccessDeniedException",
                        "UnrecognizedClientException",
                        "InvalidSignatureException"):
                # an auth failure is NOT a refusal either — on a long remote
                # ladder (temporary EC2 credentials) every call after expiry
                # would read as FAIL and the search would converge on the
                # moment the credentials died
                raise RuntimeError(
                    f"auth failure, not a model refusal: {code}: {msg[:150]}"
                    f" — refresh credentials; results printed above remain "
                    f"valid") from exc
            if code in ("ThrottlingException", "ServiceUnavailableException",
                        "InternalServerException", "ModelTimeoutException",
                        "ModelNotReadyException", "ModelErrorException"):
                # a throttle is NOT a refusal — recording it as FAIL would
                # converge the binary search on a load artifact and write a
                # wrong pin into the registry
                if retry:
                    print(f"  transient {code}, backing off 20 s and retrying once")
                    time.sleep(20)
                    continue
                raise RuntimeError(
                    f"transient {code} survived the retries — not a model "
                    f"refusal; results printed above remain valid; rerun "
                    f"to continue") from exc
            return False, f"{code}: {msg[:150]}"
        except BotoCoreError as exc:
            # a transport stall is NOT a model refusal — returning False
            # would poison the boundary. Retry once, then abort loudly;
            # the ladder printed so far remains valid.
            if retry:
                print(f"  transport error ({type(exc).__name__}), "
                      f"backing off 20 s and retrying once")
                time.sleep(20)
                continue
            raise RuntimeError(
                f"transport error, not a model refusal: {exc} — results "
                f"printed above remain valid; rerun to continue") from exc


def probe_single(client, model_id, extra, max_mib, precision_mib):
    print(f"-- single-image ladder (up to {max_mib} MiB)")
    ladder = [m for m in (1, 3, 3.6, 4, 4.5, 5, 6, 8, 10, 15, 20, 24)
              if m <= max_mib]
    last_pass, first_fail = None, None
    for mib in ladder:
        img = noise_png_of_mib(mib)
        ok, detail = attempt(client, model_id, [img], extra)
        print(f"  {len(img)/1048576:6.2f} MiB: {'PASS' if ok else 'FAIL'}  {detail}")
        if ok:
            last_pass = mib
        else:
            first_fail = mib
            break
    while (last_pass is not None and first_fail is not None
           and first_fail - last_pass > precision_mib):
        mid = round((last_pass + first_fail) / 2, 2)
        if mid in (last_pass, first_fail):
            break   # 2-decimal resolution exhausted — repeating a probed size
        img = noise_png_of_mib(mid)
        ok, detail = attempt(client, model_id, [img], extra)
        print(f"  {len(img)/1048576:6.2f} MiB: {'PASS' if ok else 'FAIL'}  {detail}")
        if ok:
            last_pass = mid
        else:
            first_fail = mid
    print(f"single-image: pass {last_pass} MiB"
          + (f" / fail {first_fail} MiB" if first_fail else
             " (ladder top reached — no refusal observed)"))
    return last_pass, first_fail


def tiny_png(px: int = 64) -> bytes:
    """A few-hundred-byte solid PNG — for COUNT probes, where payload must
    stay far from any size limit so only the image count can refuse."""
    rows = bytearray()
    for _ in range(px):
        rows.append(0)
        rows += b"\x80" * (3 * px)
    ihdr = struct.pack(">IIBBBBB", px, px, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(bytes(rows), 6))
            + _chunk(b"IEND", b""))


def probe_count(client, model_id, extra, max_count):
    """Max images per request: a ladder of tiny PNGs, then binary refinement
    to the exact boundary (measured elsewhere: pixtral refuses at 9 naming
    'maximum allowed of 8'; the GPT-6 family at 101 naming 100)."""
    print(f"-- image-count ladder (up to {max_count})")
    img = tiny_png()

    def ok_at(n):
        passed, detail = attempt(client, model_id, [img] * n, extra)
        print(f"  n={n}: {'PASS' if passed else 'FAIL'}  {detail}")
        return passed

    ladder = [n for n in (1, 2, 3, 4, 5, 8, 10, 16, 20, 32, 50, 100,
                          150, 200, 300, 500, 1000) if n <= max_count]
    if max_count not in ladder:
        ladder.append(max_count)   # --count-max is always itself probed
    last_pass, first_fail = None, None
    for n in ladder:
        if ok_at(n):
            last_pass = n
        else:
            first_fail = n
            break
    while (last_pass is not None and first_fail is not None
           and first_fail - last_pass > 1):
        mid = (last_pass + first_fail) // 2
        if ok_at(mid):
            last_pass = mid
        else:
            first_fail = mid
    print(f"image count: pass {last_pass}"
          + (f" / fail {first_fail}" if first_fail
             else " (ladder top reached — no refusal observed)"))


def probe_multi(client, model_id, extra, each_mib, max_k):
    print(f"-- per-request probe: k images of {each_mib} MiB each")
    for k in range(2, max_k + 1):
        imgs = [noise_png_of_mib(each_mib) for _ in range(k)]
        total = sum(len(i) for i in imgs) / 1048576
        ok, detail = attempt(client, model_id, imgs, extra)
        print(f"  k={k} total={total:6.2f} MiB: "
              f"{'PASS' if ok else 'FAIL'}  {detail}")
        if not ok:
            return
    print("  (no per-request refusal within the probed range)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True,
                        help="RUNTIME id, geo prefix included "
                             "(us.meta..., global.anthropic..., eu.mistral...)")
    parser.add_argument("--region", required=True)
    parser.add_argument("--extra-json", default="",
                        help="additionalModelRequestFields as JSON — required "
                             "contract for thinking-required models")
    parser.add_argument("--max-mib", type=float, default=24.0)
    parser.add_argument("--precision-mb", type=float, default=0.25,
                        help="refinement stops at this gap; floored at 0.01 "
                             "— sizes are generated at 2-decimal MiB, so a "
                             "finer ask would loop forever on paid calls")
    parser.add_argument("--multi", action="store_true",
                        help="after the single-image ceiling, probe whether "
                             "the limit is per-image or per-request")
    parser.add_argument("--multi-max-k", type=int, default=6)
    parser.add_argument("--count", action="store_true",
                        help="probe the max image COUNT per request "
                             "(tiny PNGs, so size cannot be the refusal)")
    parser.add_argument("--count-max", type=int, default=100)
    args = parser.parse_args()

    extra = json.loads(args.extra_json) if args.extra_json else None
    client = boto3.client("bedrock-runtime", region_name=args.region,
                          config=Config(read_timeout=120,
                                        retries={"max_attempts": 2}))
    print(f"== {args.model_id} @ {args.region}")
    if args.count:
        probe_count(client, args.model_id, extra, args.count_max)
        return 0
    last_pass, first_fail = probe_single(
        client, args.model_id, extra, args.max_mib,
        max(args.precision_mb, 0.01))
    if args.multi and last_pass:
        each = max(0.5, round(min(last_pass * 0.8, last_pass - 0.5), 2))
        probe_multi(client, args.model_id, extra, each, args.multi_max_k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
