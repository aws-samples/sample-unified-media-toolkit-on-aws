#!/usr/bin/env python3
"""ARTWORK_SELECTION_PLAN stage 4 — the vision rerank, validated on the
2026-09-12 stage-1-3 prototype output. One vision request per asset
over ALL its pre-qualified frames: the model judges ONLY aesthetics and
technical quality (sharpness, motion blur, eyes/expression, exposure,
composition) — content is already qualified by stage 1-3 metadata
filters — and re-ranks within each archetype, flagging unusable frames.

    venv/bin/python rerank_vision.py <workdir> [<workdir>...]
"""
import base64
import json
import os
import re
import sys

from anthropic import AnthropicBedrockMantle

MODEL = "anthropic.claude-opus-5"
FRAME_RE = re.compile(r"^(.*?)_?shot(\d{3,4})\.jpg$")

RUBRIC = """You are the aesthetic judge in an artwork-selection
pipeline. Every frame you see was ALREADY qualified on content by
metadata filters (right people, right shot type, no on-screen text, no
spoilers). Your job is ONLY the things metadata cannot see:

- technical: sharpness, motion blur, compression artifacts, exposure,
  color
- faces: eyes open, natural expression, not mid-word / mid-blink
- composition: framing, headroom, balance, background clutter
- "would this survive as a thumbnail at small size?"

For each archetype group, rank the frames best-first and flag any frame
that should never ship (badly blurred, eyes shut, grotesque mid-speech
face). Respond with ONLY a JSON object:
{"<archetype>": {"ranking": ["<file>", ...],
                 "top_reason": "<one line on why the winner wins>",
                 "unusable": [{"file": "<file>", "why": "<short>"}]}}
"""


def frames_of(wd):
    """Group frames by archetype, ordered by the stage-1-3 metadata
    rank from candidates.json (directory order is shot-number order,
    NOT the rank)."""
    by_arch = {}
    shots = {}
    for f in sorted(os.listdir(wd)):
        m = FRAME_RE.match(f)
        if m:
            arch = (m.group(1) or "frame").rstrip("_") or "frame"
            by_arch.setdefault(arch, []).append(f)
            shots[(arch, int(m.group(2)))] = f
    cands = json.load(open(os.path.join(wd, "candidates.json")))
    groups = {}
    for arch, files in by_arch.items():
        key = next((k for k in cands
                    if k == arch or k.startswith(arch)
                    or arch.startswith(k)), None)
        if key:
            ranked = [shots[(arch, e["shot"])] for e in cands[key]
                      if (arch, e["shot"]) in shots]
            ranked += [f for f in files if f not in ranked]
            groups[arch] = ranked
        else:
            groups[arch] = files
    return groups


def rerank(client, wd):
    groups = frames_of(wd)
    if not groups:
        print(f"{wd}: no frames")
        return
    content = []
    for arch, files in groups.items():
        content.append({"type": "text",
                        "text": f"--- archetype: {arch} "
                                f"({len(files)} frames) ---"})
        for f in files:
            data = base64.standard_b64encode(
                open(os.path.join(wd, f), "rb").read()).decode()
            content.append({"type": "text", "text": f"file: {f}"})
            content.append({"type": "image",
                            "source": {"type": "base64",
                                       "media_type": "image/jpeg",
                                       "data": data}})
    content.append({"type": "text", "text": "Rank now."})

    resp = client.messages.create(
        model=MODEL, max_tokens=2000, system=RUBRIC,
        messages=[{"role": "user", "content": content}])
    text = "".join(b.text for b in resp.content if b.type == "text")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    verdict = json.loads(m.group(0))
    json.dump(verdict, open(os.path.join(wd, "rerank.json"), "w"),
              indent=1)

    name = os.path.basename(os.path.abspath(wd))
    print(f"\n== {name}")
    for arch, files in groups.items():
        v = verdict.get(arch) or {}
        rank = v.get("ranking") or []
        meta_top = files[0]           # stage-1-3 order = directory order
        vis_top = rank[0] if rank else "?"
        moved = "  <- RERANKED" if vis_top != meta_top else ""
        print(f"  {arch:14} metadata-top={meta_top}  "
              f"vision-top={vis_top}{moved}")
        if v.get("top_reason"):
            print(f"    why: {v['top_reason'][:110]}")
        for u in v.get("unusable") or []:
            print(f"    UNUSABLE: {u.get('file')} — "
                  f"{str(u.get('why'))[:80]}")
    u = resp.usage
    print(f"  tokens in/out: {u.input_tokens}/{u.output_tokens}")


def main():
    client = AnthropicBedrockMantle(
        aws_region=os.environ.get("BEDROCK_REGION", "eu-west-1"))
    for wd in sys.argv[1:]:
        rerank(client, wd)


if __name__ == "__main__":
    main()
