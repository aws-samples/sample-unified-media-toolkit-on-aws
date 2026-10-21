"""
prompt_render.py — pre-declared tokens + caller values → Converse-shaped bytes.

**COPIED MODULE — source of truth: prompt-registry-on-aws/gateway_tool/.**
Distributed the `workflow_identity.py` way (micro-workflow-bundle/README.md §C2/§C5): `cp`'d verbatim
into each consumer workflow, byte-parity + golden vectors in the bundle suite.
Per-consumer variation is FORBIDDEN in this file — there is nowhere for it to
live, because two copies that render differently are two repos that disagree
about `workflow_id` (the hash of rendered bytes), and the symptom is a silent
re-run at a fresh prefix with no failure to notice.

**FREEZE BIAS (prompt-registry-on-aws/README.md Design record §6).** Every behavior change here bumps
every consumer's `.version` — old and new code must not disagree about the
identity of the same request. Designed to change never, not rarely, which is
why it is deliberately dumb:

  * NO grammar parsing — the registry parsed `{{a:b+c}}` once, at save time;
    this module substitutes the artifact's pre-declared `token` strings and
    never inspects what is inside the braces.
  * NO metadata fetching, NO defaulting. v5's `replaceVariables` substituted
    the literal string 'n/a' for an undefined value — a prompt that silently
    ships "n/a" to a model is the exact class of invisible behavior change
    this design exists to kill. A missing value RAISES instead.
  * NO unused values. A `values` key that matches no declared token raises —
    a typo'd key that silently substitutes nothing is the other half of the
    same failure mode.

The contract:

    render(artifact, values) -> {"system": str, "messages": [message]}

`artifact` is the registry's artifact.json (needs `system`, `user_template`,
`variables`). `values` is keyed by the DECLARED TOKEN STRING, exactly as it
appears in `artifact.variables[].token` — `"{{scene:transcript}}"`, braces and
all. Every declared token must be present in `values`; every `values` key must
be a declared token.

Value types, and the frozen rendering of each:

  * `str`            → substituted verbatim.
  * `int` / `float`  → `str(value)`; floats render via repr semantics
                       (`0.5` → "0.5"), never locale-formatted.
  * `dict`, or list  → compact JSON, `ensure_ascii=False`,
    of non-dicts        separators `(",", ":")`, insertion order preserved —
                        byte-stable for a given value (matches v5's
                        JSON.stringify closely; NOT sorted keys, because the
                        author's field order is often prompt-meaningful).
  * list of CONVERSE-SHAPED blocks (every element a SINGLE-KEY dict keyed
    text|image|document|video)
                     → CONTENT BLOCKS (frame/grid images and the like): the
                       user text splits at each token occurrence and the
                       blocks are spliced between text segments, producing a
                       multi-block user message. Block dicts pass through
                       byte-untouched — this module inspects only the outer
                       key, never source.bytes (the transport wants base64
                       strings; bedrock-inference validates the magic).
                       A list of PLAIN data dicts (scene_descriptions,
                       shot_list, ...) is NOT blocks — it stringifies as JSON
                       like any other structure. Settled 2026-08-15: "blocks
                       means image blocks; json means nested data" — the
                       shape test is what makes the code mean that.
  * `bool`, `None`,  → TypeError. There is no honest one-true-string for a
    anything else      bool ("True"? "true"?), and None is a missing value
                       wearing a key.

Every substitution replaces EVERY occurrence of its token (v5 `replaceAll`
semantics) in BOTH the system text and the user template — v5 collects tokens
from both sections and real builtins use them in both. Content-block values
are the one exception: a block-list token appearing in `system` raises,
because a Converse system prompt is text and there is nowhere to splice an
image.

Output shape: `{"system": <str>, "messages": [{"role": "user", "content":
[<block>, ...]}]}` — exactly what `bedrock-inference`'s `run_inference` takes
per record (`system` string + Converse messages). Text segments become
`{"text": ...}` blocks; empty text segments produced by splicing at the very
start/end of the template are dropped (a Converse text block may not be
empty), but the CONTENT of the author's text is never trimmed beyond that.
"""
from __future__ import annotations

import json

__all__ = ["render", "RenderError"]


class RenderError(ValueError):
    """The artifact and values disagree; the message names every offender."""


# The block types a caller may splice. Deliberately the MEDIA types plus text:
# "blocks means image blocks" (settled 2026-08-15) — a list of plain data dicts
# (scene_descriptions, shot_list, ...) is JSON to stringify, and only this
# shape test tells them apart, so it must be strict: every element a
# SINGLE-KEY dict keyed by a Converse content-block type.
_BLOCK_TYPES = frozenset(("text", "image", "document", "video"))


def _is_content_blocks(value) -> bool:
    return (isinstance(value, list) and len(value) > 0
            and all(isinstance(item, dict) and len(item) == 1
                    and next(iter(item)) in _BLOCK_TYPES
                    for item in value))


def _scalar_text(token: str, value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool) or value is None:
        raise RenderError(
            f"{token}: value must be a string, number, dict, or a list — got "
            f"{value!r} (no defaulting: a missing or unrenderable value must "
            "fail loudly, never ship a placeholder)")
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    raise RenderError(
        f"{token}: unrenderable type {type(value).__name__}")


def _splice(text: str, token: str, blocks: list) -> list:
    """The user text → alternating text/block segments at each token site."""
    segments = []
    parts = text.split(token)
    for index, part in enumerate(parts):
        if part:
            segments.append({"text": part})
        if index < len(parts) - 1:
            segments.extend(blocks)
    return segments


def render(artifact: dict, values: dict) -> dict:
    declared = [v.get("token") for v in (artifact.get("variables") or [])]
    declared_set = set(declared)

    missing = [t for t in declared if t not in values]
    unknown = [k for k in values if k not in declared_set]
    if missing or unknown:
        problems = []
        if missing:
            problems.append("missing values for declared tokens: "
                            + ", ".join(missing))
        if unknown:
            problems.append("values for undeclared tokens: "
                            + ", ".join(sorted(unknown)))
        raise RenderError("; ".join(problems))

    system = artifact.get("system") or ""
    user = artifact.get("user_template") or ""

    # Scalars first, into both sections; block-lists collected for splicing.
    block_tokens: dict = {}
    for token in declared:
        value = values[token]
        if _is_content_blocks(value):
            if token in system:
                raise RenderError(
                    f"{token}: content blocks cannot be spliced into the "
                    "system prompt — a Converse system is text only")
            block_tokens[token] = value
            continue
        text = _scalar_text(token, value)
        system = system.replace(token, text)
        user = user.replace(token, text)

    if not block_tokens:
        content = [{"text": user}]
    else:
        # Splice iteratively: split on one token at a time so earlier splices'
        # text segments are still searchable for the remaining tokens.
        content = [{"text": user}]
        for token, blocks in block_tokens.items():
            next_content = []
            for block in content:
                if "text" in block and token in block["text"]:
                    next_content.extend(_splice(block["text"], token, blocks))
                else:
                    next_content.append(block)
            content = next_content

    return {"system": system,
            "messages": [{"role": "user", "content": content}]}
