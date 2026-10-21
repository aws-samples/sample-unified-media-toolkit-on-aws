"""
The save-time parser: authored `.md` → the registry artifact.

REGISTRY-ONLY, deliberately NOT a copied module. The whole §3 design
(README.md Design record §3) rests on the grammar being parsed EXACTLY ONCE, at
save time, by the one component that owns it — consumers receive the artifact's
pre-declared `variables` and never see this file. That is what keeps the copied
`prompt_render.py` small enough to freeze: it substitutes declared tokens and
knows nothing about `:` or `+`.

Ported semantics, from `core-lib/lib/promptConfig/` (reference only, not a
port — v5 files: templateVariables.js, helper.js):

  * HTML-comment header lines BEFORE `# System Prompt` become aux metadata:
    `<!-- Version: default -->` → {"version": "default"}. Key is lowercased and
    trimmed; the value re-joins on ':' so a value may itself contain colons.
    Keys `system`/`user`/`md5` are reserved and ignored (they would collide
    with the artifact's own fields — v5 rule, kept).
  * The body splits case-insensitively on `# System Prompt` / `# User Prompt`;
    both sections are required and both are trimmed.
  * `{{...}}` tokens are collected from the WHOLE body (system + user — v5
    scans both, and tokens do appear in system prompts). Dedup is on the RAW
    inner text, so `{{ scene }}` and `{{scene}}` are two declared tokens that
    parse identically — preserved from v5, because render substitutes by the
    raw token string and must find each spelling.
  * Per token, spaces stripped, then:
      -  legacy `a+b+c` (no `:`)  → root a, leafs [b, c]
      -  `a:b`                    → root a, leafs [b]
      -  `a:b+c`                  → root a, leafs [b, c]
      -  `a:b:c` (nested)         → root a, leafs [{root b, leafs [c]}]
      -  bare `a`                 → root a, leafs []

Stricter than v5 where put_prompt must be strict (v5's fallback path returns
`system: undefined` and carries on; a registry that stores a half-parsed
artifact serves it forever):

  * missing `# System Prompt` or `# User Prompt` → ParseError
  * missing `Version:` header → ParseError (the version names the S3 prefix)
  * an empty token (`{{}}` or `{{ }}`) → ParseError, naming it
"""
from __future__ import annotations

import hashlib
import re

_COMMENT_RE = re.compile(r"<!--([\s\S]*?)-->")
_TOKEN_RE = re.compile(r"\{\{([^}]+)\}\}")

# Categories and versions become S3 key segments and (via history entries)
# path components — the execution-name charset, same as workflow_history's
# scope rule.
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_SYSTEM_SPLIT_RE = re.compile(r"# system prompt", re.IGNORECASE)
_USER_SPLIT_RE = re.compile(r"# user prompt", re.IGNORECASE)

# Reserved header keys that would collide with artifact fields (v5 rule).
_RESERVED_AUX_KEYS = ("system", "user", "md5")


class ParseError(ValueError):
    """The .md is not a valid prompt source; the message says exactly why."""


def parse_comments(text: str) -> dict:
    """`<!-- Key: value -->` lines → {key: value}, keys lowercased."""
    aux: dict = {}
    for match in _COMMENT_RE.finditer(text or ""):
        components = match.group(1).split(":")
        key = components[0].strip().lower()
        value = ":".join(components[1:]).strip()
        if key and key not in _RESERVED_AUX_KEYS:
            aux[key] = value
    return aux


def _parse_hierarchical(token: str) -> dict:
    """`a:b+c` / `a:b:c` / bare `a` → {root, leafs}, recursively (v5 shape:
    every leaf is itself {root, leafs})."""
    parts = token.split(":")
    root = parts[0]
    if len(parts) == 1:
        return {"root": root, "leafs": []}
    remaining = ":".join(parts[1:])
    if ":" in remaining:
        return {"root": root, "leafs": [_parse_hierarchical(remaining)]}
    if "+" in remaining:
        return {"root": root,
                "leafs": [{"root": item, "leafs": []}
                          for item in remaining.split("+")]}
    return {"root": root, "leafs": [{"root": remaining, "leafs": []}]}


def parse_variables(body: str) -> list:
    """
    Every `{{...}}` in the body → a declared variable, v5's exact shape:
    `{token, root, leafs}` where `token` is the raw `{{...}}` string render
    will substitute. Order of first appearance; dedup on the raw inner text.
    """
    seen: dict = {}
    for match in _TOKEN_RE.finditer(body or ""):
        seen.setdefault(match.group(1), None)

    variables = []
    for raw in seen:
        stripped = raw.replace(" ", "")
        if not stripped:
            raise ParseError(f"empty template token {{{{{raw}}}}}")
        if ":" not in stripped and "+" in stripped:
            # Legacy syntax: {{scene+transcript+sound_events}}
            parts = stripped.split("+")
            entry = {"root": parts[0],
                     "leafs": [{"root": item, "leafs": []}
                               for item in parts[1:]]}
        else:
            entry = _parse_hierarchical(stripped)
        variables.append({"token": "{{" + raw + "}}", **entry})
    return variables


def parse_prompt_md(text: str) -> dict:
    """
    The whole save-time parse: authored bytes → the artifact.json body
    (minus registry-supplied fields like `category`/`origin`/`created_at`,
    which `put_prompt` stamps — they are properties of the SAVE, not the text).
    """
    if not text or not text.strip():
        raise ParseError("empty prompt source")
    if not _SYSTEM_SPLIT_RE.search(text):
        raise ParseError("missing '# System Prompt' section")
    if not _USER_SPLIT_RE.search(text):
        raise ParseError("missing '# User Prompt' section")

    header, prompt = _SYSTEM_SPLIT_RE.split(text, maxsplit=1)
    aux = parse_comments(header)
    if not aux.get("version"):
        raise ParseError(
            "missing '<!-- Version: ... -->' header — the version names the "
            "registry prefix, so an unversioned prompt cannot be stored")

    system, user = _USER_SPLIT_RE.split(prompt, maxsplit=1)
    system = system.strip()
    user = user.strip()
    if not system:
        raise ParseError("'# System Prompt' section is empty")
    if not user:
        raise ParseError("'# User Prompt' section is empty")

    return {
        "version": aux.pop("version"),
        "comment": aux.pop("comment", ""),
        "system": system,
        "user_template": user,
        "variables": parse_variables(prompt),
        # md5 of the WHOLE authored text (v5 semantics) — ties artifact.json
        # back to the exact source.md bytes it was parsed from.
        "source_md5": hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest(),
        # Unrecognized headers ride along (e.g. supportedvariables, authored
        # documentation) rather than being dropped — they cost nothing and
        # deleting an author's metadata is not this parser's call.
        "aux": aux,
    }
