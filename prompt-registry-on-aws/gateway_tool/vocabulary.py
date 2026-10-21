"""
vocabulary.py — classify variable names against the reviewed vocabulary.

REGISTRY-ONLY (not a copied module): this is the lint half of the design.
The HARD contract stays per-prompt (the artifact's parsed `variables`,
enforced exactly by prompt_render); this module powers `put_prompt`'s
warnings — never refusals — so a novel name or a legacy spelling is SAID
OUT LOUD at save time and typos don't ship silently.

Aliases in vocabulary.json are ATTESTED legacy spellings from the ported v5
corpus, deliberately not guesses — guessable variants (`scene`, `scenes`,
`segment`, misspellings) are handled here instead, by suggestion: `classify`
answers ("canonical" | "alias" | "novel", detail), and for novel names the
detail is the nearest canonical name when one is plausibly meant. Three
heuristics, in order, all cheap and all explainable in a warning message:

  1. prefix either way against canonical+alias names (`scene` →
     `scene_descriptions`);
  2. the same after a singular/plural flip (`scenes` → `scene` →
     `scene_descriptions`);
  3. difflib closest match at 0.75 (`transcirpt` → `transcript`).

A suggestion is a HINT for a human, so ambiguity resolves to the best single
candidate rather than a list — the warning text should invite checking
vocabulary.json either way.
"""
from __future__ import annotations

import difflib
import json
import os

__all__ = ["load_vocabulary", "classify"]

# Repo layout keeps vocabulary.json at the ROOT with this module under
# gateway_tool/; the deployed zip is FLAT, so both land side by side. Try the
# same directory first (the zip), then the parent (the repo).
_VOCAB_CANDIDATES = (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocabulary.json"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                 "vocabulary.json"),
)


def load_vocabulary(path: str = "") -> dict:
    """{"canonical": {name: entry}, "aliases": {alias: canonical}}."""
    if not path:
        path = next((c for c in _VOCAB_CANDIDATES if os.path.exists(c)),
                    _VOCAB_CANDIDATES[0])
    with open(path) as f:
        raw = json.load(f)
    canonical = raw["variables"]
    aliases = {}
    for name, entry in canonical.items():
        for alias in entry.get("aliases", []):
            aliases[alias] = name
    return {"canonical": canonical, "aliases": aliases}


def _to_canonical(name: str, vocab: dict) -> str:
    return vocab["aliases"].get(name, name)


def _suggest(name: str, vocab: dict):
    candidates = list(vocab["canonical"]) + list(vocab["aliases"])

    def prefix_hit(probe: str):
        for candidate in candidates:
            if candidate.startswith(probe) or probe.startswith(candidate):
                return _to_canonical(candidate, vocab)
        return None

    hit = prefix_hit(name)
    if hit:
        return hit
    flipped = name[:-1] if name.endswith("s") else name + "s"
    hit = prefix_hit(flipped)
    if hit:
        return hit
    close = difflib.get_close_matches(name, candidates, n=1, cutoff=0.75)
    if close:
        return _to_canonical(close[0], vocab)
    return None


def classify(name: str, vocab: dict):
    """
    ("canonical", name)            — nothing to say
    ("alias", canonical_name)      — warn: prefer the canonical spelling
    ("novel", suggestion_or_None)  — warn: novel name; "did you mean X?"
                                     when a near-miss is plausible
    """
    if name in vocab["canonical"]:
        return ("canonical", name)
    if name in vocab["aliases"]:
        return ("alias", vocab["aliases"][name])
    return ("novel", _suggest(name, vocab))
