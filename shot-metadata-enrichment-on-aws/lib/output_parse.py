"""
Model-output parsing, coverage, and the pairing repair (README.md §3-§4).

The model returns JSONL: one self-contained document per REQUIRED shot,
keyed by shot_id. JSONL is the recovery format (GOTCHAS: the model fences
its JSON no matter how firmly you ask — unless it is JSONL): truncation
loses at most the final line, a malformed line damages only itself, and
coverage names exactly which shot_ids remain unanswered.

parse_response() reads every line STRICTLY first, and only a line that
already failed climbs the three-rung ladder in the COPIED
parse_partial_output module (source of truth: bedrock-inference-on-aws;
user decisions — 2026-08-25: "try our best to recover ... if it still
fails, fallback to the default behavior - missing data for that shot";
2026-08-26: no external package, rung 3 is our own prefix salvage, and
the gate stays at the MINIMUM):

    1. json.loads                      the answer for 3452/3462 real lines
    2. escape_inner_quotes -> STRICT   the ONE measured malformation
    3. prefix salvage, gated           truncation/damage -> honest prefix
    4. a bad line, and the shot a gap  unchanged

REQUIRED_KEYS is ("shot_id",) alone — the one key without which a
document cannot join shots.json (parse_response refuses such a line
anyway). Everything else degrades by DESIGN and is counted: a lost
starts_new_scene becomes the merge-biased false via repair_pairing
(kind: missing_flag), lost lists default to [] in normalize_doc. Gating
wider would throw away a recovered partial because it lost OPTIONAL
metadata — paid-for data discarded over fields nothing hard-requires.

The minimal gate is safe ONLY because rung 3 cannot fabricate: every
value it admits parsed strictly from the model's own bytes, and an
incomplete tail value is dropped whole. That property is why json-repair
was removed (measured, TDY/NNN + 20 reconstructed failure lines,
2026-08-26): rung 2 recovers all real failures — chyrons containing
literal quotation marks (`MIRAVOX RETURNS WITH "STILL GOLD"`) — byte-equal
to ground truth with all 3452 valid lines byte-identical, matching
json-repair's recovery exactly; but on a string cut mid-word json-repair
returns the silently trimmed `'indetermina'`, fabricated metadata nothing
downstream would catch, where our completer drops the pair. Each recovery
is COUNTED per rung and reported, so a prompt regression shows up as a
rising `line_repairs` number instead of hiding behind a forgiving parser.

repair_pairing() is the §4 rule, both directions measured: on a
flag/reason cross-pairing **`false` WINS** — a `false`-list reason forces
the flag to `false`, and a `true`-list reason on a `false` flag keeps the
flag `false`. The model's reason stays VERBATIM for debugging, the repair
is counted (the mismatch rate is an interleaved-genre detector: 0% linear,
2-8% current affairs), and the bias matches the prompt's own rule 4 — a
merge is repairable downstream, a spurious split has already destroyed
the grouping.
"""
from __future__ import annotations

import json
import re

from lib.parse_partial_output import parse_jsonl_line

TRUE_REASONS = {"location_change", "time_jump", "new_conversation",
                "new_line_of_action", "graphics_boundary", "after_black"}
FALSE_REASONS = {"same_location", "continues_action",
                 "continues_conversation", "reverse_angle",
                 "insert_of_same_scene", "intercut_return",
                 "no_location_evidence"}

_LIST_FIELDS = ("lighting", "notable", "people", "texts_on_screen",
                "named_entities", "brands", "objects")

def _section_vocabulary(prompt_system: str, header: str, entry_keys: set):
    """The CLOSED vocabulary the BOUND PROMPT ITSELF declares — the
    backticked tokens inside the named section, minus the entry keys.
    The prompt stays the single vocabulary authority (as it is for
    sentiment/shot_kind/camera): code enforces whatever the version that
    RAN declares, so a custom `prompt_version` brings its own closed
    list instead of being pruned against ours (user concern 2026-09-03 —
    a hardcoded list would fail every custom prompt).

    None means "do not prune": either the prompt has no such section
    (the field was never asked), or its section declares no backticked
    tokens (the author chose an OPEN vocabulary). The prune itself
    exists because the trailer trial caught the model inventing
    `fight_violence` and `chase_pursuit` — judgment lives in the prompt,
    verification here conforms the output to THAT judgment, never to
    somebody else's."""
    m = re.search(rf"{re.escape(header)}.*?(?=\n#{{2,3}} |\Z)",
                  prompt_system or "", re.S)
    if not m:
        return None
    # ONE declaration grammar for every attested section (review,
    # 2026-09-07): individually backticked tokens, the depicted style —
    # media_depictions was briefly declared as a pipe alternation inside
    # its JSON snippet and grew a second parser branch; the prompt was
    # conformed instead (a pipe string inside an example value also
    # invites a literal model echo).
    toks = set(re.findall(r"`([a-z_]+)`", m.group(0))) - entry_keys
    return frozenset(toks) if toks else None


def depicted_vocabulary(prompt_system: str):
    """See _section_vocabulary — the `### depicted` closed list."""
    return _section_vocabulary(prompt_system, "### depicted",
                               {"act", "detail"})


def media_vocabulary(prompt_system: str):
    """See _section_vocabulary — the `### media_depictions` closed list
    (the depicted-face guard channel, rolled in 2026-09-06 after the
    two-window news trial validated the section)."""
    return _section_vocabulary(prompt_system, "### media_depictions",
                               {"medium", "detail"})

# The salvage gate, at the MINIMUM the code cannot proceed without (user
# decision 2026-08-26): the join key, nothing else. Wider would discard a
# recovered partial over optional metadata; narrower is impossible —
# normalize_doc raises without it and parse_response refuses the line.
REQUIRED_KEYS = ("shot_id",)


def parse_response(text: str) -> dict:
    """-> {"docs": {shot_id: dict}, "bad_lines": [...], "duplicates":
    [shot_id], "line_repairs": {tier: count}}. Never raises on content; a
    duplicate shot_id keeps the FIRST document (the later one is usually
    the degraded tail)."""
    docs: dict = {}
    bad: list = []
    duplicates: list = []
    line_repairs: dict = {}
    for line_no, raw in enumerate((text or "").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        obj, tier = parse_jsonl_line(line, REQUIRED_KEYS)
        if obj is None:
            bad.append({"line_no": line_no, "error": tier,
                        "text": line[:200]})
            continue
        if not isinstance(obj, dict) or "shot_id" not in obj:
            bad.append({"line_no": line_no,
                        "error": "not a shot document", "text": line[:200]})
            continue
        try:
            sid = int(obj["shot_id"])
        except (TypeError, ValueError):
            bad.append({"line_no": line_no,
                        "error": f"non-integer shot_id "
                                 f"{obj.get('shot_id')!r}",
                        "text": line[:200]})
            continue
        if sid in docs:
            duplicates.append(sid)
            continue
        if tier != "strict":
            # Counted only for a line that BECAME a document — a salvage
            # whose output was then rejected is a bad line, not a
            # recovery (2026-08-26 review: counting it as both skewed
            # the honesty signal in two directions at once).
            line_repairs[tier] = line_repairs.get(tier, 0) + 1
        docs[sid] = obj
    return {"docs": docs, "bad_lines": bad, "duplicates": duplicates,
            "line_repairs": line_repairs}


def repair_pairing(doc: dict) -> dict:
    """§4, in place-ish: returns {"repaired": bool, "kind": str | None}.
    `false` wins in both directions; the reason stays verbatim."""
    flag = doc.get("starts_new_scene")
    reason = doc.get("boundary_reason")
    if flag is True and reason in FALSE_REASONS:
        doc["starts_new_scene"] = False
        return {"repaired": True, "kind": "true_flag_false_reason"}
    if flag is False and reason in TRUE_REASONS:
        # The flag is already the answer we keep — the pairing is still a
        # defect worth counting (and the reason worth keeping verbatim).
        return {"repaired": True, "kind": "false_flag_true_reason"}
    if not isinstance(flag, bool):
        # A missing/non-bool flag degrades to the merge-biased answer.
        doc["starts_new_scene"] = False
        return {"repaired": True, "kind": "missing_flag"}
    return {"repaired": False, "kind": None}


def normalize_doc(doc: dict, depicted_vocab=None, media_vocab=None) -> tuple:
    """Defaulted optional fields, list fields coerced — the ONE shape
    scene_assembly and the shots label consume. Field ORDER preserved from
    the model where present (hybrid-search reads keys, not positions).

    Returns (doc, pruned) where pruned lists any `depicted` entries
    dropped against `depicted_vocab` (the BOUND prompt's own declared
    list — see depicted_vocabulary) — the caller logs them (a prune must
    never be silent; a rising count is the prompt-regression signal).
    With no vocab, entry SHAPE is still coerced but no act is judged.

    `depicted` and `media_depictions` are normalized ONLY when the model
    emitted the key. An ABSENT key stays absent — [] asserts "none is
    visible" and only a prompt that ASKED may make that assertion; a doc
    from a prompt without the section must never carry a false
    attested-none (the caller injects [] for omissions when it knows
    the prompt asked)."""
    out = dict(doc)
    out["shot_id"] = int(doc["shot_id"])
    out["starts_new_scene"] = bool(doc.get("starts_new_scene"))
    for f in _LIST_FIELDS:
        v = out.get(f)
        out[f] = v if isinstance(v, list) else []
    out["description"] = str(doc.get("description") or "")
    pruned: list = []
    pruned += _prune_attested(out, "depicted", "act", depicted_vocab)
    pruned += _prune_attested(out, "media_depictions", "medium",
                              media_vocab)
    derive_text_views(out, doc)
    return out, pruned


def derive_text_views(out: dict, raw: dict) -> None:
    """Single-source-of-truth text views (user design, 2026-09-09): each
    on-screen string is SAMPLED once by the model and every other view is
    derived here — restating one observation across parallel fields made
    each restatement an independent sampling event, and they disagreed
    (measured: 40-65% of name supers had no paired entity).

    1. Super `names` flatten into `named_entities` with evidence
       `overlay_text` — post-production identities are stated inside
       their graphic's entry; the flat entity list is a view.
    2. `texts_on_screen` becomes the UNCLAIMED remainder:
       - a doc WITHOUT the key (the news prompt no longer emits it) gets
         the texts of supers that yielded no names;
       - a doc WITH the key (episodic; pre-2026-09-09 news runs) keeps
         only lines NOT fully covered by the overlay entities' word
         tokens — full coverage, so a line carrying more than its entity
         ("PARIS, 1943" vs place `Paris`) survives intact. A missed
         extraction degrades to the line staying visible here, never to
         text loss.

    Data-driven on the doc's own shape — no genre or prompt-version
    branch. Mutates `out` in place."""
    supers = [s for s in out.get("supers") or [] if isinstance(s, dict)]
    ents = list(out.get("named_entities") or [])
    # dedup ONLY against existing overlay_text copies: the model can
    # double-emit a chyron identity as a stray entity with the WRONG
    # evidence (measured live 2026-09-09: a guest arrived typed
    # environment_text), and a (name, type)-only key let that stray
    # suppress the correctly-evidenced derived copy — hiding the person
    # from every overlay_text consumer (cast-resolution's channel)
    have = {(str(e.get("name", "")).casefold(), e.get("type"))
            for e in ents if isinstance(e, dict)
            and e.get("evidence") == "overlay_text"}
    for s in supers:
        for n in s.get("names") or []:
            if not (isinstance(n, dict) and n.get("name")):
                continue
            key = (str(n["name"]).casefold(), n.get("type"))
            if key in have:
                continue
            have.add(key)
            ents.append({"name": n["name"], "type": n.get("type"),
                         "evidence": "overlay_text"})
    out["named_entities"] = ents

    if "texts_on_screen" not in raw:
        out["texts_on_screen"] = [s["text"] for s in supers
                                  if s.get("text") and not s.get("names")]
        return
    overlay_tokens: set = set()
    for e in ents:
        if isinstance(e, dict) and e.get("name") \
                and e.get("evidence") == "overlay_text":
            overlay_tokens |= set(_WORD.findall(str(e["name"]).casefold()))
    def claimed(line):
        toks = set(_WORD.findall(str(line).casefold()))
        return bool(toks) and toks <= overlay_tokens
    out["texts_on_screen"] = [t for t in out.get("texts_on_screen") or []
                              if not claimed(t)]


def _prune_attested(out: dict, field: str, key: str, vocab) -> list:
    """Normalize ONE attested-list field in place ({key, detail} entries,
    optionally judged against the bound prompt's closed vocabulary) and
    return what was pruned — the caller logs it (a prune must never be
    silent; a rising count is the prompt-regression signal)."""
    if field not in out:
        return []
    entries = out[field]
    entries = entries if isinstance(entries, list) else []

    def ok(e):
        return isinstance(e, dict) and isinstance(e.get(key), str) \
            and (vocab is None or e[key] in vocab)
    out[field] = [e for e in entries if ok(e)]
    return [str(e.get(key)) if isinstance(e, dict)
            else str(e)[:40] for e in entries if not ok(e)]


def validate_coverage(docs: dict, required_ids: list) -> dict:
    """Every REQUIRED shot_id exactly once; extra ids (context shots the
    model was told not to emit, hallucinated ids) are named, never kept.
    -> {"ok", "missing": [ids], "extra": [ids]}."""
    required = set(required_ids)
    got = set(docs)
    missing = sorted(required - got)
    extra = sorted(got - required)
    return {"ok": not missing and not extra,
            "missing": missing, "extra": extra}


_WORD = re.compile(r"\w+")


def name_super_entity_gap(docs: dict) -> dict:
    """name_super supers with NO named_entity drawn from their text —
    the model transcribed a person-identifying graphic but extracted
    nothing from it. Measured 2026-09-08 across five news runs on the
    old two-field emission: 40-65% of chyron-identified people never got
    an entity, silently starving cast-resolution's overlay channel —
    which is why extraction now nests INSIDE the super (`names`,
    2026-09-09) and the flat entity list is derived.

    Runs on DERIVED docs, so on the nested shape a gap means the model
    emitted a name_super with empty `names` — the contradiction the
    prompt forbids; on pre-2026-09-09 docs it measures the old two-field
    disagreement. The check is language-agnostic string containment
    between the model's own outputs (an entity whose casefolded
    word-tokens all appear in the super's text counts as drawn from it)
    — never a lexicon, never a name-shape rule. It UNDER-counts the
    legacy shape (affiliation-only extraction passes); zero false
    positives is the design point — a trend gauge, not a contract gate.

    Coverage is judged per SHOT, not per super (RNC eyeball,
    2026-09-09): broadcasts split one lower-third into a name line and
    an office line, BOTH classified name_super, and the office line
    correctly extracts nobody; the model also normalizes ("UNIV." ->
    "University") and completes truncated OCR across frames, defeating
    verbatim token containment on the very supers it handled BEST. A
    shot whose supers yielded ANY person extraction is covered — 19 of
    19 residuals on the RNC run were these shapes, zero real misses —
    while the real failure mode (chyron transcribed, nothing extracted
    anywhere in the shot) stays fully detected.
    `count` is name supers on UNCOVERED shots; `shot_ids` those shots.
    -> {"count", "supers_total", "shot_ids"} (shot_ids capped at 50)."""
    gap_ids, gaps, total = set(), 0, 0
    for sid, d in docs.items():
        # overlay-evidenced persons only: a landmark/placard person in
        # the filmed scene must not cover an unextracted chyron
        persons = any(isinstance(e, dict) and e.get("type") == "person"
                      and e.get("name")
                      and e.get("evidence") == "overlay_text"
                      for e in d.get("named_entities") or [])
        supers = [s for s in d.get("supers") or []
                  if isinstance(s, dict)
                  and s.get("super_type") == "name_super"
                  and s.get("text")]
        total += len(supers)
        # any typed identity on the shot's supers also covers: a
        # name_super whose subject IS an organization (a sponsor card,
        # a marching band — measured 2026-09-09) was handled, not
        # missed. A person chyron mis-typed org hides here — accepted,
        # same undercount class as the legacy shape's.
        typed = any(s.get("names") for s in supers)
        if supers and not persons and not typed:
            gaps += len(supers)
            gap_ids.add(sid)
    return {"count": gaps, "supers_total": total,
            "shot_ids": sorted(gap_ids)[:50]}
