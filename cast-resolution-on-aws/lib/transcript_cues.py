"""
transcript_cues.py — deterministic name cues mined from the transcript.

The cheapest evidence in the whole system: "I'm Miriam Okafor" binds a name to
the CURRENT speaker with zero frames (README.md §8). P0 ships regex seeds for
the strong English patterns only; fuzzier resolution (vocatives across
speakers, non-English forms) is the P2 LLM adjudication pass. Every cue
carries its mechanism so the corpus gate can score each pattern's
precision separately.

Cue kinds:
  self_id      — "I'm NAME" / "I am NAME" / "my name is NAME"
                 -> names the utterance's OWN speaker.
  sign_off     — "NAME, ORG(, PLACE)." reporter package sign-off
                 -> names the utterance's OWN speaker.
  introduction — "joining us now is NAME" / "our guest NAME" / ...
                 -> names the NEXT NEW speaker (resolved in persons.py),
                    and predicts a lower-third at that speaker's first
                    turn (the P1 read scheduler consumes this too).
  vocative     — "NAME, ..." / "..., NAME." -> names the ADDRESSEE.
                 Extracted, NOT bound in P0 (cross-speaker resolution is
                 genuinely hard; see PLAN edge cases).
"""
from __future__ import annotations

import re

# One-to-three capitalized tokens; allows O'Brien / hyphenated surnames.
_NAME = r"([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){0,2})"

_SELF_ID = re.compile(
    rf"\b(?:I'm|I am|my name is)\s+{_NAME}", re.UNICODE)
# trigger phrases are case-tolerant per-word (sentence-initial "Joining…");
# a global IGNORECASE would also relax _NAME's capitalization requirement
_INTRO = re.compile(
    rf"\b(?:[Jj]oining (?:us|me)(?: now)?(?: is)?|[Oo]ur guest(?:s)?,?|"
    rf"[Pp]lease welcome|[Ww]e're joined (?:now )?by|"
    rf"[Ww]ith (?:us|me) (?:now|today)(?: is)?)[,:]?\s+{_NAME}")
# "Miriam Okafor, Metro News, Dallas." — name comma org; org must contain a
# newsroom word so plain appositives don't fire.
_SIGN_OFF = re.compile(
    rf"\b{_NAME},\s+([A-Z][A-Za-z&\s]*(?:News|Network|Radio|Sports|"
    rf"Correspondent|Tonight))\b")
_VOCATIVE_LEAD = re.compile(rf"^{_NAME},\s")
_VOCATIVE_TAIL = re.compile(rf",\s{_NAME}[.!?]$")

# Sentence-initial "I'm Sorry." style false hits: a single capitalized
# common word is not a name. Two-token matches skip this filter entirely.
_NON_NAME_SINGLES = frozenset(
    w.lower() for w in (
        "sorry afraid sure fine okay ok good great serious here home back "
        "done ready right dead gone lost listening talking asking telling "
        "saying going coming leaving staying trying").split())

# Sentence-lead interjections match the vocative shape ("Well, ..." /
# "Yeah, ...") and can rack up huge vote counts (P0: 'Well' x25 on
# a business-news asset) — vote aggregation cannot fix systematic
# extraction noise.
# English heuristic seeds, like the patterns themselves; the P2 LLM pass
# generalizes.
_INTERJECTIONS = frozenset(
    w.lower() for w in (
        "well yeah yes no oh hey now so look listen wait wow okay ok um "
        "uh hmm right sure please thanks god jesus man boy anyway besides "
        "meanwhile still first second finally honestly seriously actually "
        "remember stop come go see why how what when where who and but or "
        "because also maybe perhaps really then there gosh dude whoa damn "
        "hell gee huh alright hello hi goodbye bye welcome congratulations "
        "objection relax careful easy quick hurry help").split())

# Honorifics and role terms of address are REAL evidence ("Detective",
# "Your Honor", "Doc" — one detective is addressed as all three) but
# they are
# ROLES, never person NAMES. Emitted as kind="address_term" for the role
# work; excluded from name binding.
_ADDRESS_TERMS = frozenset(
    w.lower() for w in (
        "detective doctor doc officer sergeant sarge lieutenant captain "
        "chief judge counselor professor coach agent deputy commissioner "
        "senator governor congressman congresswoman president madam sir "
        "maam mister miss missus mr mrs ms dr mom dad mother father honey "
        "sweetheart darling buddy pal folks everyone everybody guys boot "
        "maestro "
        # officials / military / clergy (measured 2026-09-24: a news
        # guest addressed "Ambassador" 7 times was vocative-named
        # Ambassador and it became his character; "Chairman" was minted
        # as a person on an earlier run of the same programme)
        "ambassador secretary chancellor minister premier chairman "
        "chairwoman mayor sheriff marshal general colonel admiral "
        "corporal reverend pastor rabbi bishop nurse principal dean "
        "warden").split()) | frozenset(("your honor",))


def _plausible(name: str) -> bool:
    parts = name.split()
    if len(parts) >= 2:
        return True
    return parts[0].lower() not in _NON_NAME_SINGLES


def _vocative_class(name: str) -> str | None:
    """'vocative' (name-bindable) | 'address_term' (role evidence) |
    None (interjection noise)."""
    low = name.lower().replace("'", "")
    if low in _ADDRESS_TERMS:
        return "address_term"
    parts = low.split()
    if parts[0] in _INTERJECTIONS:
        return None
    if len(parts) == 1 and parts[0] in _NON_NAME_SINGLES:
        return None
    if all(p in _ADDRESS_TERMS for p in parts):
        return "address_term"
    return "vocative"


# Every pattern in this module is an ENGLISH seed. On any other language
# they do not merely miss — some FALSE-FIRE (German capitalizes every
# noun, so the capitalization-shaped vocative pattern extracts junk that
# the English interjection stoplist cannot filter, and vote aggregation
# would then bind wrong names). The gate fails CLOSED to abstention;
# multilingual cue mining is the P2 LLM pass (language-capable by
# construction), never a second stack of per-language regexes.
CUE_LANGUAGES = frozenset({"en"})


def extract_cues(turns: list[dict], language: str = "en") -> list[dict]:
    """turns: sources.load_transcript()['turns'] -> cue list
    {kind, name, speaker_id, sentence_id, t, confidence}.
    language: the transcript's detected language; non-English returns []
    (callers should treat that as NOT ASSESSED, not attested-none)."""
    if (language or "").lower().split("-")[0] not in CUE_LANGUAGES:
        return []
    cues = []

    def add(kind, name, turn, confidence):
        name = " ".join(name.split())
        if not _plausible(name):
            return
        cues.append({"kind": kind, "name": name,
                     "speaker_id": turn["speaker_id"],
                     "sentence_id": turn["sentence_id"],
                     "t": turn["start"], "confidence": confidence})

    for turn in turns:
        text = turn["text"]
        for m in _SELF_ID.finditer(text):
            # a full name is strong; a lone given name is weak evidence
            add("self_id", m.group(1), turn,
                "high" if " " in m.group(1) else "low")
        for m in _INTRO.finditer(text):
            add("introduction", m.group(1), turn,
                "high" if " " in m.group(1) else "low")
        m = _SIGN_OFF.search(text)
        if m:
            add("sign_off", m.group(1), turn, "high")
        # lead and tail are independent slots ("Objection, Your Honor.")
        for m in (_VOCATIVE_LEAD.match(text), _VOCATIVE_TAIL.search(text)):
            if not m:
                continue
            kind = _vocative_class(m.group(1))
            if kind:
                cues.append({"kind": kind,
                             "name": " ".join(m.group(1).split()),
                             "speaker_id": turn["speaker_id"],
                             "sentence_id": turn["sentence_id"],
                             "t": turn["start"], "confidence": "low"})
    return cues
