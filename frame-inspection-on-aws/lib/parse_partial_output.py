"""Salvage structured content from a truncated or malformed model response.

COPIED, NOT IMPORTED (bedrock-inference-on-aws/README.md §P6.2). The source of truth lives
in bedrock-inference-on-aws/lib/; consumer workflows copy the file verbatim, and
the bundle suite pins byte-parity plus golden known-answer vectors the day a
second copy exists. Per-consumer variation must live in the CALL, never here —
that is what keeps parity a file comparison. Freeze bias applies: behaviour
changes re-identify work in every consumer, so prefer never to rarely.

Four helpers, one contract:

    parse_partial_json(text)              -> (value | None, was_repaired: bool)
    parse_partial_xml(text)               -> (xml_string | None, was_repaired: bool)
    escape_inner_quotes(line)             -> line, in-value quotes escaped
    parse_jsonl_line(line, required_keys) -> (obj, tier) | (None, error)

All are SCHEMA-AGNOSTIC — they recover the largest well-formed prefix of a
`stop_reason: max_tokens` truncation or a prose-wrapped answer, and nothing
more. Whether the salvaged value is a valid scene-metadata record (or whatever
the caller asked for) is the caller's own check, next to its schema: salvage
that knew schemas would leak every workflow's output shape into a shared file.
`parse_jsonl_line` keeps that rule the same way its siblings do: the
`required_keys` minimum is the CALLER's, stated per call, never encoded here.

Why the tuple rather than raising: to the caller a failed salvage is a normal
per-record outcome (a FAILED line with the raw text kept), not an exception
path — and `was_repaired` must survive into the record so a repaired answer is
never mistaken for a clean one. `(None, False)` means nothing recoverable.

Adapted from v5's `segmentHelper/models/outputParser.js` (reference, not a
port): same bracket-scanning entry, but the repair is a single well-defined
truncation-completion pass instead of jsonrepair + a quote-stripping
"exhaustive" loop — that loop deletes arbitrary quotes on each retry, and a
salvage that silently ALTERS balanced content is worse than one that gives up.
Everything here only ever completes, trims, or — the one verb added
2026-08-26, measured before it was admitted — ADDS an escape character;
nothing ever deletes, replaces or reorders content. That third verb is
`escape_inner_quotes`, and adding is safe where deleting is not: an escape
that lands on a quote which really did close its string leaves the line
unparseable (the caller's next rung or gap accounting takes over), while
v5's deletions turned wrong guesses into silently different content. The
external packages were measured against this file before being rejected
(TDY/NNN corpus, 3,462 real JSONL documents, 2026-08-26): npm jsonrepair /
PyPI json-repair and npm partial-json both silently trim a string cut
mid-word to `"indeterm"` — fabricated data no downstream check would catch
— where `_complete_truncated_json` drops the incomplete tail value whole.

`parse_partial_xml` is new (2026-08-11, no v5 counterpart): prompts that ask
for XML output truncate mid-tag just as JSON ones truncate mid-string. It
returns a well-formed STRING rather than a parsed tree, deliberately — callers
differ on parser (ElementTree, regex extraction, an LLM re-read), and handing
back text keeps this dependency-free and the caller's parser their own.
"""
from __future__ import annotations

import json
import re

__all__ = ["parse_partial_json", "parse_partial_xml",
           "escape_inner_quotes", "parse_jsonl_line"]


# ── JSON ──────────────────────────────────────────────────────────────────────

def parse_partial_json(text: str):
    """`(value, was_repaired)` — the parsed JSON payload of a model answer, or
    `(None, False)` when nothing recoverable is there.

    Handles, in order of increasing damage:
      1. clean JSON, possibly wrapped in prose or a ```json fence  -> repaired=False
         (wrapping alone does not count as repair: the JSON itself was intact);
      2. trailing garbage after a complete value (prose after the close);
      3. truncation — unterminated string, unclosed brackets, dangling comma or
         `"key":` with no value — completed to the largest coherent prefix.
    """
    if not text or not isinstance(text, str):
        return None, False

    start = _first_bracket(text)
    if start < 0:
        return None, False
    candidate = text[start:]

    # The intact case, including "value then prose": scan to the matching close
    # of the FIRST value and parse exactly that span.
    span = _balanced_span(candidate)
    if span:
        try:
            return json.loads(span), False
        except ValueError:
            pass

    # Truncated. Complete the prefix: cut back to the last comma-safe point,
    # close any open string, strip a dangling `"key":` or `,`, then close every
    # open bracket in stack order.
    repaired = _complete_truncated_json(candidate)
    if repaired is not None:
        try:
            return json.loads(repaired), True
        except ValueError:
            pass
    return None, False


def _first_bracket(text: str) -> int:
    match = re.search(r"[{\[]", text)
    return match.start() if match else -1


# A bare number token at the very END of a truncated text — possibly cut
# mid-digits, so never trustworthy. Includes partial exponents ("1.5e").
_TRAILING_NUMBER = re.compile(r"[+-]?(?:\d[\d.]*|\.\d+)(?:[eE][+-]?\d*)?$")


def _balanced_span(text: str):
    """The substring from position 0 to the close matching text[0], honouring
    strings and escapes; None when the value never closes (truncated)."""
    stack = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or char != stack[-1]:
                return None          # mismatched — not salvageable as a span
            stack.pop()
            if not stack:
                return text[:index + 1]
    return None


def _complete_truncated_json(text: str):
    """Close a truncated JSON prefix. Only completes and trims — never rewrites
    content, which is the deliberate divergence from v5's exhaustive repair."""
    stack = []
    in_string = False
    escaped = False
    string_open_at = -1
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            string_open_at = index
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or char != stack[-1]:
                return None
            stack.pop()

    if not stack and not in_string:
        return None                   # nothing to complete; parse failed anyway

    if in_string:
        # A string cut mid-flight is unfinishable content — drop it whole. What
        # remains ends just before the orphaned open quote.
        text = text[:string_open_at]

    # A NUMBER cut mid-digits would parse as a shorter, WRONG number
    # ("end_id": 1780 cut after 178 reads as 178) — fabrication through
    # the very gate this module exists to keep honest. Drop a trailing
    # bare-number token whole, the same invariant strings get. ONE-SHOT
    # and BEFORE the rstrip loop, deliberately: a number followed by a
    # comma/space/closer was completed by the model, and the loop's own
    # trimming must not re-expose it to this cut (code-review find,
    # 2026-08-26 — verified by execution before the fix).
    text = _TRAILING_NUMBER.sub("", text)

    # Trim dangling structure, repeatedly — a cut can leave several layers: a
    # `,`, an unvalued `"key":`, or a container opened with nothing yet in it
    # (whose closer must also come off the stack).
    while True:
        trimmed = text.rstrip()
        trimmed = re.sub(r',\s*$', "", trimmed)
        trimmed = re.sub(r'"(?:[^"\\]|\\.)*"\s*:\s*$', "", trimmed)
        trimmed = re.sub(r',\s*$', "", trimmed)
        if trimmed and trimmed[-1] in "{[" and stack:
            trimmed = trimmed[:-1]
            stack.pop()
        if trimmed == text:
            break
        text = trimmed

    if not text.strip():
        return None
    return text + "".join(reversed(stack))


# ── JSONL lines ───────────────────────────────────────────────────────────────

# A quote ends its JSON string only when the next non-space character is one
# of these (or the line ends). Anything else is a quotation mark the model
# wrote inside a value and forgot to escape — a property of JSON, not of any
# particular content.
_STRUCTURAL = ",:}]"


def escape_inner_quotes(line: str) -> str:
    """Escape unescaped double quotes sitting INSIDE a JSON string value.

    Only ever ADDS a backslash — never deletes, trims, completes or
    reorders — so a line damaged any other way comes back still
    unparseable and the caller's next rung (or its gap accounting) takes
    over. The one malformation measured in the wild (TDY/NNN, 3,462 real
    documents, 2026-08-25): a model transcribing on-screen text that
    literally contains quotation marks (`MIRAVOX RETURNS WITH "STILL GOLD"`)
    without escaping them. Recovered all of them byte-equal to the
    hand-escaped ground truth and left every already-valid line
    byte-identical.

    KNOWN RESIDUAL (accepted, 2026-08-26 review): an in-value quote that
    legitimately precedes a structural character — `... "STOP", said...`
    — reads as a string close, so this rung yields invalid JSON and the
    line falls to prefix salvage, which keeps an early-closed (i.e.
    content-trimmed) value and drops the tail, counted as an ordinary
    "prefix" repair. Undecidable at this layer without semantics; the
    real mitigations are upstream — the prompts' escape instruction and
    the risky-fields-last key order — and the tier count is the signal.
    """
    out: list = []
    i, n, in_string, escaped = 0, len(line), False, False
    while i < n:
        char = line[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                j = i + 1
                while j < n and line[j] in " \t":
                    j += 1
                if j < n and line[j] not in _STRUCTURAL:
                    out.append("\\")      # content quote, not a close
                else:
                    in_string = False
        elif char == '"':
            in_string = True
        out.append(char)
        i += 1
    return "".join(out)


def parse_jsonl_line(line: str, required_keys):
    """One JSONL document line, up a three-rung ladder.
    -> (obj, tier) with tier in {"strict", "escaped", "prefix"}, or
    (None, error) when nothing admissible survives.

        1. json.loads                     the answer for almost every line
        2. escape_inner_quotes -> strict  the measured malformation
        3. prefix salvage, gated          truncation or mid-line damage ->
                                          the honest prefix, if it still
                                          carries `required_keys`

    `required_keys` is the MINIMUM a caller's code cannot proceed without
    — join ids, never the full schema (user decision 2026-08-26: a
    recovered partial that merely lost optional metadata is kept, not
    discarded). Either a flat iterable of key names (all required), or an
    iterable of sets when one response mixes document shapes — admitted if
    ANY one set is fully present. An explicitly empty iterable means no
    gate. Rung 3 cannot fabricate: every value it returns parsed strictly
    from the model's own bytes, and an incomplete tail value is dropped
    whole rather than trimmed.
    """
    try:
        return json.loads(line), "strict"
    except ValueError as exc:
        first_error = str(exc)

    escaped = escape_inner_quotes(line)
    if escaped != line:
        try:
            return json.loads(escaped), "escaped"
        except ValueError:
            pass

    alternatives = _required_alternatives(required_keys)
    detail = "no salvageable prefix"
    for candidate in (line, escaped) if escaped != line else (line,):
        obj = _prefix_salvage(candidate)
        if not isinstance(obj, dict):
            continue
        if any(alt <= set(obj) for alt in alternatives):
            return obj, "prefix"
        lost = sorted(min(alternatives, key=lambda alt: len(alt - set(obj)))
                      - set(obj))
        detail = f"prefix lost required {', '.join(lost)}"
    return None, f"{first_error} (unrecovered: {detail})"


def _required_alternatives(required_keys) -> list:
    """A flat iterable of names is ONE conjunctive set; an iterable of
    sets is a disjunction of them. Empty -> no gate (must be stated with
    an explicit empty CONTAINER — a bare string is refused, because
    list("shot_id") explodes into seven single-character keys and "" is
    the ungated configuration this parameter exists to make unreachable
    by omission; code-review find, 2026-08-26)."""
    if isinstance(required_keys, str):
        raise TypeError(
            "required_keys must be an iterable of key names (or of sets), "
            f"never a bare string: {required_keys!r}")
    keys = list(required_keys)
    if not keys:
        return [set()]
    if all(isinstance(key, str) for key in keys):
        return [set(keys)]
    return [set(alternative) for alternative in keys]


def _prefix_salvage(line: str):
    """The largest honestly-parsed prefix of one damaged line, or None.

    Truncation is `parse_partial_json`'s existing job. For a line that is
    balanced but damaged mid-flight (partial-json's insight: stop at the
    damage and keep what scanned), the strict parser itself names the
    first offending offset — cut there and complete the prefix.
    """
    value, _ = parse_partial_json(line)
    if isinstance(value, dict):
        return value
    try:
        json.loads(line)
    except json.JSONDecodeError as exc:
        if exc.pos > 0:
            value, _ = parse_partial_json(line[:exc.pos])
            if isinstance(value, dict):
                return value
    except ValueError:
        pass
    return None


# ── XML ───────────────────────────────────────────────────────────────────────

_XML_TOKEN = re.compile(r"<[^>]*>|[^<]+")
_XML_OPEN = re.compile(r"<([A-Za-z_][\w.-]*)(\s[^>]*)?>$")
_XML_CLOSE = re.compile(r"</([A-Za-z_][\w.-]*)\s*>$")
_XML_SELF = re.compile(r"<[^>]*/>$|<[?!][^>]*>$")


def parse_partial_xml(text: str):
    """`(xml_string, was_repaired)` — the largest well-formed XML fragment of a
    model answer as a STRING, or `(None, False)`.

    Same salvage classes as the JSON half: prose/fence wrapping (not repair),
    trailing garbage after the root closes, and truncation — a cut mid-tag or
    mid-element is trimmed back to the last complete node and every open
    element is closed in stack order. Attribute values are assumed quoted with
    double or single quotes; a `<` inside one is the caller's malformed prompt,
    not this function's problem.
    """
    if not text or not isinstance(text, str):
        return None, False

    start = text.find("<")
    if start < 0:
        return None, False
    candidate = text[start:]

    stack: list[str] = []
    end_of_last_complete = 0          # end offset of the last whole token
    root_closed_at = -1
    seen_element = False

    pos = 0
    for match in _XML_TOKEN.finditer(candidate):
        if match.start() != pos:      # a bare `<` swallowed by a partial tag
            break
        token = match.group(0)
        pos = match.end()
        if token.startswith("<"):
            if not token.endswith(">"):
                break                 # cut mid-tag
            if _XML_SELF.match(token):
                seen_element = True
            elif token.startswith("</"):
                closed = _XML_CLOSE.match(token)
                if not closed or not stack or stack[-1] != closed.group(1):
                    break             # mismatched close — stop at the damage
                stack.pop()
                if not stack:
                    root_closed_at = pos
                    end_of_last_complete = pos
                    break             # root complete; the rest is trailing prose
            else:
                opened = _XML_OPEN.match(token)
                if not opened:
                    break
                stack.append(opened.group(1))
                seen_element = True
        end_of_last_complete = pos

    if not seen_element:
        return None, False

    if root_closed_at > 0 and not stack:
        fragment = candidate[:root_closed_at]
        # Wrapping/trailing prose alone is not repair, same rule as JSON.
        return fragment, False

    fragment = candidate[:end_of_last_complete].rstrip()
    if not fragment:
        return None, False
    closers = "".join(f"</{name}>" for name in reversed(stack))
    return fragment + closers, True
