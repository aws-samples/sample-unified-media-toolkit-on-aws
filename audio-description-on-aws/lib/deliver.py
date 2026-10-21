"""
Delivery renderers, ALL PURE (strings out): SRT/VTT, DAPT (W3C Dubbing
and Audio description Profile of TTML2 — the interchange format
professional post tooling ingests; JSONL stays the pipeline-canonical
script, user decision 2026-08-31), and review.html. The MIX is not
rendered here — no audio is opened anywhere in this workflow (settled
2026-08-31): placed.json carries each take's render instruction (tempo +
the AF chain), and the mix is a media-processing job once that stack
grows the render operation.
"""
from __future__ import annotations

import json
import os
# html.escape rather than xml.sax.saxutils: identical for &<> and it
# ALSO escapes quotes, which two call sites interpolate into XML
# attribute values; no xml import means no XXE surface at all.
from html import escape


def _ts(sec: float, sep: str = ".") -> str:
    ms = round(sec * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def subs(placed: list, cues: dict) -> tuple[str, str]:
    """(srt, vtt) for the placed cues, in mix order."""
    srt, vtt = [], ["WEBVTT", ""]
    for n, p in enumerate(sorted(placed, key=lambda p: p["at"]), 1):
        text = cues[str(p["window_id"])].strip()
        a, b = p["at"], p["at"] + p["dur"]
        srt += [str(n), f"{_ts(a, ',')} --> {_ts(b, ',')}",
                f"[AD] {text}", ""]
        vtt += [f"{_ts(a)} --> {_ts(b)}", f"[AD] {text}", ""]
    return "\n".join(srt), "\n".join(vtt)


def dapt(placed: list, cues: dict, title: str, lang: str = "en") -> str:
    """Minimal conforming DAPT document: one Script Event per placed cue
    with its timed narration text. Deliberately the smallest useful
    subset — audio embedding/gain and workflow metadata are downstream
    tooling's business."""
    events = []
    for n, p in enumerate(sorted(placed, key=lambda p: p["at"]), 1):
        text = escape(cues[str(p["window_id"])].strip())
        events.append(
            f'      <div xml:id="e{n}" begin="{_ts(p["at"])}" '
            f'end="{_ts(p["at"] + p["dur"])}" '
            f'daptm:eventType="audioDescription">\n'
            f'        <p><span daptm:textType="audioDescription">'
            f'{text}</span></p>\n'
            f'      </div>')
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<tt xmlns="http://www.w3.org/ns/ttml"\n'
        '    xmlns:ttp="http://www.w3.org/ns/ttml#parameter"\n'
        '    xmlns:daptm="http://www.w3.org/ns/ttml/profile/dapt#metadata"\n'
        f'    xml:lang="{escape(lang)}"\n'
        '    ttp:contentProfiles="http://www.w3.org/ns/ttml/profile/'
        'dapt1.0/content"\n'
        '    daptm:scriptRepresents="visualNonText"\n'
        '    daptm:scriptType="asRecordedScript"\n'
        f'    daptm:langSrc="{escape(lang)}">\n'
        '  <head>\n'
        f'    <metadata><ttm:title xmlns:ttm="http://www.w3.org/ns/ttml'
        f'#metadata">{escape(title)}</ttm:title></metadata>\n'
        '  </head>\n'
        '  <body>\n'
        + "\n".join(events) + "\n"
        '  </body>\n'
        '</tt>\n')


def review_html(title: str, video_name: str, ledger: list,
                ws: dict) -> tuple[str, int]:
    """review.html from the LEDGER, not the placed list: the rows the
    reviewer most needs are the ones that did NOT ship. Returns
    (html, shipped_count)."""
    def mts(sec):
        m, s = divmod(int(sec), 60)
        return f"{m:02d}:{s:02d}"

    rows = [{**r, "n": i, "ts": mts(r["at"]),
             "entry": bool(ws.get(r.get("window_id"), {})
                           .get("scene_entry"))}
            for i, r in enumerate(ledger, 1)]
    shipped = sum(1 for r in rows if r["state"] == "placed")
    tpl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "review_template.html")
    tpl = open(tpl_path, encoding="utf-8").read()
    # "</" must not survive inside the inline <script> block: a cue text
    # carrying a literal "</script>" would terminate the element before any
    # JS-level escaping runs. "<\/" is identical to "</" after JS string
    # parsing, so benign data is unchanged.
    cues_js = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    html = (tpl.replace("__TITLE__", escape(title))
               .replace("__VIDEO__", escape(video_name))
               .replace("__NCUES__", str(shipped))
               .replace("__CUES__", cues_js))
    return html, shipped
