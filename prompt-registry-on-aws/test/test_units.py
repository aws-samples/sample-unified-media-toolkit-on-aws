"""
Unit tests — prompt-registry-on-aws. No AWS calls, no moto; the S3 client in
the history tests is a hand-mock.

The through-line: `prompt_render.py`'s output feeds `workflow_id` (the hash of
rendered bytes), so the failures this suite exists to catch are IDENTITY
failures — two copies of the module disagreeing about rendering is two repos
silently re-running the same request at fresh prefixes. Hence golden vectors
(fixed artifact + values → fixed BYTES, pinned by digest) rather than
property tests alone: determinism and key-order independence hold equally well
for a *wrong* rendering.

The 18 real v5 builtins under prompts/ are the grammar corpus: every one must
parse, and the parse of a known one is pinned field-by-field. That keeps the
port honest against the reference implementation rather than against itself.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import unittest

def _read(path, **kw):
    """_read() without the ResourceWarning — tests leave no
    unclosed handles (public build logs read clean)."""
    with open(path, **kw) as f:
        return f.read()


def _load_json(path):
    with open(path) as f:
        return json.load(f)


from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# gateway_tool/ itself goes on the path — the zip is flat, so at runtime these
# are TOP-LEVEL modules (a family-wide flat-zip packaging trap).
sys.path.insert(0, os.path.join(ROOT, "gateway_tool"))

import prompt_md                                     # noqa: E402
import prompt_render                                 # noqa: E402
import workflow_history                              # noqa: E402
from prompt_md import ParseError, parse_prompt_md    # noqa: E402
from prompt_render import RenderError, render        # noqa: E402


def _md(header: str, system: str = "Be terse.",
        user: str = "Summarize: {{transcript}}") -> str:
    return (f"{header}\n# System Prompt\n\n{system}\n\n"
            f"# User Prompt\n\n{user}\n")


VALID_HEADER = "<!-- Version: default -->\n<!-- Comment: a test prompt -->"


def load_corpus() -> dict:
    """(category, version) -> source text, for every flattened builtin.
    Module-level so the corpus tests and the vocabulary lint share one loader
    instead of one class reaching into another's setUpClass state."""
    sources = {}
    base = os.path.join(ROOT, "prompts")
    for category in sorted(os.listdir(base)):
        cat_dir = os.path.join(base, category)
        if not os.path.isdir(cat_dir):
            continue
        for name in sorted(os.listdir(cat_dir)):
            if name.endswith(".md"):
                with open(os.path.join(cat_dir, name)) as f:
                    sources[(category, name[:-3])] = f.read()
    return sources



# Silence console log HANDLERS for the whole suite: the failure-path
# tests exercise refusal/retry/fallback branches whose designed
# behavior includes ERROR/WARNING logs, and those lines made build
# logs read as if the build were failing. Handlers only — record flow
# is untouched, so assertLogs() still sees everything.
import logging as _logging


def _silence_console_logging():
    _logging.getLogger().handlers = [_logging.NullHandler()]
    for _name in list(_logging.root.manager.loggerDict):
        _lg = _logging.getLogger(_name)
        if _lg.handlers:
            _lg.handlers = [_logging.NullHandler()]


_silence_console_logging()


class TestGrammar(unittest.TestCase):
    """The token grammar, ported from templateVariables.js — every v5 form."""

    def parse(self, user):
        return {v["token"]: v for v in prompt_md.parse_variables(user)}

    def test_bare_token(self):
        v = self.parse("{{transcript}}")["{{transcript}}"]
        self.assertEqual(v, {"token": "{{transcript}}",  # nosec B105
                             "root": "transcript", "leafs": []})

    def test_hierarchical_single_leaf(self):
        v = self.parse("{{scene:transcript}}")["{{scene:transcript}}"]
        self.assertEqual(v["root"], "scene")
        self.assertEqual(v["leafs"], [{"root": "transcript", "leafs": []}])

    def test_hierarchical_plus_leafs(self):
        v = self.parse("{{scene:transcript+sound_events}}")[
            "{{scene:transcript+sound_events}}"]
        self.assertEqual(v["root"], "scene")
        self.assertEqual(v["leafs"], [{"root": "transcript", "leafs": []},
                                      {"root": "sound_events", "leafs": []}])

    def test_nested_hierarchy(self):
        v = self.parse("{{a:b:c}}")["{{a:b:c}}"]
        self.assertEqual(
            v, {"token": "{{a:b:c}}", "root": "a",  # nosec B105
                "leafs": [{"root": "b",
                           "leafs": [{"root": "c", "leafs": []}]}]})

    def test_legacy_plus_syntax(self):
        """v5 back-compat: {{a+b+c}} with no ':' — first part is the root."""
        v = self.parse("{{scene+transcript+sound_events}}")[
            "{{scene+transcript+sound_events}}"]
        self.assertEqual(v["root"], "scene")
        self.assertEqual(v["leafs"], [{"root": "transcript", "leafs": []},
                                      {"root": "sound_events", "leafs": []}])

    def test_spaces_strip_for_parsing_but_the_raw_token_is_kept(self):
        """v5 dedups on the RAW inner text, so two spellings of one variable
        are two declared tokens — render substitutes by raw string and must
        find each. Preserved, not 'fixed': normalizing here would orphan the
        spaced spelling in the template."""
        parsed = self.parse("{{ scene : transcript }} and {{scene:transcript}}")
        self.assertEqual(len(parsed), 2)
        for v in parsed.values():
            self.assertEqual(v["root"], "scene")

    def test_duplicate_tokens_declare_once(self):
        self.assertEqual(len(self.parse("{{a}} {{a}} {{a}}")), 1)

    def test_empty_token_is_a_parse_error(self):
        with self.assertRaises(ParseError):
            self.parse("{{ }}")


class TestParsePromptMd(unittest.TestCase):
    def test_the_full_parse_shape(self):
        artifact = parse_prompt_md(_md(VALID_HEADER))
        self.assertEqual(artifact["version"], "default")
        self.assertEqual(artifact["comment"], "a test prompt")
        self.assertEqual(artifact["system"], "Be terse.")
        self.assertEqual(artifact["user_template"],
                         "Summarize: {{transcript}}")
        self.assertEqual(artifact["variables"][0]["token"], "{{transcript}}")
        self.assertEqual(len(artifact["source_md5"]), 32)

    def test_sections_split_case_insensitively(self):
        text = ("<!-- Version: v1 -->\n# SYSTEM PROMPT\nsys\n"
                "# user PROMPT\nuser {{x}}\n")
        artifact = parse_prompt_md(text)
        self.assertEqual(artifact["system"], "sys")
        self.assertEqual(artifact["user_template"], "user {{x}}")

    def test_tokens_are_collected_from_system_too(self):
        """v5 scans the whole body after '# System Prompt'; real builtins use
        variables in both sections."""
        artifact = parse_prompt_md(_md(
            VALID_HEADER, system="Video runs {{total_duration}}s.",
            user="Go: {{transcript}}"))
        tokens = {v["token"] for v in artifact["variables"]}
        self.assertEqual(tokens, {"{{total_duration}}", "{{transcript}}"})

    def test_header_values_may_contain_colons(self):
        artifact = parse_prompt_md(_md(
            "<!-- Version: default -->\n<!-- Comment: a:b:c -->"))
        self.assertEqual(artifact["comment"], "a:b:c")

    def test_unknown_headers_ride_in_aux_and_reserved_keys_are_dropped(self):
        artifact = parse_prompt_md(_md(
            "<!-- Version: v1 -->\n"
            "<!-- SupportedVariables: transcript -->\n"
            "<!-- md5: feedface -->"))
        self.assertEqual(artifact["aux"], {"supportedvariables": "transcript"})

    def test_the_reject_matrix(self):
        cases = {
            "no version": _md("<!-- Comment: x -->"),
            "no system section": "<!-- Version: v -->\n# User Prompt\nhi",
            "no user section": "<!-- Version: v -->\n# System Prompt\nhi",
            "empty": "",
        }
        for name, text in cases.items():
            with self.subTest(name):
                with self.assertRaises(ParseError):
                    parse_prompt_md(text)


class TestRealBuiltinsCorpus(unittest.TestCase):
    """Every flattened v5 builtin under prompts/ must parse — the port is
    honest against the reference corpus, not just against its own fixtures."""

    @classmethod
    def setUpClass(cls):
        cls.sources = load_corpus()

    def test_the_corpus_is_the_full_flattened_set(self):
        # 18 ported 2026-08-15, then 4 example variants culled in review
        # (example_compact, example_sop, example_cov, example_segment) —
        # 14 files across the same 12 categories. Then `shot-metadata-
        # enrichment/default` authored NEW here 2026-08-18 (not a v5 port —
        # the first prompt this registry originates), making 15 across 13.
        # `shot-metadata-enrichment/terse` (its losing A/B variant: 8.9%
        # fewer prompt tokens but leaks filmed-environment signage into the
        # post-production-only texts_on_screen field) was kept briefly as
        # recorded evidence, then DELETED 2026-08-19 (user decision) before
        # the seeder could land it as stage: prod — the A/B result lives in
        # shot-metadata-enrichment's own design records, which is where
        # evidence belongs; prompts/ is what ships. Then the two `default_news`
        # variants (conversation-analysis, shot-metadata-enrichment)
        # authored 2026-08-23 for the news-programme workstream — named
        # versions of existing categories, selected per request by the
        # caller workflows' prompt_version knob: 17 across the same 13.
        # Then the three frame-inspection categories (credits-read,
        # title-read, claim-verify) authored 2026-08-29 for frame-
        # inspection's witness workflow: 20 across 16. Then the five
        # audio-description categories (ad-writer, ad-critic, ad-repair,
        # ad-name-map, ad-watch-notes) authored 2026-08-31 from
        # audio-description's corpus-validated runner: 25
        # across 21. Then the v5-era orphans DELETED 2026-08-31 (user
        # decision; grep-verified zero consumers): the nine segment/
        # image menu prompts, program-structure-analysis (PSA binds no
        # prompts) and scene-metadata-enrichment (enrichment binds only
        # shot-metadata-enrichment — an orphan despite the name):
        # 12 across 10. The seeded S3 pairs for deleted categories
        # linger (default versions are protected); manual sweep only.
        # 12/10 pre-2026-09-08; +3 with the cast-* migration; +2 with
        # the artwork-selection pair (artwork-rerank / artwork-brief,
        # 2026-09-16)
        self.assertEqual(len(self.sources), 17)
        self.assertEqual(len({c for c, _ in self.sources}), 15)

    def test_every_builtin_parses_and_declares_its_filename_version(self):
        for (category, version), text in self.sources.items():
            with self.subTest(f"{category}/{version}"):
                artifact = parse_prompt_md(text)
                # v5 stores by `{version}.md` where version is the header
                # value — the flattened copy must agree with itself.
                self.assertEqual(artifact["version"], version)
                self.assertTrue(artifact["variables"],
                                "every builtin uses at least one variable")

    def test_a_known_builtin_parses_to_known_fields(self):
        """REBUILT 2026-08-17 for the conversation-analysis workflow: the
        four v5 context variables moved in-band into the transcript envelope
        (conversation-analysis's envelope design) — the rebuilt default
        declares exactly ONE variable."""
        artifact = parse_prompt_md(
            self.sources[("conversation-analysis", "default")])
        tokens = {v["token"] for v in artifact["variables"]}
        self.assertEqual(tokens, {"{{transcript}}"})
        # The new output contract is JSONL keyed by cue ids — pin the two
        # phrases the caller workflow's parser depends on.
        self.assertIn("start_id", artifact["system"])
        self.assertIn("one JSON object per line", artifact["system"])


class TestRender(unittest.TestCase):
    ARTIFACT = {
        "system": "You are terse. Asset runs {{total_duration}}s.",
        "user_template": ("From {{chunk_start_time}} to {{chunk_end_time}}:\n"
                          "{{transcript}}\nAgain: {{transcript}}"),
        "variables": [
            {"token": "{{total_duration}}", "root": "total_duration",  # nosec B105
             "leafs": []},
            {"token": "{{chunk_start_time}}", "root": "chunk_start_time",  # nosec B105
             "leafs": []},
            {"token": "{{chunk_end_time}}", "root": "chunk_end_time",  # nosec B105
             "leafs": []},
            {"token": "{{transcript}}", "root": "transcript", "leafs": []},  # nosec B105
        ],
    }
    VALUES = {"{{total_duration}}": 3670.5, "{{chunk_start_time}}": 0,
              "{{chunk_end_time}}": 300,
              "{{transcript}}": "He vanished."}

    def test_every_occurrence_substitutes_in_both_sections(self):
        out = render(self.ARTIFACT, self.VALUES)
        self.assertEqual(out["system"],
                         "You are terse. Asset runs 3670.5s.")
        text = out["messages"][0]["content"][0]["text"]
        self.assertEqual(text,
                         "From 0 to 300:\nHe vanished.\nAgain: He vanished.")

    def test_missing_and_unknown_values_raise_and_name_every_offender(self):
        with self.assertRaises(RenderError) as ctx:
            render(self.ARTIFACT, {"{{transcript}}": "x",
                                   "{{typo}}": "y"})
        message = str(ctx.exception)
        self.assertIn("{{total_duration}}", message)
        self.assertIn("{{typo}}", message)

    def test_no_defaulting_ever(self):
        """v5 substituted the literal 'n/a' for undefined — the exact silent
        behavior change this module exists to kill."""
        values = dict(self.VALUES, **{"{{transcript}}": None})
        with self.assertRaises(RenderError):
            render(self.ARTIFACT, values)
        for bad in (True, object()):
            with self.subTest(type(bad).__name__):
                with self.assertRaises(RenderError):
                    render(self.ARTIFACT,
                           dict(self.VALUES, **{"{{transcript}}": bad}))

    def test_dicts_and_scalar_lists_render_as_compact_json(self):
        artifact = {"system": "s", "user_template": "{{x}}|{{y}}",
                    "variables": [
                        {"token": "{{x}}", "root": "x", "leafs": []},  # nosec B105
                        {"token": "{{y}}", "root": "y", "leafs": []}]}  # nosec B105
        out = render(artifact, {"{{x}}": {"b": 1, "a": [2, None]},
                                "{{y}}": ["p", 3]})
        self.assertEqual(out["messages"][0]["content"][0]["text"],
                         '{"b":1,"a":[2,null]}|["p",3]')

    def test_content_blocks_splice_into_the_user_message(self):
        artifact = {"system": "s", "user_template":
                    "Look:\n{{frames}}\nDescribe.",
                    "variables": [{"token": "{{frames}}", "root": "frames",  # nosec B105
                                   "leafs": []}]}
        blocks = [{"image": {"format": "png",
                             "source": {"bytes": "AAAA"}}},
                  {"image": {"format": "png",
                             "source": {"bytes": "BBBB"}}}]
        out = render(artifact, {"{{frames}}": blocks})
        content = out["messages"][0]["content"]
        self.assertEqual([list(b)[0] for b in content],
                         ["text", "image", "image", "text"])
        self.assertEqual(content[0], {"text": "Look:\n"})
        self.assertEqual(content[3], {"text": "\nDescribe."})
        # Blocks pass through by identity of content — never copied-and-edited
        self.assertEqual(content[1], blocks[0])

    def test_a_list_of_plain_data_dicts_is_json_not_blocks(self):
        """The hazard the strict shape test exists for (settled 2026-08-15:
        "blocks means image blocks; json means nested data"): a json-kind
        value that HAPPENS to be a list of dicts — scene_descriptions is one
        dict per scene — must stringify, never splice. Under the old
        any-list-of-dicts rule this spliced scene payloads into message
        content as if they were images."""
        artifact = {"system": "s", "user_template": "Scenes: {{scenes}}",
                    "variables": [{"token": "{{scenes}}", "root": "scenes",  # nosec B105
                                   "leafs": []}]}
        scenes = [{"scene_id": 0, "transcript": "hi",
                   "relevant_tags": ["a"]},
                  {"scene_id": 1, "transcript": "bye"}]
        out = render(artifact, {"{{scenes}}": scenes})
        content = out["messages"][0]["content"]
        self.assertEqual(len(content), 1)
        self.assertIn('"scene_id":0', content[0]["text"])
        # Multi-key and unknown-key dicts are data too, even in a list of one:
        for not_blocks in ([{"text": "x", "extra": 1}],
                           [{"toolUse": {}}],
                           [{"image": {}}, {"scene_id": 2}]):
            with self.subTest(str(not_blocks)[:40]):
                out = render(artifact, {"{{scenes}}": not_blocks})
                self.assertEqual(len(out["messages"][0]["content"]), 1)
                self.assertIn("text", out["messages"][0]["content"][0])

    def test_blocks_in_the_system_prompt_raise(self):
        artifact = {"system": "See {{frames}}", "user_template": "u",
                    "variables": [{"token": "{{frames}}", "root": "frames",  # nosec B105
                                   "leafs": []}]}
        with self.assertRaises(RenderError):
            render(artifact, {"{{frames}}": [{"image": {}}]})

    def test_token_at_the_template_edge_produces_no_empty_text_block(self):
        artifact = {"system": "s", "user_template": "{{frames}}",
                    "variables": [{"token": "{{frames}}", "root": "frames",  # nosec B105
                                   "leafs": []}]}
        out = render(artifact, {"{{frames}}": [{"image": {"k": 1}}]})
        self.assertEqual(out["messages"][0]["content"],
                         [{"image": {"k": 1}}])


class TestRenderGoldenVectors(unittest.TestCase):
    """
    Fixed artifact + values → fixed BYTES. These digests are the freeze: a
    failure here means either an unintended rendering change (fix the code)
    or an intended one — in which case update the vector AND bump every
    consumer's .version, because old and new code now disagree about the
    identity of the same request (micro-workflow-bundle/README.md §C2's golden-vector rule).
    """

    def digest(self, out):
        return hashlib.sha256(json.dumps(
            out, ensure_ascii=False, separators=(",", ":"),
            sort_keys=True).encode("utf-8")).hexdigest()

    def test_scalar_vector(self):
        out = render(TestRender.ARTIFACT, TestRender.VALUES)
        self.assertEqual(
            self.digest(out),
            "b89e6a190093fa668421415874d8c4d90b37daabe2f66d8237b782d9fb14f98e")

    def test_block_splice_vector(self):
        artifact = {"system": "Frames of {{asset}}.",
                    "user_template": "Between:\n{{frames}}\nname the cut.",
                    "variables": [
                        {"token": "{{asset}}", "root": "asset", "leafs": []},  # nosec B105
                        {"token": "{{frames}}", "root": "frames",  # nosec B105
                         "leafs": []}]}
        out = render(artifact, {
            "{{asset}}": "Meridian",
            "{{frames}}": [{"image": {"format": "jpeg",
                                      "source": {"bytes": "Zm9v"}}}]})
        self.assertEqual(
            self.digest(out),
            "d90d7fa3cf306bfc6b3d93611303e89b3f42455852fa8d166e27cc33924dfb6c")

    def test_unicode_stays_unescaped(self):
        """ensure_ascii=False is part of the freeze: an escaped rendering is
        different bytes, hence a different workflow_id, for every non-ASCII
        transcript on earth."""
        artifact = {"system": "s", "user_template": "{{t}}",
                    "variables": [{"token": "{{t}}", "root": "t",  # nosec B105
                                   "leafs": []}]}
        out = render(artifact, {"{{t}}": {"quote": "Küste — 海岸"}})
        self.assertIn("Küste — 海岸",
                      out["messages"][0]["content"][0]["text"])


class TestVocabulary(unittest.TestCase):
    """
    The reviewed vocabulary (vocabulary.json, settled 2026-08-15) is a LINT
    pinned in both directions: every name a builtin uses must be canonical or
    a listed alias (a novel coinage fails until the vocabulary is deliberately
    amended), and the vocabulary itself must stay internally coherent. It is
    NOT a runtime gate — the per-prompt contract is the artifact's own parsed
    variables, and put_prompt will WARN on novel names, never refuse.
    """

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, "vocabulary.json")) as f:
            cls.vocab = json.load(f)
        cls.canonical = set(cls.vocab["variables"])
        cls.aliases = {}
        for name, entry in cls.vocab["variables"].items():
            for alias in entry.get("aliases", []):
                cls.aliases[alias] = name

    @staticmethod
    def _names(variable):
        """Every root/leaf name a declared variable touches, recursively."""
        out = {variable["root"]}
        for leaf in variable.get("leafs", []):
            out |= TestVocabulary._names(leaf)
        return out

    def test_vocabulary_is_internally_coherent(self):
        self.assertEqual(self.vocab["vocabulary_version"], 1)
        collisions = set(self.aliases) & self.canonical
        self.assertFalse(collisions,
                         f"aliases colliding with canonical names: {collisions}")
        seen = {}
        for alias, name in self.aliases.items():
            self.assertNotIn(alias, seen,
                             f"alias {alias!r} maps to both {seen.get(alias)} "
                             f"and {name}")
            seen[alias] = name
        kinds = {e["kind"] for e in self.vocab["variables"].values()}
        self.assertTrue(kinds <= {"text", "number", "json", "blocks"}, kinds)

    def test_every_builtin_name_is_canonical_or_a_listed_alias(self):
        """The coverage pin: run over the REAL corpus, so a new prompt (or an
        edit) coining a name arrives as a two-file diff — prompt + vocabulary
        — and gets reviewed as vocabulary, not smuggled in as spelling."""
        known = self.canonical | set(self.aliases)
        unknown = {}
        for (category, version), text in load_corpus().items():
            artifact = parse_prompt_md(text)
            for variable in artifact["variables"]:
                for name in self._names(variable):
                    if name not in known:
                        unknown.setdefault(name, []).append(
                            f"{category}/{version}")
        self.assertFalse(
            unknown,
            "names outside vocabulary.json (typo, or amend the vocabulary "
            f"deliberately): {json.dumps(unknown, indent=1)}")

    def test_workflow_provided_variables_are_annotated(self):
        """provided_by: workflow (2026-08-17, conversation-analysis's
        values-catalog design): these tokens are injected programmatically by the
        calling llm-prompt-workflow — the annotation is how describe_tool
        tells prompt authors 'declare, never compute'. Pinned in both
        directions: exactly this set, and no other value of provided_by
        sneaks in."""
        provided = {name for name, e in self.vocab["variables"].items()
                    if e.get("provided_by")}
        self.assertEqual(provided, {
            "transcript", "chunk_start_time", "chunk_end_time",
            "total_duration", "transcript_duration",
            # frame-inspection-on-aws (2026-08-29): span framing + the
            # claim-verify question, both injected by the workflow.
            "span_context", "question"})
        for name in provided:
            entry = self.vocab["variables"][name]
            self.assertEqual(entry["provided_by"], "workflow", name)
            self.assertIn("WORKFLOW-PROVIDED", entry["description"], name)

    def test_the_consolidations_are_encoded(self):
        """The 2026-08-15 review's three consolidation decisions, pinned."""
        self.assertEqual(self.aliases.get("latitude"), "gps_location")
        self.assertEqual(self.aliases.get("longitude"), "gps_location")
        # frame_images = SINGLE full frames; grid_images = TILED multi-frame
        # grids — different contracts (corrected in review round 2, from v5's
        # own helper docs: "Grid tiling is NOT used here"). grid_images is
        # canonical, never an alias.
        for legacy in ("scene_frame_images", "shot_frame_images"):
            self.assertEqual(self.aliases.get(legacy), "frame_images")
        self.assertIn("grid_images", self.canonical)
        self.assertNotIn("grid_images", self.aliases)
        # Roots follow v5's value names: the composite values are LISTS.
        self.assertEqual(self.aliases.get("scene_description"),
                         "scene_descriptions")
        self.assertEqual(self.aliases.get("segments"), "segment_descriptions")
        # The singular/plural drift, resolved toward the plural (or the more
        # precise) spelling:
        # transcript and dialogue(s) are ONE concept (round 3): the same
        # per-scene spoken text, spelled three ways across the composites.
        # program_name(+potential_) → program_names (round 4): same data,
        # 1..N names comma-separated; the known-vs-candidate framing lives in
        # the prompt's prose. No v5 prompt used both, so no collision.
        for legacy, canonical in (("program_name", "program_names"),
                                  ("potential_program_names",
                                   "program_names"),
                                  ("dialogue", "transcript"),
                                  ("dialogues", "transcript"),
                                  ("camera_shot", "camera_shot_type"),
                                  ("sound_event", "sound_events"),
                                  ("text_on_screen", "texts_on_screen"),
                                  # round 8: canonical follows the DATA field
                                  # name (v5 emits {relevant_tags: ...} into
                                  # every scene payload) — same precedent as
                                  # scene_descriptions/segment_descriptions.
                                  ("tag", "relevant_tags"),
                                  ("tags", "relevant_tags")):
            self.assertEqual(self.aliases.get(legacy), canonical)


class TestClassify(unittest.TestCase):
    """The put_prompt warning brain: aliases are attested legacy spellings
    ONLY (settled 2026-08-15 round 5 — no speculative aliases), and every
    guessable variant is handled by suggestion instead, so `scene`/`scenes`/
    `segment` never need pre-registering and typos get a 'did you mean'."""

    @classmethod
    def setUpClass(cls):
        import vocabulary
        cls.v = vocabulary
        cls.vocab = vocabulary.load_vocabulary()

    def check(self, name):
        return self.v.classify(name, self.vocab)

    def test_the_three_states(self):
        self.assertEqual(self.check("transcript"),
                         ("canonical", "transcript"))
        self.assertEqual(self.check("dialogue"), ("alias", "transcript"))
        self.assertEqual(self.check("segments"),
                         ("alias", "segment_descriptions"))

    def test_guessable_variants_suggest_without_being_aliases(self):
        for probe, want in (("scene", "scene_descriptions"),
                            ("scenes", "scene_descriptions"),
                            ("segment", "segment_descriptions"),
                            ("transcirpt", "transcript"),
                            ("program_nam", "program_names")):
            with self.subTest(probe):
                kind, suggestion = self.check(probe)
                self.assertEqual(kind, "novel")
                self.assertEqual(suggestion, want)
                self.assertNotIn(probe, self.vocab["aliases"],
                                 "guessable variants must stay OUT of "
                                 "aliases — suggestion handles them")

    def test_a_genuinely_novel_name_suggests_nothing(self):
        kind, suggestion = self.check("weather_report")
        self.assertEqual((kind, suggestion), ("novel", None))


class TestWorkflowHistory(unittest.TestCase):
    SUBMIT = {"workflow_id": "wid-1", "job_id": "arn:exec",
              "output_s3_prefix": "s3://b/micro-workflow/bedrock-inference/wid-1"}

    def entry(self, **kw):
        s3 = mock.MagicMock()
        submit = kw.pop("submit", None) or self.SUBMIT
        args = dict(scope="asset.meridian", category="conversation-analysis",
                    prompt_version="default", model_chain=["m1", "m2"],
                    submitted_at_ms=1786500000000)
        args.update(kw)
        key = workflow_history.write_history_entry(s3, "bkt", submit, **args)
        return s3, key

    def test_key_layout_and_pointer_shape(self):
        s3, key = self.entry()
        self.assertEqual(
            key, "micro-workflow/histories/asset.meridian/"
                 "conversation-analysis/default/1786500000000-wid-1.json")
        body = json.loads(s3.put_object.call_args.kwargs["Body"])
        self.assertEqual(body, {
            "workflow_id": "wid-1", "job_id": "arn:exec",
            "output_s3_prefix":
                "s3://b/micro-workflow/bedrock-inference/wid-1",
            "scope": "asset.meridian",
            "prompt": {"category": "conversation-analysis",
                       "version": "default"},
            "model_chain": ["m1", "m2"], "idempotency_token": None,  # nosec B105
            "submitted_at": 1786500000000})
        # A pointer, not a copy: no outcome, no usage, no records — the
        # reader follows output_s3_prefix/summary.json, which is authoritative.
        for forbidden in ("status", "records_succeeded", "summary",
                          "estimated_cost"):
            self.assertNotIn(forbidden, body)

    def test_an_absorbed_resubmit_writes_nothing(self):
        """The submit response's `message` is the absorption signal — writing
        anyway manufactures a phantom second run pointing at one execution."""
        s3 = mock.MagicMock()
        key = workflow_history.write_history_entry(
            s3, "bkt", dict(self.SUBMIT, message="identical request already "
                            "submitted — this call did not resubmit; ..."),
            scope="s", category="c", prompt_version="v",
            model_chain=[], submitted_at_ms=1)
        self.assertIsNone(key)
        s3.put_object.assert_not_called()

    def test_scope_charset_is_the_execution_name_charset(self):
        for bad in ("has space", "slash/", "", "x" * 65, "ünïcode"):
            with self.subTest(bad):
                with self.assertRaises(workflow_history.HistoryError):
                    self.entry(scope=bad)

    def test_a_submit_without_a_workflow_id_is_refused(self):
        with self.assertRaises(workflow_history.HistoryError):
            self.entry(submit={"job_id": "arn:exec"})



# ═══════════════════════════════════════════════════════════════════════════
# The service: seven tools, the seeder, the stack, the build script
# ═══════════════════════════════════════════════════════════════════════════

class _FakeS3:
    """A dict-backed S3: put/get/delete/list_objects_v2 with Prefix+Delimiter
    and the conditional-put half of write-if-absent. Unmodelled arguments
    raise rather than silently succeeding."""

    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body, ContentType=None, **kw):
        if kw.pop("IfNoneMatch", None) == "*" and (Bucket, Key) in self.objects:
            import botocore.exceptions
            raise botocore.exceptions.ClientError(
                {"Error": {"Code": "PreconditionFailed",
                           "Message": "exists"}}, "PutObject")
        if kw:
            raise AssertionError(f"unmodelled put_object args: {kw}")
        self.objects[(Bucket, Key)] = Body if isinstance(Body, bytes) \
            else Body.encode()
        return {}

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            import botocore.exceptions
            raise botocore.exceptions.ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": Key}}, "GetObject")
        import io
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def delete_object(self, Bucket, Key):
        self.objects.pop((Bucket, Key), None)
        return {}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        fake = self

        class _P:
            def paginate(self, Bucket, Prefix="", Delimiter=None):
                keys = sorted(k for (b, k) in fake.objects
                              if b == Bucket and k.startswith(Prefix))
                if Delimiter:
                    prefixes = []
                    seen = set()
                    for k in keys:
                        rest = k[len(Prefix):]
                        if Delimiter in rest:
                            p = Prefix + rest.split(Delimiter)[0] + Delimiter
                            if p not in seen:
                                seen.add(p)
                                prefixes.append({"Prefix": p})
                    yield {"CommonPrefixes": prefixes, "Contents": []}
                else:
                    yield {"Contents": [{"Key": k} for k in keys]}
        return _P()


class _ToolContext:
    def __init__(self, tool):
        custom = {"bedrockAgentCoreToolName": f"prompt-registry___{tool}"}
        self.client_context = type("C", (), {"custom": custom})()


def _valid_md(version="v1", user="Summarize {{transcript}} for {{tags}}."):
    return (f"<!-- Version: {version} -->\n<!-- Comment: t -->\n"
            f"# System Prompt\nBe terse.\n# User Prompt\n{user}\n")


class _ServiceCase(unittest.TestCase):
    """Imports the dispatcher the way its flat zip resolves it and gives each
    test a fresh fake S3 + PROXY_BUCKET."""

    @classmethod
    def setUpClass(cls):
        import lambda_function
        cls.gw = lambda_function

    def setUp(self):
        self.s3 = _FakeS3()
        patches = [
            mock.patch.object(self.gw, "_s3", self.s3),
            mock.patch.object(self.gw, "PROXY_BUCKET", "test-proxy"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def call(self, tool, **event):
        return self.gw.lambda_handler(event, _ToolContext(tool))


class TestPutGetTools(_ServiceCase):
    def test_put_then_get_roundtrip(self):
        out = self.call("put_prompt", category="my-cat", text=_valid_md())
        self.assertNotIn("error", out)
        self.assertEqual((out["category"], out["version"], out["stage"],
                          out["origin"]), ("my-cat", "v1", "dev", "user"))
        self.assertIn("_promptregistry/my-cat/v1/source.md",
                      out["source_s3uri"])
        got = self.call("get_prompt", category="my-cat", version="v1")
        self.assertEqual(got["user_template"],
                         "Summarize {{transcript}} for {{tags}}.")
        self.assertEqual(got["stage"], "dev")
        self.assertFalse(got["deprecated"])
        # the pair exists
        self.assertIn(("test-proxy", "_promptregistry/my-cat/v1/source.md"),
                      self.s3.objects)

    def test_put_warnings_carry_the_vocabulary_lint(self):
        out = self.call("put_prompt", category="c",
                        text=_valid_md(user="Use {{tags}} and {{scene}}."))
        joined = " | ".join(out["warnings"])
        self.assertIn("'tags': legacy spelling — canonical is 'relevant_tags'",
                      joined)
        self.assertIn("did you mean 'scene_descriptions'", joined)

    def test_versions_are_immutable(self):
        self.call("put_prompt", category="c", text=_valid_md())
        out = self.call("put_prompt", category="c", text=_valid_md())
        self.assertEqual(out["errorCode"], "VERSION_EXISTS")

    def test_default_is_born_prod(self):
        out = self.call("put_prompt", category="c",
                        text=_valid_md(version="default"), stage="dev")
        self.assertEqual(out["stage"], "prod")
        self.assertIn("always prod", out["message"])

    def test_get_without_version_resolves_to_default(self):
        self.call("put_prompt", category="c",
                  text=_valid_md(version="default"))
        got = self.call("get_prompt", category="c")
        self.assertEqual(got["version"], "default")

    def test_a_named_missing_version_never_falls_back(self):
        """v5's getBuiltinMarkdown silently substituted default.md — the
        invisible-behavior-change class this design kills."""
        self.call("put_prompt", category="c",
                  text=_valid_md(version="default"))
        out = self.call("get_prompt", category="c", version="nope")
        self.assertEqual(out["errorCode"], "PROMPT_NOT_FOUND")

    def test_the_reject_matrix(self):
        big = _valid_md(user="x" * 1_100_000)
        cases = {
            "INVALID_PROMPT": dict(category="c", text="no sections"),
            "PROMPT_TOO_LARGE": dict(category="c", text=big),
            "INVALID_PARAMETER": dict(category="c", text=_valid_md(),
                                      stage="production"),
        }
        for code, event in cases.items():
            with self.subTest(code):
                self.assertEqual(self.call("put_prompt", **event)["errorCode"],
                                 code)
        self.assertEqual(
            self.call("put_prompt", category="bad cat",
                      text=_valid_md())["errorCode"], "INVALID_PARAMETER")

    def test_a_stray_token_is_ignored_never_rejected(self):
        """CONTRACT §11 — no tool here starts an execution, so both spellings
        warn-and-pass on every tool, submits included."""
        out = self.call("put_prompt", category="c", text=_valid_md(),
                        idempotency_token="tok", uuid="u")  # nosec B106
        self.assertNotIn("error", out)


class TestListAndLifecycle(_ServiceCase):
    def seed(self, category="c", version="v1", stage="dev"):
        return self.call("put_prompt", category=category,
                         text=_valid_md(version=version), stage=stage)

    def test_list_shape(self):
        self.seed("c1", "default")
        self.seed("c1", "v1")
        self.seed("c2", "default")
        out = self.call("list_prompts")
        self.assertEqual([c["category"] for c in out["categories"]],
                         ["c1", "c2"])
        self.assertEqual(out["total_versions"], 3)
        one = self.call("list_prompts", category="c1")
        self.assertEqual(len(one["categories"][0]["versions"]), 2)
        entry = one["categories"][0]["versions"][0]
        self.assertEqual(set(entry), {"version", "stage", "origin",
                                      "deprecated", "comment", "created_at"})

    def test_promote_ratchet(self):
        self.seed()
        out = self.call("promote_prompt", category="c", version="v1")
        self.assertEqual((out["previous_stage"], out["stage"]),
                         ("dev", "prod"))
        again = self.call("promote_prompt", category="c", version="v1")
        self.assertIn("changed nothing", again["message"])
        down = self.call("promote_prompt", category="c", version="v1",
                         to_stage="dev")
        self.assertIn("changed nothing", down["message"])
        self.assertEqual(down["stage"], "prod")

    def test_promotion_rewrites_only_lifecycle_fields(self):
        self.seed()
        before = self.call("get_prompt", category="c", version="v1")
        self.call("promote_prompt", category="c", version="v1")
        after = self.call("get_prompt", category="c", version="v1")
        for field in ("system", "user_template", "variables", "source_md5",
                      "created_at", "origin", "comment", "aux"):
            self.assertEqual(before[field], after[field], field)
        self.assertEqual(after["stage"], "prod")
        self.assertIsNotNone(after["promoted_at"])

    def test_default_promote_is_a_noop_success(self):
        self.seed("c", "default")
        out = self.call("promote_prompt", category="c", version="default")
        self.assertIn("changed nothing", out["message"])

    def test_delete_frees_the_name(self):
        self.seed()
        out = self.call("delete_prompt", category="c", version="v1")
        self.assertTrue(out["deleted"])
        self.assertEqual(out["stage_was"], "dev")
        # the name is genuinely reusable
        again = self.seed()
        self.assertNotIn("error", again)

    def test_prod_is_undeletable_and_default_is_undeprecatable(self):
        self.seed("c", "v1", stage="dev")
        self.call("promote_prompt", category="c", version="v1")
        self.assertEqual(
            self.call("delete_prompt", category="c",
                      version="v1")["errorCode"], "PROD_UNDELETABLE")
        self.seed("c", "default")
        self.assertEqual(
            self.call("deprecate_prompt", category="c",
                      version="default")["errorCode"], "DEFAULT_PROTECTED")

    def test_deprecate_is_idempotent(self):
        self.seed()
        first = self.call("deprecate_prompt", category="c", version="v1")
        self.assertTrue(first["deprecated"])
        second = self.call("deprecate_prompt", category="c", version="v1")
        self.assertIn("changed nothing", second["message"])

    def test_describe_tool_has_contract_and_vocabulary_but_no_arn(self):
        out = self.call("describe_tool")
        self.assertEqual(out["output_contract"]["target"], "prompt-registry")
        self.assertIn("variables", out["vocabulary"])
        # No state machine exists, so no arn is claimed — the first target
        # for which that is honest.
        self.assertNotIn("state_machine_arn", out)
        self.assertEqual(out["version"], self.gw.TOOL_VERSION)


class TestInputExamples(_ServiceCase):
    """input_examples.json — the INPUT twin of the output contract, served
    by describe_tool. Every example round-trips through the REAL
    lambda_handler (the suite's fake S3 makes all seven cheap — no
    validation-only shortcuts needed), so an example that drifts from the
    real contract fails this build, not an agent's first call."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with open(os.path.join(ROOT, "gateway_tool",
                               "input_examples.json")) as f:
            cls.examples = json.load(f)["tools"]

    def test_input_examples_cover_exactly_the_dispatched_tools(self):
        self.assertEqual(sorted(self.examples),
                         ["delete_prompt", "deprecate_prompt",
                          "describe_tool", "get_prompt", "list_prompts",
                          "promote_prompt", "put_prompt"])
        for tool, entry in self.examples.items():
            # `alternatives` is optional: present only on tools with a
            # one-of rule (none here today), holding one valid example of
            # each other form
            self.assertIn(sorted(entry),
                          (["example", "required"],
                           ["alternatives", "example", "required"]), tool)
            # every PLAIN required name must appear in the example; a prose
            # one-of entry (contains a space or "|") states a rule the
            # schema cannot express and is exempt
            for name in entry["required"]:
                if " " in name or "|" in name:
                    continue
                self.assertIn(name, entry["example"],
                              f"{tool}: required field {name!r} missing "
                              f"from its example")

    def test_describe_tool_carries_input_examples(self):
        out = self.call("describe_tool")
        self.assertIn("put_prompt", out["input_examples"]["tools"])

    def test_examples_round_trip_through_the_real_handler(self):
        ex = self.examples
        put = self.call("put_prompt", **ex["put_prompt"]["example"])
        self.assertNotIn("error", put, f"put_prompt example refused: {put}")
        self.assertFalse(put["warnings"],
                         "the example prompt must lint clean")
        got = self.call("get_prompt", **ex["get_prompt"]["example"])
        self.assertNotIn("error", got, f"get_prompt example refused: {got}")
        self.assertEqual(got["version"], put["version"])
        lst = self.call("list_prompts", **ex["list_prompts"]["example"])
        self.assertEqual(lst["total_versions"], 1)
        # delete while still dev (prod is undeletable), then re-put so the
        # lifecycle examples have a version to act on
        deleted = self.call("delete_prompt", **ex["delete_prompt"]["example"])
        self.assertTrue(deleted.get("deleted"),
                        f"delete_prompt example refused: {deleted}")
        self.call("put_prompt", **ex["put_prompt"]["example"])
        promoted = self.call("promote_prompt",
                             **ex["promote_prompt"]["example"])
        self.assertEqual(promoted.get("stage"), "prod",
                         f"promote_prompt example refused: {promoted}")
        dep = self.call("deprecate_prompt",
                        **ex["deprecate_prompt"]["example"])
        self.assertTrue(dep.get("deprecated"),
                        f"deprecate_prompt example refused: {dep}")


class TestAdminSurface(_ServiceCase):
    """SERVE_INTERNAL (2026-09-08): internal categories exist only on
    the admin surface. Internality = the DEFAULT version's Audience
    header; the cache is cleared per test."""

    INTERNAL_MD = ("<!-- Version: default -->\n"
                   "<!-- Audience: workflow-internal — owning workflow"
                   "'s judgment config -->\n<!-- Comment: t -->\n"
                   "# System Prompt\nX.\n# User Prompt\n{{body}}\n")

    def setUp(self):
        super().setUp()
        self.gw._INTERNAL_CACHE.clear()
        self.addCleanup(self.gw._INTERNAL_CACHE.clear)

    def _seed_internal(self, category="ad-x"):
        with mock.patch.object(self.gw, "SERVE_INTERNAL", True):
            out = self.call("put_prompt", category=category,
                            text=self.INTERNAL_MD)
            self.assertNotIn("error", out)

    def test_standard_surface_hides_internal_categories(self):
        self._seed_internal("ad-x")
        with mock.patch.object(self.gw, "SERVE_INTERNAL", True):
            self.call("put_prompt", category="visible",
                      text=_valid_md("default"))
        self.gw._INTERNAL_CACHE.clear()
        with mock.patch.object(self.gw, "SERVE_INTERNAL", False):
            out = self.call("list_prompts")
            names = [c["category"] for c in out["categories"]]
            self.assertEqual(names, ["visible"])
            got = self.call("get_prompt", category="ad-x")
            self.assertEqual(got.get("errorCode"), "INTERNAL_CATEGORY")
            self.assertIn("admin surface", got["message"])
            # every category-taking mutation is refused the same way
            for tool in ("promote_prompt", "delete_prompt",
                         "deprecate_prompt", "put_prompt"):
                out = self.call(tool, category="ad-x",
                                version="v2", text=_valid_md("v2"))
                self.assertEqual(out.get("errorCode"),
                                 "INTERNAL_CATEGORY", tool)

    def test_admin_surface_serves_everything(self):
        self._seed_internal("ad-x")
        self.gw._INTERNAL_CACHE.clear()
        with mock.patch.object(self.gw, "SERVE_INTERNAL", True):
            got = self.call("get_prompt", category="ad-x")
            self.assertNotIn("error", got)
            names = [c["category"]
                     for c in self.call("list_prompts")["categories"]]
            self.assertIn("ad-x", names)

    def test_a_category_cannot_be_born_internal_on_the_standard_surface(self):
        with mock.patch.object(self.gw, "SERVE_INTERNAL", False):
            out = self.call("put_prompt", category="new-int",
                            text=self.INTERNAL_MD)
            self.assertEqual(out.get("errorCode"), "INTERNAL_CATEGORY")

    def test_default_less_internal_versions_are_still_hidden(self):
        # the backstop: a category with no default cannot smuggle an
        # internal VERSION past the standard surface (review,
        # 2026-09-08 — the category gate alone keyed on the default)
        md = ("<!-- Version: wip -->\n"
              "<!-- Audience: workflow-internal — x -->\n"
              "<!-- Comment: t -->\n"
              "# System Prompt\nX.\n# User Prompt\n{{body}}\n")
        with mock.patch.object(self.gw, "SERVE_INTERNAL", True):
            out = self.call("put_prompt", category="orphan", text=md)
            self.assertNotIn("error", out)
        self.gw._INTERNAL_CACHE.clear()
        with mock.patch.object(self.gw, "SERVE_INTERNAL", False):
            got = self.call("get_prompt", category="orphan",
                            version="wip")
            self.assertEqual(got.get("errorCode"), "INTERNAL_CATEGORY")
            listed = self.call("list_prompts", category="orphan")
            self.assertEqual(listed.get("errorCode"), "PROMPT_NOT_FOUND")

    def test_visible_categories_unaffected_by_the_gate(self):
        with mock.patch.object(self.gw, "SERVE_INTERNAL", False):
            out = self.call("put_prompt", category="tunable",
                            text=_valid_md("default"))
            self.assertNotIn("error", out)
            got = self.call("get_prompt", category="tunable")
            self.assertNotIn("error", got)


class TestAgentTunableCategoriesStayVisible(unittest.TestCase):
    """The 2026-09-08 near-miss: EVERY builtin carried the prose
    'Audience: workflow-internal' header, so the machine-readable gate
    would have emptied the standard surface — hiding the two categories
    that are agent-tunable BY DESIGN. Pin their headers."""

    def test_ca_and_sme_headers_are_not_internal(self):
        import glob
        root = os.path.join(ROOT, "prompts")
        files = (glob.glob(os.path.join(
                     root, "conversation-analysis", "*.md"))
                 + glob.glob(os.path.join(
                     root, "shot-metadata-enrichment", "*.md")))
        self.assertGreaterEqual(len(files), 4)
        for f in files:
            text = _read(f, encoding="utf-8")
            self.assertNotIn("Audience: workflow-internal", text, f)

    # the agent-tunable set is EXACTLY these two categories; every
    # other builtin is internal — scanned, never hand-enumerated
    AGENT_TUNABLE = {"conversation-analysis", "shot-metadata-enrichment"}

    def test_every_builtin_declares_a_recognized_audience(self):
        import glob
        root = os.path.join(ROOT, "prompts")
        for f in glob.glob(os.path.join(root, "*", "*.md")):
            cat = os.path.basename(os.path.dirname(f))
            text = _read(f, encoding="utf-8")
            if cat in self.AGENT_TUNABLE:
                self.assertIn("Audience: agent-tunable", text, f)
            else:
                self.assertIn("Audience: workflow-internal", text, f)


class TestCastPromptParserPins(unittest.TestCase):
    """The matched prompt+parser pins, IN THE REPO A PROMPT EDIT
    TOUCHES (review, 2026-09-08): cast-resolution's parsers drop
    unrecognized JSONL rows by design, so a field rename here would
    silently zero that workflow's enrichment output — this suite runs
    on every prompt-registry build, unconditionally.
    (cast-resolution keeps sibling-read copies as belt-and-braces.)"""

    _PAIRS = (
        ("cast-adjudicate",
         ("person_id", "verdict", "name", "character_name", "evidence")),
        ("cast-critic2",
         ("person_id", "verdict", "suspect_field", "conflicts")),
        ("cast-vocatives", ("window_id", "addressee")),
    )

    @staticmethod
    def _text(category):
        return _read(os.path.join(ROOT, "prompts", category,
                                 "default.md"), encoding="utf-8")

    def test_each_prompt_states_the_fields_its_parser_reads(self):
        for category, tokens in self._PAIRS:
            text = self._text(category)
            for token in tokens:
                self.assertIn(f'"{token}"', text,
                              f"{category} no longer names {token!r} — "
                              "cast-resolution's parser would silently "
                              "drop every row")

    def test_critic2_prompt_states_the_two_rules(self):
        text = self._text("cast-critic2")
        self.assertIn("ABSENCE of additional evidence is NEVER suspect",
                      text)
        self.assertIn("NEVER question an observed item's reliability",
                      text)


class TestArtworkPromptParserPins(unittest.TestCase):
    """artwork-selection's parsers (assemble: the vision verdict;
    filter: the brief-amendment firewall) accept ONLY these fields and
    silently drop anything else — a field rename here would zero that
    workflow's rerank/brief output. Same pin discipline as the cast
    pairs above."""

    _PAIRS = (
        ("artwork-rerank", ("ranking", "top_reason", "unusable",
                            "file", "why")),
        ("artwork-brief", ("archetype_counts", "must_include_people",
                           "avoid", "emphasis")),
    )

    @staticmethod
    def _text(category):
        return _read(os.path.join(ROOT, "prompts", category,
                                 "default.md"), encoding="utf-8")

    def test_each_prompt_states_the_fields_its_parser_reads(self):
        for category, tokens in self._PAIRS:
            text = self._text(category)
            for token in tokens:
                self.assertIn(f'"{token}"', text,
                              f"{category} no longer names {token!r} — "
                              "artwork-selection's parser would "
                              "silently drop it")

    def test_brief_prompt_states_the_containment_rules(self):
        # the two rules the firewall depends on: closed vocabulary
        # only, and no invented people
        text = self._text("artwork-brief")
        self.assertIn("archetype_vocabulary", text)
        self.assertIn("verbatim from the brief", text)


class TestSeeder(_ServiceCase):
    def _run(self, request_type="Create", bucket="test-proxy",
             seed_mode=None):
        import seed_prompts
        responses = []
        props = {"Bucket": bucket}
        if seed_mode is not None:
            props["SeedMode"] = seed_mode
        with mock.patch.object(seed_prompts, "_respond",
                               lambda *a, **k: responses.append((a, k))):
            result = seed_prompts.lambda_handler(
                {"RequestType": request_type,
                 "ResourceProperties": props}, None)
        return result, responses

    def test_seeds_every_builtin_as_prod_builtin(self):
        result, responses = self._run()
        self.assertEqual(result["seeded"], len(load_corpus()))
        self.assertEqual(result["skipped"], 0)
        got = self.call("get_prompt", category="conversation-analysis")
        self.assertEqual((got["stage"], got["origin"]), ("prod", "builtin"))
        self.assertEqual(responses[0][0][2], "SUCCESS")

    def test_second_run_skips_everything(self):
        self._run()
        result, _ = self._run()
        self.assertEqual(result["seeded"], 0)
        self.assertEqual(result["skipped"], len(load_corpus()))

    def test_delete_is_a_noop_success_that_removes_nothing(self):
        self._run()
        count = len(self.s3.objects)
        result, responses = self._run("Delete")
        self.assertEqual(result["action"], "delete-noop")
        self.assertEqual(len(self.s3.objects), count)
        self.assertEqual(responses[0][0][2], "SUCCESS")

    def test_every_category_ships_a_default(self):
        """The protected entry point must exist in every seeded category —
        a culling that removes a category's default.md fails here, not in a
        user's version-less get_prompt."""
        categories = {c for c, _ in load_corpus()}
        defaults = {c for c, v in load_corpus() if v == "default"}
        self.assertEqual(categories, defaults)

    # --- SeedMode: overwrite-builtins, the dev backdoor (2026-08-17) ------

    def _tamper_builtin(self, category="conversation-analysis",
                        version="default"):
        """Simulate an OLDER seeded builtin: rewrite the stored pair so its
        source bytes (and md5) differ from what prompts/ now ships."""
        import lambda_function
        key = ("test-proxy",
               f"_promptregistry/{category}/{version}/artifact.json")
        art = json.loads(self.s3.objects[key])
        art["source_md5"] = "0" * 32
        art["system"] = "an older builtin body"
        self.s3.objects[key] = json.dumps(art).encode("utf-8")
        return key

    def test_overwrite_mode_updates_a_changed_builtin_in_place(self):
        self._run()
        key = self._tamper_builtin()
        result, responses = self._run(seed_mode="overwrite-builtins")
        self.assertEqual(result["seeded"], 0)
        self.assertEqual(result["updated"], 1)
        self.assertEqual(result["skipped"], len(load_corpus()) - 1)
        got = self.call("get_prompt", category="conversation-analysis")
        self.assertEqual((got["stage"], got["origin"]), ("prod", "builtin"))
        self.assertNotEqual(got["source_md5"], "0" * 32)
        self.assertIn("updated 1", responses[0][0][4]["Message"])

    def test_overwrite_mode_skips_unchanged_builtins(self):
        """Same shipped bytes -> no rewrite: flipping the mode on without a
        prompt edit must be a no-op (updated 0), not a mass re-stamp."""
        self._run()
        result, _ = self._run(seed_mode="overwrite-builtins")
        self.assertEqual((result["seeded"], result["updated"]), (0, 0))
        self.assertEqual(result["skipped"], len(load_corpus()))

    def test_overwrite_mode_never_touches_a_user_version(self):
        """A user version whose name a later release ships as a builtin is
        NOT clobbered — origin gates the backdoor."""
        import lambda_function
        self._run()
        key = ("test-proxy",
               "_promptregistry/conversation-analysis/default/artifact.json")
        art = json.loads(self.s3.objects[key])
        art["origin"] = "user"
        art["source_md5"] = "0" * 32
        self.s3.objects[key] = json.dumps(art).encode("utf-8")
        result, _ = self._run(seed_mode="overwrite-builtins")
        self.assertEqual(result["updated"], 0)
        got = self.call("get_prompt", category="conversation-analysis")
        self.assertEqual(got["origin"], "user")
        self.assertEqual(got["source_md5"], "0" * 32)

    def test_default_mode_never_overwrites_even_a_changed_builtin(self):
        self._run()
        self._tamper_builtin()
        result, _ = self._run()   # write-if-absent
        self.assertNotIn("updated", {k: v for k, v in result.items()
                                     if k == "updated" and v})
        self.assertEqual(result.get("updated", 0), 0)
        got = self.call("get_prompt", category="conversation-analysis")
        self.assertEqual(got["source_md5"], "0" * 32)

    def test_unknown_seed_mode_fails_the_stack(self):
        result, responses = self._run(seed_mode="overwrite-everything")
        self.assertFalse(result["handled"])
        self.assertEqual(responses[0][0][2], "FAILED")

    def test_put_prompt_has_no_overwrite_path(self):
        """The API surface stays fully immutable: no request field reaches
        save_prompt's overwrite kwarg through put_prompt."""
        self._run()
        text = _read(os.path.join(ROOT, "prompts", "conversation-analysis",
                                 "default.md"), encoding="utf-8")
        res = self.call("put_prompt", category="conversation-analysis",
                        text=text, overwrite=True)
        self.assertTrue(res.get("error"))
        self.assertEqual(res["errorCode"], "VERSION_EXISTS")


class TestTemplateAndBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml

        class L(yaml.SafeLoader):
            pass

        def unknown(loader, suffix, node):
            if isinstance(node, yaml.ScalarNode):
                return loader.construct_scalar(node)
            if isinstance(node, yaml.SequenceNode):
                return loader.construct_sequence(node)
            return loader.construct_mapping(node)
        L.add_multi_constructor("!", unknown)
        with open(os.path.join(ROOT, "cfnstack.yaml")) as f:
            cls.tpl = yaml.load(f, Loader=L)  # nosec B506
        with open(os.path.join(ROOT, "build-deploy-s3-dist.sh")) as f:
            cls.build = f.read()

    def test_no_state_machine_no_container_no_table(self):
        types = {r["Type"] for r in self.tpl["Resources"].values()}
        for forbidden in ("AWS::StepFunctions::StateMachine",
                          "AWS::CodeBuild::Project", "AWS::ECR::Repository",
                          "AWS::DynamoDB::Table"):
            self.assertNotIn(forbidden, types)

    def test_lambdas_are_unnamed_with_logging_config(self):
        for name, res in self.tpl["Resources"].items():
            if res["Type"] != "AWS::Lambda::Function":
                continue
            self.assertNotIn("FunctionName", res["Properties"], name)
            self.assertIn("LoggingConfig", res["Properties"], name)

    def test_seeder_depends_on_its_log_group(self):
        self.assertEqual(self.tpl["Resources"]["SeederLambda"]["DependsOn"],
                         "SeederLogGroup")

    def test_no_role_carries_the_basic_execution_policy(self):
        self.assertNotIn("AWSLambdaBasicExecutionRole",
                         json.dumps(self.tpl))

    def test_s3_grants_are_scoped_to_the_registry_prefix(self):
        body = json.dumps(self.tpl)
        self.assertIn("_promptregistry/*", body)
        # The seeder role must NOT be able to delete.
        seeder_policy = json.dumps(
            self.tpl["Resources"]["SeederRole"]["Properties"]["Policies"])
        self.assertNotIn("s3:DeleteObject", seeder_policy)
        # ...but MUST carry the scoped ListBucket: without it S3 answers
        # GetObject on a missing key with 403 instead of 404, and the
        # write-if-absent existence check reads a permission error where
        # "absent" is the truth. Failed the first live deploy (2026-08-15).
        self.assertIn("s3:ListBucket", seeder_policy)

    def test_seed_mode_defaults_overwrite_builtins_UNTIL_GA(self):
        """Pre-GA posture (user decision 2026-08-18): the prompts are under
        heavy development, so the DEFAULT is the backdoor — every deploy
        reseeds builtins whose shipped source changed, with no knob to
        remember anywhere (the bundle deliberately passes nothing). THE GA
        OBLIGATION LIVES HERE: flip the Default to write-if-absent when the
        defaults freeze for real, and update this test in the same diff —
        its name is the reminder. The parameter must also actually reach
        the custom resource, or flipping it changes nothing."""
        param = self.tpl["Parameters"]["PromptSeedMode"]
        self.assertEqual(param["Default"], "overwrite-builtins")
        self.assertEqual(sorted(param["AllowedValues"]),
                         ["overwrite-builtins", "write-if-absent"])
        props = self.tpl["Resources"]["PromptSeeder"]["Properties"]
        self.assertEqual(props["SeedMode"], "PromptSeedMode")  # !Ref target

    def test_output_exposes_the_gateway_arn(self):
        self.assertIn("GatewayToolLambdaArn", self.tpl["Outputs"])

    def test_gateway_zip_members_are_guarded(self):
        match = re.search(r"members=\(([^)]*)\)", self.build)
        members = match.group(1).split()
        for required in ("lambda_function.py", "seed_prompts.py",
                         "prompt_md.py", "prompt_render.py",
                         "workflow_history.py", "vocabulary.py",
                         "output_contract.json", "input_examples.json"):
            self.assertIn(required, members)
        self.assertIn('[ ! -f "gateway_tool/$f" ]', self.build)
        self.assertEqual(self.build.count("members=("), 1)

    def test_token_guard_is_digit_tolerant(self):
        self.assertIn("%%[A-Z0-9_]*%%", self.build)


class TestOutputContractFile(unittest.TestCase):
    def test_envelope_and_inline_kinds(self):
        with open(os.path.join(ROOT, "gateway_tool",
                               "output_contract.json")) as f:
            contract = json.load(f)
        self.assertEqual(contract["target"], "prompt-registry")
        self.assertEqual(contract["contract_version"], 1)
        self.assertTrue(contract["stability"])
        self.assertEqual(set(contract["outputs"]),
                         {"prompt-artifact", "put-result", "list-entry",
                          "lifecycle-result"})
        for label, entry in contract["outputs"].items():
            self.assertEqual(entry["kind"], "inline", label)
            self.assertNotIn("\n", entry["description"], label)
            self.assertIn("example", entry, label)

if __name__ == "__main__":
    unittest.main()
