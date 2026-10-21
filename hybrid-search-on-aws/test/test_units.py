"""
Unit tests — hybrid-search Phase 2 (multi-collection, contracts, per-table
locks). Mocked boto3 only; no AWS calls. Run: cd test && python3 test_units.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import unittest
from unittest import mock

try:
    import yaml
except ImportError:                                  # pragma: no cover
    yaml = None                                      # see test/requirements.txt

os.environ.setdefault("INDEX_STATE_TABLE", "test-index-state")
os.environ.setdefault("DEDUP_TABLE", "test-dedup")
os.environ.setdefault("SEARCH_HISTORY_TABLE", "test-history")
os.environ.setdefault("COLLECTION_TABLE", "test-collections")
os.environ.setdefault("PROXY_BUCKET", "test-proxy")
os.environ.setdefault("VECTOR_BUCKET", "test-vectors")
# the gateway dispatcher reads these (and creates boto3 clients) at import
# time; no AWS call is ever made in tests
os.environ.setdefault("AWS_DEFAULT_REGION", "eu-west-1")
os.environ.setdefault(
    "INDEX_STATE_MACHINE_ARN",
    "arn:aws:states:eu-west-1:000000000000:stateMachine:test-hs-indexer")
os.environ.setdefault(
    "SEARCH_STATE_MACHINE_ARN",
    "arn:aws:states:eu-west-1:000000000000:stateMachine:test-hs-search")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# lib/collections — id validation, key derivation
# ─────────────────────────────────────────────────────────────────────────────


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


class TestCollections(unittest.TestCase):
    def test_validate_collection_id(self):
        from lib.collections import validate_collection_id
        self.assertEqual(validate_collection_id("news"), "news")
        self.assertEqual(validate_collection_id("news_2024-q1"), "news_2024-q1")
        for bad in ("", "ab", "News", "a" * 49, "schema", "has space", "-lead"):
            with self.assertRaises(ValueError, msg=bad):
                validate_collection_id(bad)
        with self.assertRaises(ValueError):
            validate_collection_id("scene", known_doc_types={"scene", "shot"})

    def test_index_key_and_location(self):
        from lib.collections import index_key, lancedb_location
        self.assertEqual(index_key("news", "scene:cohere.embed-v4:0:en"),
                         "news:scene:cohere.embed-v4:0:en")
        loc = lancedb_location(
            "news", "_lancedb_index/scene/cohere.embed-v4:0/1024/en")
        self.assertEqual(loc, "_lancedb_index/news/scene/cohere.embed-v4:0/1024/en")

    def test_s3v_index_name_deterministic_and_capped(self):
        from lib.collections import s3v_index_name
        a = s3v_index_name("news", "scene:cohere.embed-v4:0:en")
        b = s3v_index_name("news", "scene:cohere.embed-v4:0:en")
        c = s3v_index_name("sport", "scene:cohere.embed-v4:0:en")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertTrue(a.startswith("news--scene--"))
        self.assertLessEqual(len(a), 63)
        self.assertNotIn("_", a)  # underscores mapped to hyphens
        long = s3v_index_name("a" * 48, "verylongdoctype:model:en")
        self.assertLessEqual(len(long), 63)

    def test_s3v_index_name_gateway_parity(self):
        # gateway_tool/common.py mirrors lib/collections.s3v_index_name
        # (the gateway zip excludes lib/) — drift here breaks the admin
        # delete_collection sweep. Keep the two byte-identical.
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gateway_tool"))
        from lib.collections import s3v_index_name as lib_name
        from common import s3v_index_name as gw_name
        for cid, tk in [("news", "scene:cohere.embed-v4:0:en"),
                        ("a" * 48, "shot:titan-embed-image-v1"),
                        ("my_news", "topic:nova-2-multimodal-embeddings:en")]:
            self.assertEqual(lib_name(cid, tk), gw_name(cid, tk))

    def test_max_inline_documents_gateway_parity(self):
        # gateway_tool/common.py mirrors lib/input_schema.MAX_INLINE_DOCUMENTS
        # (submit-time cap vs backend cap) — drift would let the gateway
        # accept batches the SFN then rejects async, or vice versa.
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gateway_tool"))
        from lib.input_schema import MAX_INLINE_DOCUMENTS as lib_cap
        from common import MAX_INLINE_DOCUMENTS as gw_cap
        self.assertEqual(lib_cap, gw_cap)

    def test_idempotency_token_is_ignored_with_a_log_never_rejected(self):
        """No tool on either hybrid-search target honours an idempotency token,
        and none rejects one either.

        `index_content` names its execution uuid4() on purpose — indexing is an
        accumulating write, so two calls with different documents must both run
        — and dedups per-document on a content hash one layer down, which makes
        a retry safe with no token at all. Searches and reads must reflect
        current state. So the field can only ever be ignored; the question is
        whether that is auditable or invisible.

        It returns a LOG LINE, not an error dict: the caller is an LLM, a
        hallucinated parameter name is a routine failure mode, and failing an
        otherwise-correct search over a field that could not have changed the
        results is the worse trade. A regression to the error shape would be
        silent at import and only surface as failed customer requests.
        """
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gateway_tool"))
        from common import warn_idempotency_token as warn
        for event in ({"idempotency_token": "tok"}, {"uuid": "tok"}):  # nosec B105
            note = warn(event)
            self.assertIsInstance(note, str, f"not a log line: {note!r}")
            self.assertTrue(note)
        # Conditional, or the warning is noise on every single call.
        self.assertIsNone(warn({}))
        self.assertIsNone(warn({"idempotency_token": ""}))  # nosec B105
        self.assertIsNone(warn({"collection_id": "news", "query": "x"}))

    def test_both_gateway_dispatchers_wire_the_token_warning(self):
        """The check belongs at the dispatcher, once, before routing — so a tool
        added later is covered by default rather than by someone remembering.

        Asserted against the parsed AST of each handler rather than a text
        search: `common.py` names the helper in its own docstring, so grepping
        for the string stays green even after the call is deleted. That exact
        false PASS already happened once in this repo (Phase 1's zip-member
        guard), which is why this walks the tree.
        """
        import ast
        gw_dir = os.path.join(os.path.dirname(__file__), "..", "gateway_tool")
        for fname in ("lambda_function.py", "admin_function.py"):
            with self.subTest(module=fname):
                tree = ast.parse(open(os.path.join(gw_dir, fname)).read())
                handler = next(
                    n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "lambda_handler")
                called = {
                    n.func.id for n in ast.walk(handler)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
                self.assertIn("_warn_idempotency_token", called,
                              f"{fname}'s lambda_handler never calls it")


# ─────────────────────────────────────────────────────────────────────────────
# lib/contracts — doc_id, validation, entity merge
# ─────────────────────────────────────────────────────────────────────────────

def _fake_schema(doc_type="scene", ann_field="vector",
                 mappings=None):
    sd = mock.Mock()
    sd.doc_type = doc_type
    sd.ann_field = ann_field
    sd._mappings = mappings or {
        "doc_id": {}, "asset_id": {}, "collection_id": {}, "start_ms": {},
        "end_ms": {}, "text": {}, "named_entities": {}, "vector": {},
        "segment_parent_id": {},
    }
    return sd


class TestContracts(unittest.TestCase):
    def test_doc_id_includes_collection(self):
        from lib.contracts import compute_doc_id
        a = compute_doc_id("news", "asset:1", "scene", 0, 1000)
        b = compute_doc_id("sport", "asset:1", "scene", 0, 1000)
        self.assertNotEqual(a, b)  # cross-collection dedup collision guard
        self.assertEqual(a, compute_doc_id("news", "asset:1", "scene", 0, 1000, 0))
        self.assertNotEqual(a, compute_doc_id("news", "asset:1", "scene", 0, 1000, 1))

    def _contract(self, **over):
        base = {
            "asset_id": "asset:uuid:abc", "doc_type": "scene",
            "start_ms": 0, "end_ms": 5000,
            "content": [{"kind": "text", "field": "text", "value": "hello"},
                        {"kind": "text", "field": "vector", "value": "hello"}],
        }
        base.update(over)
        return base

    def test_validate_ok(self):
        from lib.contracts import validate_contract
        out = validate_contract(self._contract(), _fake_schema(), "news")
        self.assertEqual(out["collection_id"], "news")
        self.assertTrue(out["doc_id"])
        self.assertFalse(out["refresh"])

    def test_validate_failures(self):
        from lib.contracts import validate_contract
        sd = _fake_schema()
        cases = [
            self._contract(doc_type="shot"),                      # schema mismatch
            self._contract(asset_id=""),                          # missing asset
            self._contract(start_ms=None),                        # bad span
            self._contract(content=[{"kind": "audio", "field": "text", "value": "x"}]),
            self._contract(content=[{"kind": "text", "field": "nope", "value": "x"}]),
            self._contract(content=[{"kind": "image", "field": "text",
                                     "value": "s3://b/k.jpg"}]),  # image not at ann
            self._contract(content=[{"kind": "image", "field": "vector",
                                     "value": "http://x"}]),      # non-s3 image
            self._contract(content=[], fields={}),                # empty
            self._contract(fields={"vector": [1.0]}, content=[]), # verbatim at ann
            self._contract(content=[{"kind": "text", "field": "text",
                                     "value": "x"}]),             # no vector item
        ]
        for c in cases:
            with self.assertRaises(ValueError, msg=json.dumps(c)[:80]):
                validate_contract(c, sd, "news")

    def test_merge_entities_dedupe_append(self):
        from lib.contracts import merge_entities
        out = merge_entities(["Jane Doe", "John Roe"],
                             ["JOHN ROE", " Richard Roe ", "jane doe"])
        self.assertEqual(out, ["Jane Doe", "John Roe", "Richard Roe"])


# ─────────────────────────────────────────────────────────────────────────────
# lib/lock_utils — per-table locks, per-lock pending queues
# ─────────────────────────────────────────────────────────────────────────────

class _FakeLockStore:
    """Minimal pk/sk item store standing in for ddb_utils + table queries."""
    def __init__(self):
        self.items: dict[tuple, dict] = {}

    def conditional_put(self, item, condition, attr_values=None):
        key = (item["pk"], item["sk"])
        if key in self.items:
            existing = self.items[key]
            if attr_values and existing.get("execution_arn") == attr_values.get(":arn"):
                self.items[key] = item
                return True
            return False
        self.items[key] = item
        return True

    def conditional_delete(self, pk, sk, condition, attr_values=None):
        key = (pk, sk)
        it = self.items.get(key)
        if it and it.get("execution_arn") == (attr_values or {}).get(":arn"):
            del self.items[key]
            return True
        return False

    def delete(self, pk, sk):
        self.items.pop((pk, sk), None)

    def put(self, item):
        self.items[(item["pk"], item["sk"])] = item

    def query_begins(self, pk, sk_prefix, limit=None):
        out = [v for (p, s), v in self.items.items()
               if p == pk and s.startswith(sk_prefix)]
        return out[:limit] if limit else out


class TestLockUtils(unittest.TestCase):
    def setUp(self):
        import lib.lock_utils as lu
        self.lu = lu
        self.store = _FakeLockStore()
        self.patches = [
            mock.patch.object(lu, "conditional_put_item", self.store.conditional_put),
            mock.patch.object(lu, "conditional_delete_item", self.store.conditional_delete),
            mock.patch.object(lu, "delete_item", self.store.delete),
            mock.patch.object(lu, "put_item", self.store.put),
        ]
        for p in self.patches:
            p.start()
        # get_table().query for pop_next_pending / release_all
        table = mock.Mock()
        def fake_query(KeyConditionExpression=None, Limit=None, **kw):
            # Extract string leaves from the boto3 ConditionBase tree:
            # Key("pk").eq(x) -> values (Key, x); & adds nested conditions.
            def leaves(e):
                out = []
                vals = getattr(e, "_values", None)
                if vals is None:
                    if isinstance(e, str):
                        out.append(e)
                    return out
                for v in vals:
                    out.extend(leaves(v))
                return out
            strs = leaves(KeyConditionExpression)
            pk     = strs[0] if strs else "lock"
            prefix = strs[1] if len(strs) > 1 else ""
            return {"Items": self.store.query_begins(pk, prefix, Limit)}
        table.query = fake_query
        self.patches.append(mock.patch.object(lu, "get_table", lambda: table))
        self.patches[-1].start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_locks_are_independent(self):
        lu = self.lu
        self.assertTrue(lu.acquire("news:scene:m", "index", "arn:A"))
        # DIFFERENT table: no contention
        self.assertTrue(lu.acquire("sport:scene:m", "index", "arn:B"))
        # SAME table: contention
        self.assertFalse(lu.acquire("news:scene:m", "index", "arn:C"))
        # re-entrant for the same execution
        self.assertTrue(lu.acquire("news:scene:m", "index", "arn:A"))

    def test_release_if_owner(self):
        lu = self.lu
        lu.acquire("news:scene:m", "index", "arn:A")
        self.assertFalse(lu.release_if_owner("news:scene:m", "arn:B"))
        self.assertTrue(lu.release_if_owner("news:scene:m", "arn:A"))
        self.assertFalse(lu.release_if_owner("news:scene:m", "arn:A"))  # idempotent

    def test_release_all_for_execution(self):
        lu = self.lu
        lu.acquire("news:scene:m", "index", "arn:A")
        lu.acquire("news:utterance:m", "index", "arn:A")
        lu.acquire("sport:scene:m", "index", "arn:B")
        self.assertEqual(lu.release_all_for_execution("arn:A"), 2)
        self.assertEqual(self.store.query_begins("lock", ""),
                         [{"pk": "lock", "sk": "sport:scene:m",
                           "holder": "index", "execution_arn": "arn:B",
                           "ttl": mock.ANY}])

    def test_pending_queue_is_per_lock(self):
        lu = self.lu
        lu.acquire("news:scene:m", "index", "arn:A")
        lu.acquire("sport:scene:m", "index", "arn:B")
        lu.store_pending("news:scene:m", "token-news", "u1", "arn:C")
        lu.store_pending("sport:scene:m", "token-sport", "u2", "arn:D")
        # release news lock; pop for news must return ONLY the news waiter
        lu.release("news:scene:m")
        popped = lu.pop_next_pending("news:scene:m")
        self.assertEqual(popped["task_token"], "token-news")
        # news lock now held by arn:C (the waiter)
        self.assertFalse(lu.acquire("news:scene:m", "index", "arn:E"))
        # sport waiter untouched
        self.assertEqual(
            len(self.store.query_begins("pending", "sport:scene:m#")), 1)

    def test_pop_next_pending_fifo_within_lock(self):
        lu = self.lu
        lu.store_pending("news:scene:m", "tok-1", "u1", "arn:1")
        # force distinct created_at
        for (p, s), v in self.store.items.items():
            if v.get("task_token") == "tok-1":
                v["created_at"] = 1
        lu.store_pending("news:scene:m", "tok-2", "u2", "arn:2")
        for (p, s), v in self.store.items.items():
            if v.get("task_token") == "tok-2":
                v["created_at"] = 2
        popped = lu.pop_next_pending("news:scene:m")
        self.assertEqual(popped["task_token"], "tok-1")


# ─────────────────────────────────────────────────────────────────────────────
# HandleLockRelease — extracts sk from stream records, per-lock dispatch
# ─────────────────────────────────────────────────────────────────────────────

class TestGatewayValidation(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gateway_tool"))

    def test_validate_retrieval_paths(self):
        from common import validate_retrieval_paths as v
        self.assertIsNone(v({"visual": {"shot_ann": 0.7}}))
        self.assertIsNone(v({}))
        # LLM-agent typos: strings, lists, negatives, bools, NaN
        self.assertIsNotNone(v("visual"))
        self.assertIsNotNone(v({"visual": "high"}))
        self.assertIsNotNone(v({"visual": {"shot_ann": "0.7"}}))
        self.assertIsNotNone(v({"visual": {"shot_ann": -1}}))
        self.assertIsNotNone(v({"visual": {"shot_ann": True}}))
        self.assertIsNotNone(v({"visual": {"shot_ann": float("nan")}}))
        # json.loads accepts "Infinity"/1e999 → inf; str(inf) breaks DDB's
        # N type AFTER validation, so isfinite must catch it here.
        self.assertIsNotNone(v({"visual": {"shot_ann": float("inf")}}))
        self.assertIsNotNone(v({"visual": {"shot_ann": 1e300}}))

    def test_sanitize_cause_strips_stack_trace(self):
        from common import sanitize_cause
        cause = json.dumps({"errorMessage": "boom", "errorType": "ValueError",
                            "stackTrace": ["  File \"/var/task/lib/x.py\"..."]})
        out = sanitize_cause(cause)
        self.assertEqual(out, "ValueError: boom")
        self.assertNotIn("stackTrace", out)
        self.assertNotIn("/var/task", out)
        # Unparseable cause must NEVER leak raw content — SFN truncates
        # cause at 32 KB, so a trace cut mid-JSON fails parsing; the raw
        # fallback would leak exactly what this function exists to strip.
        self.assertEqual(sanitize_cause("plain text", "States.Timeout"),
                         "States.Timeout")
        truncated_trace = json.dumps({"errorMessage": "boom",
                                      "stackTrace": ["/var/task/lib/x.py"]})[:40]
        out = sanitize_cause(truncated_trace)
        self.assertNotIn("/var/task", out)
        self.assertNotIn("boom", out)  # unparseable → generic, no fragments
        # parsed branch is capped too
        big = json.dumps({"errorMessage": "y" * 9000, "errorType": "ValueError"})
        self.assertLessEqual(len(sanitize_cause(big)), 500)


class _Ctx:
    """AgentCore Lambda context stub — only the tool-name channel."""
    def __init__(self, tool):
        self.client_context = type(
            "CC", (), {"custom": {"bedrockAgentCoreToolName": tool}})()


_GW_MODULES: dict = {}


def _load_gateway_module(filename: str, alias: str):
    """ONE loaded instance per dispatcher module for every test class — a
    second spec_from_file_location would build a second live module (own
    boto3 clients, own contract load), and sys.path gains gateway_tool at
    most once.  importlib by file path: a bare `import lambda_function`
    would collide with the root operation router on sys.path."""
    if alias not in _GW_MODULES:
        gw_dir = os.path.join(ROOT, "gateway_tool")
        if gw_dir not in sys.path:
            sys.path.insert(0, gw_dir)  # dispatchers do `from common import …`
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location(
            alias, os.path.join(gw_dir, filename))
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _GW_MODULES[alias] = mod
    return _GW_MODULES[alias]


def _load_hs_gateway():
    return _load_gateway_module("lambda_function.py", "hs_gateway")


def _load_hs_admin():
    return _load_gateway_module("admin_function.py", "hs_admin")


class TestConfigureFullReplacementGuard(unittest.TestCase):
    """configure_search_backend is a FULL-replacement write; a payload that
    omits path groups the record carries is refused by name unless
    replace_all confirms the narrowing (measured live 2026-10-13: a
    dialogue-only payload silently wiped visual/entity_fts/text_overlay on
    two collections)."""

    @classmethod
    def setUpClass(cls):
        cls.adm = _load_hs_admin()

    _CUR = {"collection_id": {"S": "news"},
            "active_scopes": {"L": [{"S": "segment"}, {"S": "utterance"},
                                    {"S": "scene"}, {"S": "topic"}]},
            "retrieval_paths": {"M": {
                "visual": {"M": {"scene_ann": {"N": "0.5"}}},
                "dialogue": {"M": {"utterance_ann": {"N": "0.4"}}}}}}

    def _call(self, event):
        ddb = mock.Mock()
        ddb.get_item.return_value = {"Item": dict(self._CUR)}
        ddb.update_item.return_value = {
            "Attributes": {"weights_locked": {"BOOL": True}}}
        with mock.patch.object(self.adm, "_ddb", ddb):
            out = self.adm.lambda_handler(
                dict(event, collection_id="news"),
                _Ctx("hybrid-search-admin___configure_search_backend"))
        return out, ddb

    def test_partial_payload_refused_naming_the_omitted_groups(self):
        out, ddb = self._call(
            {"retrieval_paths": {"dialogue": {"utterance_ann": 0.4,
                                              "topic_ann": 0.4}}})
        self.assertEqual(out.get("errorCode"), "PARTIAL_REPLACEMENT_PAYLOAD")
        self.assertIn("visual", out["message"])
        self.assertIn("replace_all", out["message"])
        ddb.update_item.assert_not_called()

    def test_string_false_replace_all_keeps_the_guard_armed(self):
        # same string-boolean coercion as `force`: an agent's "false" must
        # not read as true and disable the wipe protection
        out, ddb = self._call(
            {"retrieval_paths": {"dialogue": {"utterance_ann": 0.4}},
             "replace_all": "false"})
        self.assertEqual(out.get("errorCode"), "PARTIAL_REPLACEMENT_PAYLOAD")
        ddb.update_item.assert_not_called()

    def test_active_scopes_omission_is_guarded_too(self):
        # every field this tool writes is a full replacement — the scopes
        # list has the identical silent-wipe hazard as the weights
        out, ddb = self._call({"active_scopes": ["topic"]})
        self.assertEqual(out.get("errorCode"), "PARTIAL_REPLACEMENT_PAYLOAD")
        self.assertIn("segment", out["message"])
        ddb.update_item.assert_not_called()

    def test_inner_subquery_omission_is_guarded_too(self):
        # all groups named, one INNER weight forgotten — the same silent
        # wipe one nesting level down; the diff runs at subquery depth
        out, ddb = self._call(
            {"retrieval_paths": {"visual": {},
                                 "dialogue": {"utterance_ann": 0.4}}})
        self.assertEqual(out.get("errorCode"), "PARTIAL_REPLACEMENT_PAYLOAD")
        self.assertIn("visual.scene_ann", out["message"])
        ddb.update_item.assert_not_called()

    def test_malformed_stored_weights_refuse_the_guarded_write(self):
        # the guard cannot verify omissions against a malformed record —
        # refusing (with replace_all as the deliberate overwrite path) beats
        # silently comparing against an empty set
        ddb = mock.Mock()
        ddb.get_item.return_value = {"Item": {
            "collection_id": {"S": "news"},
            "retrieval_paths": {"S": "corrupted"}}}
        with mock.patch.object(self.adm, "_ddb", ddb):
            out = self.adm.lambda_handler(
                {"collection_id": "news",
                 "retrieval_paths": {"visual": {"scene_ann": 0.5}}},
                _Ctx("hybrid-search-admin___configure_search_backend"))
        self.assertEqual(out.get("errorCode"), "MALFORMED_STORED_RECORD")
        self.assertIn("replace_all", out["message"])
        ddb.update_item.assert_not_called()

    def test_replace_all_permits_a_deliberate_narrowing(self):
        out, ddb = self._call(
            {"retrieval_paths": {"dialogue": {"utterance_ann": 0.4}},
             "replace_all": True})
        self.assertTrue(out.get("updated"))
        ddb.update_item.assert_called_once()

    def test_complete_set_passes_without_the_flag(self):
        out, ddb = self._call(
            {"retrieval_paths": {"visual": {"scene_ann": 0.5},
                                 "dialogue": {"utterance_ann": 0.4,
                                              "topic_ann": 0.4}}})
        self.assertTrue(out.get("updated"))
        ddb.update_item.assert_called_once()


class TestGatewayInputExamples(unittest.TestCase):
    """input_examples.json — the INPUT twin of the output contract, served
    by describe_tool. The submit examples round-trip through the REAL
    lambda_handler with mocked sfn/ddb so drift fails this build, not an
    agent's first call; the index_content document additionally passes the
    backend content-contract validator against the packaged scene schema,
    because the gateway deliberately defers deep document validation to
    StatePrepareDocuments."""

    @classmethod
    def setUpClass(cls):
        cls.gw = _load_hs_gateway()
        with open(os.path.join(ROOT, "gateway_tool",
                               "input_examples.json")) as f:
            cls.examples = json.load(f)["tools"]

    def _call(self, tool, event):
        return self.gw.lambda_handler(dict(event),
                                      _Ctx(f"hybrid-search___{tool}"))

    def test_list_collections_returns_retrieval_paths(self):
        """list_collections carries each record's retrieval_paths /
        image_retrieval_paths — the READ half of configure_search_backend's
        full-replacement write; without it a weights migration cannot compose
        the complete set (measured live 2026-10-13: a partial payload wiped
        every path it omitted)."""
        ddb = mock.Mock()
        ddb.scan.return_value = {"Items": [{
            "collection_id": {"S": "news"},
            "name": {"S": "News"},
            "active_scopes": {"L": [{"S": "topic"}]},
            "retrieval_paths": {"M": {"dialogue": {"M": {
                "topic_ann": {"N": "0.4"}}}}},
        }]}
        ddb.query.return_value = {"Items": []}
        with mock.patch.object(self.gw, "_ddb", ddb):
            out = self._call("list_collections", {})
        rec = out["collections"][0]
        self.assertEqual(rec["retrieval_paths"],
                         {"dialogue": {"topic_ann": 0.4}})
        self.assertEqual(rec["image_retrieval_paths"], {})

    def test_list_collections_tolerates_a_malformed_record(self):
        """One malformed retrieval_paths attribute degrades THAT record's
        paths to {} — it must never fail the registry's only read surface
        for every collection (the tolerant arm the strict create path
        deliberately does not share)."""
        ddb = mock.Mock()
        ddb.scan.return_value = {"Items": [
            {"collection_id": {"S": "bad"},
             "retrieval_paths": {"M": {"visual": {"M": {
                 "scene_ann": {"S": "not-a-number"}}}}}},
            {"collection_id": {"S": "good"},
             "retrieval_paths": {"M": {"visual": {"M": {
                 "scene_ann": {"N": "0.5"}}}}}},
        ]}
        ddb.query.return_value = {"Items": []}
        with mock.patch.object(self.gw, "_ddb", ddb):
            out = self._call("list_collections", {})
        by_id = {c["collection_id"]: c for c in out["collections"]}
        self.assertEqual(by_id["bad"]["retrieval_paths"], {})
        # the degradation is MARKED — a silent {} would dead-end the
        # migration recipe that reads weights here before a full-replacement
        # configure write
        self.assertTrue(by_id["bad"].get("retrieval_paths_malformed"))
        self.assertEqual(by_id["good"]["retrieval_paths"],
                         {"visual": {"scene_ann": 0.5}})
        self.assertNotIn("retrieval_paths_malformed", by_id["good"])

    def test_create_collection_refuses_a_malformed_default_record(self):
        """The inherit path is STRICT: inheriting {} from a malformed
        default record would birth a collection with no retrieval paths
        (searchable in name only), so creation refuses by name instead."""
        ddb = mock.Mock()
        ddb.get_item.return_value = {"Item": {
            "collection_id": {"S": "default"},
            "retrieval_paths": {"M": {"visual": {"M": {
                "scene_ann": {"S": "oops"}}}}}}}
        with mock.patch.object(self.gw, "_ddb", ddb):
            out = self._call("create_collection",
                             {"collection_id": "newsroom",
                              "active_scopes": ["scene"]})
        self.assertEqual(out.get("errorCode"), "MALFORMED_DEFAULT_RECORD")
        ddb.put_item.assert_not_called()

    def test_input_examples_cover_exactly_the_dispatched_tools(self):
        self.assertEqual(sorted(self.examples),
                         ["create_collection", "describe_tool",
                          "get_job_status", "index_content",
                          "list_collections", "search_content"])
        for tool, entry in self.examples.items():
            # `alternatives` is optional: present only on tools with a
            # one-of rule, holding one valid example of each other form
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
        out = self._call("describe_tool", {})
        self.assertIn("index_content", out["input_examples"]["tools"])

    def test_submit_examples_pass_the_gateway_validation(self):
        # index_content: async submit — mocked StartExecution + mocked
        # registry read (the submit-time collection check must not reach
        # a real DynamoDB from a unit test)
        fake = mock.Mock()
        fake.start_execution.return_value = {"executionArn": "arn:fake"}
        ddb = mock.Mock()
        ddb.get_item.return_value = {"Item": {"collection_id": {"S": "default"}}}
        with mock.patch.object(self.gw, "_sfn", fake), \
             mock.patch.object(self.gw, "_ddb", ddb):
            out = self._call("index_content",
                             self.examples["index_content"]["example"])
        self.assertNotIn("error", out, f"index_content example refused: {out}")
        self.assertEqual(out["status"], "RUNNING")
        fake.start_execution.assert_called_once()
        # the gateway defers per-document validation to StatePrepareDocuments
        # — pin the example against the real content-contract validator and
        # the packaged scene schema it would meet downstream
        from lib.backends.lancedb_backend import SchemaDefinition
        from lib.contracts import validate_contract
        with open(os.path.join(ROOT, "schema", "default_scene.json")) as f:
            sd = SchemaDefinition("schema/default_scene.json", json.load(f))
        for doc in self.examples["index_content"]["example"]["documents"]:
            validate_contract(doc, sd, "default")

        # search_content: sync — the handler starts the Search SFN and polls
        # DescribeExecution to completion
        fake = mock.Mock()
        fake.start_execution.return_value = {"executionArn": "arn:fake"}
        fake.describe_execution.return_value = {
            "status": "SUCCEEDED",
            "output": json.dumps({"results": [], "total": 0})}
        with mock.patch.object(self.gw, "_sfn", fake):
            out = self._call("search_content",
                             self.examples["search_content"]["example"])
        self.assertNotIn("error", out, f"search_content example refused: {out}")
        fake.start_execution.assert_called_once()

        # create_collection: sync conditional DDB put.  The default record
        # must carry weights for the example's scopes — an absent/empty one
        # now REFUSES creation (MISSING_DEFAULT_WEIGHTS) instead of birthing
        # a weightless collection; the old "empty maps are valid" posture
        # was reversed 2026-10-13.
        fake = mock.Mock()
        fake.get_item.return_value = {"Item": {"retrieval_paths": {"M": {
            "visual": {"M": {"scene_ann": {"N": "0.5"}}},
            "dialogue": {"M": {"utterance_ann": {"N": "0.4"}}}}}}}
        fake.put_item.return_value = {}
        with mock.patch.object(self.gw, "_ddb", fake):
            out = self._call("create_collection",
                             self.examples["create_collection"]["example"])
        self.assertNotIn("error", out,
                         f"create_collection example refused: {out}")
        self.assertTrue(out["created"])

    def test_staged_alternatives_pass_the_gateway_validation(self):
        # index_content's documents_s3uri form: the gateway validates only
        # the proxy-bucket prefix at submit (manifest reading is
        # StatePrepareDocuments' job), so the round-trip needs the module's
        # PROXY_BUCKET pointed at the examples' own placeholder bucket plus
        # the same StartExecution mock the primary example uses
        alt = self.examples["index_content"]["alternatives"]["documents_s3uri"]
        fake = mock.Mock()
        fake.start_execution.return_value = {"executionArn": "arn:fake"}
        ddb = mock.Mock()
        ddb.get_item.return_value = {"Item": {"collection_id": {"S": "default"}}}
        with mock.patch.object(self.gw, "_sfn", fake), \
             mock.patch.object(self.gw, "_ddb", ddb), \
             mock.patch.object(self.gw, "PROXY_BUCKET", "my-proxy-bucket"):
            out = self._call("index_content", alt)
        self.assertNotIn("error", out,
                         f"index_content alternative refused: {out}")
        self.assertEqual(out["status"], "RUNNING")
        self.assertEqual(out["documents_s3uri"], alt["documents_s3uri"])
        fake.start_execution.assert_called_once()

        # search_content's search_image_s3uri form: sync like the primary —
        # the handler starts the Search SFN and polls to completion
        alt = self.examples["search_content"]["alternatives"]["search_image_s3uri"]
        fake = mock.Mock()
        fake.start_execution.return_value = {"executionArn": "arn:fake"}
        fake.describe_execution.return_value = {
            "status": "SUCCEEDED",
            "output": json.dumps({"results": [], "total": 0})}
        with mock.patch.object(self.gw, "_sfn", fake):
            out = self._call("search_content", alt)
        self.assertNotIn("error", out,
                         f"search_content alternative refused: {out}")
        fake.start_execution.assert_called_once()
        sent = json.loads(fake.start_execution.call_args.kwargs["input"])
        self.assertEqual(sent["search_image_s3uri"], alt["search_image_s3uri"])
        self.assertEqual(sent["search_term"], "")


class TestIndexContentFailsFastOnUnknownCollection(unittest.TestCase):
    """index_content refuses an unknown collection AT SUBMIT with the same
    message the async path raises — measured live, the alternative was 22
    Step Functions executions each failing minutes later in prepare. The
    check is availability-safe: any DDB/transport error falls through and
    the async path still enforces it."""

    @classmethod
    def setUpClass(cls):
        cls.gw = _load_hs_gateway()

    def _submit(self, ddb, collection_id="news"):
        sfn = mock.Mock()
        sfn.start_execution.return_value = {"executionArn": "arn:fake"}
        event = {"collection_id": collection_id,
                 "documents": [{"asset_id": "a", "doc_type": "scene",
                                "start_ms": 0, "end_ms": 1,
                                "content": [{"kind": "text", "field": "vector",
                                             "value": "x"}]}]}
        with mock.patch.object(self.gw, "_sfn", sfn), \
             mock.patch.object(self.gw, "_ddb", ddb):
            out = self.gw.lambda_handler(
                event, _Ctx("hybrid-search___index_content"))
        return out, sfn

    def test_unknown_collection_refused_before_any_execution(self):
        ddb = mock.Mock()
        ddb.get_item.return_value = {}            # no Item → unknown
        out, sfn = self._submit(ddb)
        self.assertEqual(out.get("errorCode"), "UNKNOWN_COLLECTION")
        self.assertIn("Unknown collection 'news'", out["message"])
        self.assertIn("create_collection", out["message"])
        sfn.start_execution.assert_not_called()

    def test_known_collection_submits(self):
        ddb = mock.Mock()
        ddb.get_item.return_value = {"Item": {"collection_id": {"S": "news"}}}
        out, sfn = self._submit(ddb)
        self.assertEqual(out.get("status"), "RUNNING")
        sfn.start_execution.assert_called_once()
        ddb.get_item.assert_called_once()
        key = ddb.get_item.call_args.kwargs["Key"]
        self.assertEqual(key["collection_id"]["S"], "news")

    def test_empty_collection_id_checks_default(self):
        # omitted collection_id resolves to 'default' for the registry
        # check (and the error, if any, must say 'default', not '')
        ddb = mock.Mock()
        ddb.get_item.return_value = {"Item": {"collection_id": {"S": "default"}}}
        out, sfn = self._submit(ddb, collection_id="")
        self.assertEqual(out.get("status"), "RUNNING")
        key = ddb.get_item.call_args.kwargs["Key"]
        self.assertEqual(key["collection_id"]["S"], "default")
        ddb.get_item.return_value = {}
        out, _ = self._submit(ddb, collection_id="")
        self.assertEqual(out.get("errorCode"), "UNKNOWN_COLLECTION")
        self.assertIn("'default'", out["message"])

    def test_transport_error_also_falls_through(self):
        """The fail-open arm covers BotoCoreError too — NoCredentials or an
        endpoint outage on the registry read must not block an otherwise
        valid submit (narrowing the except back to ClientError alone is the
        regression this pins)."""
        from botocore.exceptions import EndpointConnectionError
        ddb = mock.Mock()
        ddb.get_item.side_effect = EndpointConnectionError(
            endpoint_url="https://dynamodb.test")
        out, sfn = self._submit(ddb)
        self.assertEqual(out.get("status"), "RUNNING")
        sfn.start_execution.assert_called_once()

    def test_registry_error_falls_through_to_submit(self):
        from botocore.exceptions import ClientError
        ddb = mock.Mock()
        ddb.get_item.side_effect = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "slow"}},
            "GetItem")
        out, sfn = self._submit(ddb)
        self.assertEqual(out.get("status"), "RUNNING")
        sfn.start_execution.assert_called_once()


class TestHandleLockRelease(unittest.TestCase):
    def test_extracts_all_lock_sks(self):
        from states.handle_lock_release.lambda_function import _released_lock_sk
        rec = {"eventName": "REMOVE",
               "dynamodb": {"Keys": {"pk": {"S": "lock"},
                                     "sk": {"S": "news:scene:m"}}}}
        self.assertEqual(_released_lock_sk(rec), "news:scene:m")
        self.assertIsNone(_released_lock_sk({**rec, "eventName": "INSERT"}))
        self.assertIsNone(_released_lock_sk(
            {"eventName": "REMOVE",
             "dynamodb": {"Keys": {"pk": {"S": "pending"},
                                   "sk": {"S": "x"}}}}))


# ─────────────────────────────────────────────────────────────────────────────
# SchemaDefinition — vector_store field, collection-scoped backend location
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_RAW = {
    "doc_type": "scene",
    "embedding_model": "cohere.embed-v4:0",
    "dims": 4,
    "language": "en",
    "mappings": {
        "doc_id":   {"type": "keyword", "required": True},
        "asset_id": {"type": "keyword", "required": True},
        "collection_id": {"type": "keyword"},
        "start_ms": {"type": "long", "required": True},
        "end_ms":   {"type": "long", "required": True},
        "text":     {"type": "text", "fts": True},
        "vector":   {"type": "float32_vector", "required": True, "ann": True},
    },
    "result_fields": ["text"],
}


class TestEmbedInvocationId(unittest.TestCase):
    """Bedrock serves some embed models ONLY through an inference
    profile (live incident 2026-09-03: indexing died with
    'Invocation of model ID cohere.embed-v4:0 with on-demand throughput
    isn't supported'). The LOGICAL model id must stay bare — it names
    the vector space and is baked into every LanceDB table path — and
    the geo profile is applied at invoke time only."""

    def test_profile_only_model_maps_and_others_pass_through(self):
        # global by default (user decision 2026-09-03): the same
        # invocation id works from every region — the incident was
        # reported from us-west-2 AND eu-west-1, and global.cohere
        # verified live from both. Residency-constrained deployments
        # override EMBED_INFERENCE_PROFILE_PREFIX to their geography.
        from lib import bedrock_utils as bu
        self.assertEqual(bu.invocation_id("cohere.embed-v4:0"),
                         "global.cohere.embed-v4:0")
        self.assertEqual(bu.invocation_id("amazon.titan-embed-image-v1"),
                         "amazon.titan-embed-image-v1")
        self.assertEqual(
            bu.invocation_id("amazon.nova-2-multimodal-embeddings-v1:0"),
            "amazon.nova-2-multimodal-embeddings-v1:0")

    def test_table_paths_never_carry_the_profile(self):
        # the schema/table identity keeps the bare id: a profile prefix
        # leaking into table_key would orphan every existing index
        from lib.backends.lancedb_backend import SchemaDefinition
        sd = SchemaDefinition("s3://b/schema.json", SCHEMA_RAW)
        self.assertNotIn("eu.", sd.table_key)


class TestSchemaAndBackend(unittest.TestCase):
    def test_vector_store_default_and_validation(self):
        from lib.backends.lancedb_backend import SchemaDefinition
        sd = SchemaDefinition("s3://b/schema.json", SCHEMA_RAW)
        self.assertEqual(sd.vector_store, "lancedb")
        sd2 = SchemaDefinition("s3://b/schema.json",
                               {**SCHEMA_RAW, "vector_store": "s3vectors"})
        self.assertEqual(sd2.vector_store, "s3vectors")
        with self.assertRaises(ValueError):
            SchemaDefinition("s3://b/schema.json",
                             {**SCHEMA_RAW, "vector_store": "chroma"})

    def test_backend_location_is_collection_scoped(self):
        from lib.backends.lancedb_backend import LanceDBBackend, SchemaDefinition
        sd = SchemaDefinition("s3://b/schema.json", SCHEMA_RAW)
        b = LanceDBBackend(sd, "news", bucket="test-proxy")
        self.assertEqual(
            b._s3_uri,
            "s3://test-proxy/_lancedb_index/news/scene/cohere.embed-v4:0/4/en/")
        self.assertEqual(b.index_key, "news:scene:cohere.embed-v4:0:en")
        with self.assertRaises(ValueError):
            LanceDBBackend(sd, "", bucket="test-proxy")


# ─────────────────────────────────────────────────────────────────────────────
# StatePrepareDocuments — grouping, dedup-before-embed, entity flow
# ─────────────────────────────────────────────────────────────────────────────

class TestPrepareDocuments(unittest.TestCase):
    def _run(self, event, already_indexed=frozenset(), collection=None,
             puts=None, embed_fn=None, image_fn=None, head_fn=None):
        import states.prepare_documents.lambda_function as mod
        from lib.backends.lancedb_backend import SchemaDefinition
        collection = collection or {
            "collection_id": "news",
            "schema_map": {"scene": "_lancedb_index/schema/default_scene.json"},
        }
        # puts doubles as the fake S3 store: put_json writes into it and
        # _load_checkpoint reads from it, so retry attempts see checkpoints.
        puts = {} if puts is None else puts
        embeds = []

        def _default_embed(m, text=None, image_b64=None, dims=None, usage=None):
            embeds.append((text, image_b64))
            return [0.0] * 4

        with mock.patch.object(mod, "resolve_collection", return_value=collection), \
             mock.patch.object(mod, "load_schema",
                               return_value=SchemaDefinition("s3://b/s.json", SCHEMA_RAW)), \
             mock.patch.object(mod, "batch_is_indexed",
                               side_effect=lambda ids: {i for i in ids if i in already_indexed}), \
             mock.patch.object(mod, "embed_document",
                               side_effect=(embed_fn or _default_embed)), \
             mock.patch.object(mod, "get_image_base64",
                               side_effect=(image_fn or (lambda b, k: "b64"))), \
             mock.patch.object(mod, "object_size",
                               side_effect=(head_fn or (lambda b, k: 1))), \
             mock.patch.object(mod, "put_json",
                               side_effect=lambda b, k, d: puts.update({k: json.loads(json.dumps(d))})), \
             mock.patch.object(mod, "_load_checkpoint",
                               side_effect=lambda k: json.loads(json.dumps(puts.get(k, {})))), \
             mock.patch.object(mod, "upsert_asset_metadata") as meta:
            result = mod.lambda_handler(event, None)
        return result, puts, embeds, meta

    def _event(self, docs):
        return {"collection_id": "news", "documents": docs,
                "documents_s3uri": "", "request_id": "req1"}

    def _doc(self, **over):
        base = {"asset_id": "asset:uuid:a", "doc_type": "scene",
                "start_ms": 0, "end_ms": 5000,
                "content": [
                    {"kind": "text", "field": "text", "value": "hello world"},
                    {"kind": "text", "field": "vector", "value": "hello world"},
                ]}
        base.update(over)
        return base

    def test_prepare_groups_and_embeds(self):
        result, puts, embeds, _ = self._run(self._event([self._doc()]))
        self.assertEqual(result["prepared"], 1)
        self.assertEqual(len(result["map_items"]), 1)
        item = result["map_items"][0]
        self.assertEqual(item["index_key"], "news:scene:cohere.embed-v4:0:en")
        rows = puts[item["rows_s3_key"]]["documents"]
        self.assertEqual(rows[0]["collection_id"], "news")
        self.assertEqual(len(rows[0]["vector"]), 4)
        self.assertEqual(len(embeds), 1)  # one embed call for the ann item

    def test_dedup_before_embedding(self):
        from lib.contracts import compute_doc_id
        did = compute_doc_id("news", "asset:uuid:a", "scene", 0, 5000)
        result, puts, embeds, _ = self._run(
            self._event([self._doc()]), already_indexed={did})
        self.assertEqual(result["prepared"], 0)
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(embeds, [])   # NO Bedrock spend on deduped docs
        self.assertEqual(puts, {})

    def test_unknown_scope_fails_hard(self):
        with self.assertRaises(ValueError):
            self._run(self._event([self._doc(doc_type="podcast")]))

    def test_bad_image_uri_refused_before_any_embed(self):
        """Refuse-before-spend: a wrong image path fails the request on the
        pre-embed HEAD sweep — zero Bedrock spend, ONE error naming every bad
        document and key (measured live: one guessed grid path failed a 93-doc
        scene batch after its text siblings had already embedded)."""
        from botocore.exceptions import ClientError
        bad_uri = "s3://test-proxy/micro-workflow/shots/nope/shot_000001.jpg"
        doc = self._doc(content=[
            {"kind": "text", "field": "text", "value": "anchor at desk"},
            {"kind": "image", "field": "vector", "value": bad_uri},
        ])

        def head_404(bucket, key):
            raise ClientError({"Error": {"Code": "404", "Message": "x"}},
                              "HeadObject")

        embeds = []
        def counting_embed(m, text=None, image_b64=None, dims=None, usage=None):
            embeds.append(1); return [0.0] * 4
        with self.assertRaises(ValueError) as cm:
            self._run(self._event([doc]), head_fn=head_404,
                      embed_fn=counting_embed)
        msg = str(cm.exception)
        self.assertIn(bad_uri, msg)
        self.assertIn("404", msg)
        self.assertEqual(embeds, [])   # zero model spend before the refusal

    def test_resume_never_rejudges_checkpointed_images(self):
        """The sweep HEADs PENDING docs' images only: a grid image deleted
        AFTER its document's rows were paid for and checkpointed must not
        kill the redrive the checkpoint exists to protect."""
        from botocore.exceptions import ClientError
        docs = [self._doc(start_ms=i * 1000, end_ms=i * 1000 + 900,
                          content=[
                              {"kind": "text", "field": "text", "value": "x"},
                              {"kind": "image", "field": "vector",
                               "value": f"s3://test-proxy/img_{i}.jpg"},
                          ]) for i in range(5)]
        puts: dict = {}
        calls = {"n": 0}

        def throttle_on_third(m, text=None, image_b64=None, dims=None,
                              usage=None):
            calls["n"] += 1
            if calls["n"] == 3:
                raise ClientError({"Error": {"Code": "ThrottlingException",
                                             "Message": "x"}}, "InvokeModel")
            return [0.0] * 4

        # attempt 1: docs 0-1 embed, doc 2 throttles -> partial checkpoint
        with self.assertRaises(ClientError):
            self._run(self._event(docs), puts=puts,
                      embed_fn=throttle_on_third)
        self.assertTrue(any(v.get("partial") for v in puts.values()))

        # between attempts, the PAID docs' images vanish
        def gone_for_paid(bucket, key):
            if key in ("img_0.jpg", "img_1.jpg"):
                raise ClientError({"Error": {"Code": "404", "Message": "x"}},
                                  "HeadObject")
            return 1

        # attempt 2 (same request_id via the shared puts store) must resume
        # cleanly: the sweep judges only docs 2-4
        result, _, embeds2, _ = self._run(self._event(docs), puts=puts,
                                          head_fn=gone_for_paid)
        self.assertEqual(result["prepared"], 5)
        self.assertEqual(len(embeds2), 3)   # only the pending docs embed

    def test_sweep_aggregates_every_bad_image_into_one_error(self):
        """The sweep's point versus the old one-at-a-time failure: ALL bad
        documents are named in ONE error (capped at 5 with a '+N more'
        tally), so the caller fixes the batch in one pass instead of
        resubmitting once per discovery."""
        from botocore.exceptions import ClientError
        docs = [self._doc(start_ms=i * 1000, end_ms=i * 1000 + 900,
                          content=[
                              {"kind": "text", "field": "text", "value": "x"},
                              {"kind": "image", "field": "vector",
                               "value": f"s3://test-proxy/missing_{i}.jpg"},
                          ]) for i in range(7)]

        def head_404(bucket, key):
            raise ClientError({"Error": {"Code": "404", "Message": "x"}},
                              "HeadObject")

        with self.assertRaises(ValueError) as cm:
            self._run(self._event(docs), head_fn=head_404)
        msg = str(cm.exception)
        self.assertEqual(sum(1 for i in range(7)
                             if f"missing_{i}.jpg" in msg), 5)
        self.assertIn("(+2 more)", msg)

    def test_empty_key_image_uri_is_refused_as_invalid(self):
        """'s3://bucket/' passes validate_contract (prefix check only) but
        carries no object key — botocore would raise ParamValidationError (a
        BotoCoreError the transient-tolerant paths swallow), so parse_s3_uri
        now rejects it and the sweep reports it as a named permanent
        failure, before any embed."""
        doc = self._doc(content=[
            {"kind": "text", "field": "text", "value": "anchor at desk"},
            {"kind": "image", "field": "vector", "value": "s3://test-proxy/"},
        ])
        embeds = []
        def counting_embed(m, text=None, image_b64=None, dims=None, usage=None):
            embeds.append(1); return [0.0] * 4
        with self.assertRaises(ValueError) as cm:
            self._run(self._event([doc]), embed_fn=counting_embed)
        self.assertIn("invalid s3 uri", str(cm.exception))
        self.assertIn("s3://test-proxy/", str(cm.exception))
        self.assertEqual(embeds, [])

    def test_transient_head_failure_does_not_block_the_sweep(self):
        """A connection blip on the HEAD sweep (BotoCoreError, not
        ClientError) must be ignored — the fetch path owns transient
        handling; the sweep only ever refuses on PERMANENT codes."""
        from botocore.exceptions import EndpointConnectionError
        doc = self._doc(content=[
            {"kind": "text", "field": "text", "value": "anchor at desk"},
            {"kind": "image", "field": "vector", "value": "s3://test-proxy/x.jpg"},
        ])

        def flaky_head(bucket, key):
            raise EndpointConnectionError(endpoint_url="https://s3.test")

        result, _, embeds, _ = self._run(self._event([doc]),
                                         head_fn=flaky_head)
        self.assertEqual(result["prepared"], 1)
        self.assertEqual(len(embeds), 1)   # the run proceeded to embed

    def test_missing_image_names_the_document_and_key(self):
        """The fetch-time wrap (the race backstop when HEAD succeeded but the
        object vanished before GET): a permanent image-fetch failure
        (NoSuchKey/404/AccessDenied) must fail TERMINALLY and name the doc_id
        and the s3 uri — a bare NoSuchKey bubbling out of the group gives the
        caller nothing to fix."""
        from botocore.exceptions import ClientError
        bad_uri = "s3://test-proxy/micro-workflow/shots/nope/shot_000001.jpg"
        doc = self._doc(content=[
            {"kind": "text", "field": "text", "value": "anchor at desk"},
            {"kind": "image", "field": "vector", "value": bad_uri},
        ])

        def nosuchkey(bucket, key):
            raise ClientError({"Error": {"Code": "NoSuchKey",
                                         "Message": "The specified key does not exist."}},
                              "GetObject")

        puts: dict = {}
        with self.assertRaises(ValueError) as cm:
            self._run(self._event([doc]), image_fn=nosuchkey, puts=puts)
        msg = str(cm.exception)
        self.assertIn(bad_uri, msg)
        self.assertIn("NoSuchKey", msg)
        from lib.contracts import compute_doc_id
        self.assertIn(compute_doc_id("news", "asset:uuid:a", "scene", 0, 5000),
                      msg)
        # rows embedded before the failing doc are checkpointed — the error
        # is terminal, but the redrive after fixing the uri must not re-pay
        self.assertTrue(any(v.get("partial") for v in puts.values()),
                        f"no partial checkpoint on the ValueError path: "
                        f"{list(puts)}")

    def test_transient_image_error_keeps_checkpoint_semantics(self):
        """Only PERMANENT codes convert to the terminal ValueError; a throttle
        on the image read must stay a ClientError so the group checkpoints —
        a REDRIVE (S3 transients surface as errorType 'ClientError', which no
        ASL Retry rule matches) resumes without re-embedding."""
        from botocore.exceptions import ClientError
        doc = self._doc(content=[
            {"kind": "text", "field": "text", "value": "anchor at desk"},
            {"kind": "image", "field": "vector", "value": "s3://test-proxy/x.jpg"},
        ])

        def throttled(bucket, key):
            raise ClientError({"Error": {"Code": "SlowDown",
                                         "Message": "Reduce your request rate."}},
                              "GetObject")

        puts: dict = {}
        with self.assertRaises(ClientError):
            self._run(self._event([doc]), image_fn=throttled, puts=puts)
        # the named semantics: completed rows persisted as a PARTIAL
        # checkpoint so a redrive resumes instead of re-embedding
        self.assertTrue(any(v.get("partial") for v in puts.values()),
                        f"no partial checkpoint written: {list(puts)}")

    def test_checkpoint_resume_after_throttle(self):
        # Attempt 1: throttle on the 3rd embed -> partial checkpoint + raise.
        # Attempt 2 (same request_id, same fake S3): resumes after doc 2,
        # embeds ONLY the remaining docs, clears the partial marker.
        from botocore.exceptions import ClientError
        docs  = [self._doc(start_ms=i * 1000, end_ms=i * 1000 + 900)
                 for i in range(5)]
        puts: dict = {}
        calls = {"n": 0}

        def throttling_embed(m, text=None, image_b64=None, dims=None, usage=None):
            calls["n"] += 1
            if calls["n"] == 3:
                raise ClientError(
                    {"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
                    "InvokeModel")
            if usage is not None:
                u = usage.setdefault(m, {"invocations": 0})
                u["invocations"] += 1
            return [0.0] * 4

        with self.assertRaises(ClientError):
            self._run(self._event(docs), puts=puts, embed_fn=throttling_embed)
        (ckpt,) = puts.values()
        self.assertTrue(ckpt["partial"])
        self.assertEqual(len(ckpt["documents"]), 2)   # docs 1-2 checkpointed
        self.assertEqual(ckpt["embed_usage"]["cohere.embed-v4:0"]["invocations"], 2)

        result, puts, _, _ = self._run(self._event(docs), puts=puts,
                                       embed_fn=throttling_embed)
        self.assertEqual(result["prepared"], 5)
        self.assertEqual(calls["n"], 3 + 3)           # only 3 re-embeds, not 5
        (final,) = puts.values()
        self.assertNotIn("partial", final)
        self.assertEqual(len(final["documents"]), 5)
        # spend across BOTH attempts is reported, never double-counted
        self.assertEqual(result["embed_usage"]["cohere.embed-v4:0"]["invocations"], 5)

    def test_checkpoint_resume_after_non_throttle_client_error(self):
        # A transient Bedrock error OTHER than the two throttle codes (e.g.
        # InternalServerException) must still checkpoint — narrowing to
        # specific codes previously lost every row in the group.
        from botocore.exceptions import ClientError
        docs = [self._doc(start_ms=i * 1000, end_ms=i * 1000 + 900)
                for i in range(4)]
        puts: dict = {}
        calls = {"n": 0}

        def erroring_embed(m, text=None, image_b64=None, dims=None, usage=None):
            calls["n"] += 1
            if calls["n"] == 2:
                raise ClientError(
                    {"Error": {"Code": "InternalServerException", "Message": "oops"}},
                    "InvokeModel")
            return [0.0] * 4

        with self.assertRaises(ClientError):
            self._run(self._event(docs), puts=puts, embed_fn=erroring_embed)
        (ckpt,) = puts.values()
        self.assertTrue(ckpt["partial"])
        self.assertEqual(len(ckpt["documents"]), 1)   # doc 1 checkpointed

        result, _, _, _ = self._run(self._event(docs), puts=puts,
                                    embed_fn=erroring_embed)
        self.assertEqual(result["prepared"], 4)
        self.assertEqual(calls["n"], 2 + 3)           # no re-embed of doc 1

    def test_time_budget_checkpoints_and_resumes(self):
        # Attempt 1 runs with a fake context reporting <90s remaining after
        # 2 docs: checkpoint + LambdaResourceTimeoutException. Attempt 2
        # (fresh budget) finishes the remaining docs without re-embedding.
        import states.prepare_documents.lambda_function as mod
        docs = [self._doc(start_ms=i * 1000, end_ms=i * 1000 + 900)
                for i in range(5)]
        puts: dict = {}
        calls = {"n": 0}

        class Ctx:
            def get_remaining_time_in_millis(self):
                # plenty of time for docs 1-2, then out of budget
                return 900_000 if calls["n"] < 2 else 10_000

        def counting_embed(m, text=None, image_b64=None, dims=None, usage=None):
            calls["n"] += 1
            return [0.0] * 4

        from lib.custom_exceptions import LambdaResourceTimeoutException
        orig = mod.lambda_handler
        with self.assertRaises(LambdaResourceTimeoutException):
            # _run passes context=None; call the handler directly with Ctx
            from lib.backends.lancedb_backend import SchemaDefinition
            with mock.patch.object(mod, "resolve_collection", return_value={
                    "collection_id": "news",
                    "schema_map": {"scene": "_lancedb_index/schema/default_scene.json"}}), \
                 mock.patch.object(mod, "load_schema",
                                   return_value=SchemaDefinition("s3://b/s.json", SCHEMA_RAW)), \
                 mock.patch.object(mod, "batch_is_indexed", return_value=set()), \
                 mock.patch.object(mod, "embed_document", side_effect=counting_embed), \
                 mock.patch.object(mod, "put_json",
                                   side_effect=lambda b, k, d: puts.update({k: json.loads(json.dumps(d))})), \
                 mock.patch.object(mod, "_load_checkpoint",
                                   side_effect=lambda k: json.loads(json.dumps(puts.get(k, {})))), \
                 mock.patch.object(mod, "upsert_asset_metadata"):
                orig(self._event(docs), Ctx())
        (ckpt,) = puts.values()
        self.assertTrue(ckpt["partial"])
        self.assertEqual(len(ckpt["documents"]), 2)

        result, _, _, _ = self._run(self._event(docs), puts=puts,
                                    embed_fn=counting_embed)
        self.assertEqual(result["prepared"], 5)
        self.assertEqual(calls["n"], 5)   # 2 + 3, no re-embeds

    def test_checkpoint_reuses_complete_group(self):
        # A fully-written group from a prior attempt is reused whole —
        # zero embed calls on re-entry (tier 1).
        docs = [self._doc()]
        result, puts, embeds, _ = self._run(self._event(docs))
        self.assertEqual(len(embeds), 1)
        result, _, embeds, _ = self._run(self._event(docs), puts=puts)
        self.assertEqual(result["prepared"], 1)
        self.assertEqual(embeds, [])                  # no Bedrock spend

    def test_inline_cap(self):
        import states.prepare_documents.lambda_function as mod
        docs = [self._doc(start_ms=i) for i in range(101)]
        with self.assertRaises(ValueError):
            mod._load_contracts({"documents": docs, "documents_s3uri": ""})

    def test_exactly_one_input_mode(self):
        import states.prepare_documents.lambda_function as mod
        for ev in ({"documents": [], "documents_s3uri": ""},
                   {"documents": [self._doc()], "documents_s3uri": "s3://b/m.jsonl"}):
            with self.assertRaises(ValueError):
                mod._load_contracts(ev)


# ─────────────────────────────────────────────────────────────────────────────
# StateDecomposeSearchRequest — scope restriction vs path classification
#
# The live trap (2026-09-03): '#utterance' restricted the fan-out to one
# index while the SLM classified the query visual, whose paths give every
# utterance sub-query weight 0 — the intersection was a SILENT empty result.
# The contract now: a scope restriction guarantees the scope is searched
# (weights backfilled from the collection's own configured paths), and any
# scope that still cannot run is REPORTED via dropped_scopes ->
# skipped_indices, never silently absent.
# ─────────────────────────────────────────────────────────────────────────────

# Mirrors the SHAPE (not the values) of workflow_config retrievalPaths;
# the sub-query keys are never hand-written — they are derived from these
# schema defs by the production _derive_subquery_spec in setUp, so a
# key-convention change breaks these tests the same way it breaks the
# collection's configured paths.
_SCHEMAS = {
    "shot":      {"doc_type": "shot", "embedding_model": "m", "dims": 4,
                  "mappings": {"vector": {"type": "float32_vector",
                                          "ann": True}}},
    "scene":     {"doc_type": "scene", "embedding_model": "m", "dims": 4,
                  "mappings": {"vector": {"type": "float32_vector",
                                          "ann": True}}},
    "utterance": {"doc_type": "utterance", "embedding_model": "m", "dims": 4,
                  "language": "en",
                  "mappings": {"vector": {"type": "float32_vector",
                                          "ann": True},
                               "text": {"type": "text", "fts": True}}},
}
_PATHS = {
    "visual":   {"shot_ann": 0.7, "scene_ann": 0.5},
    "dialogue": {"utterance_text_fts": 0.8, "utterance_ann": 0.4},
}


class TestScopeRestriction(unittest.TestCase):
    def setUp(self):
        import states.decompose_search_request.lambda_function as mod
        self.mod = mod
        self.subq = mod._derive_subquery_spec(_SCHEMAS)
        # pins the derived key convention to the keys _PATHS is written in
        self.assertEqual(set(self.subq),
                         {"shot_ann", "scene_ann", "utterance_ann",
                          "utterance_text_fts"})

    def test_parse_operators_extracts_scope_tokens(self):
        op = self.mod._parse_operators("burn the coast #Utterance #scene")
        self.assertEqual(op["index_scopes"], ["utterance", "scene"])
        self.assertEqual(op["clean_query"], "burn the coast")

    def test_slm_scopes_are_lowercased(self):
        slm = self.mod._parse_slm_xml(
            "<index_scopes>Utterance, SCENE</index_scopes>")
        self.assertEqual(slm["index_scopes"], ["utterance", "scene"])

    def test_backfill_gives_restricted_scope_its_configured_weights(self):
        # classifier said visual; caller said #utterance
        w = self.mod._compute_weights(["visual"], _PATHS)
        self.assertNotIn("utterance_ann", w)
        out = self.mod._backfill_scope_weights(w, ["utterance"],
                                               self.subq, _PATHS)
        self.assertEqual(out["utterance_ann"], 0.4)
        self.assertEqual(out["utterance_text_fts"], 0.8)
        # the classifier's own choices are untouched
        self.assertEqual(out["shot_ann"], 0.7)

    def test_backfill_is_pure(self):
        w = self.mod._compute_weights(["visual"], _PATHS)
        before = dict(w)
        self.mod._backfill_scope_weights(w, ["utterance"], self.subq, _PATHS)
        self.assertEqual(w, before)

    def test_backfill_never_overrides_a_chosen_weight(self):
        w = self.mod._compute_weights(["dialogue"], _PATHS)
        out = self.mod._backfill_scope_weights(w, ["utterance"],
                                               self.subq, _PATHS)
        self.assertEqual(out, w)

    def _items(self, **over):
        kw = dict(active_scopes=["utterance"],
                  weights={"utterance_ann": 0.4, "utterance_text_fts": 0.8},
                  has_vectors={"utterance"},
                  vectors_s3_key="_search/r/vectors.json",
                  asset_ids=[], k=10, candidate_k=50, entities=[],
                  dialog_query="", quoted_phrase="",
                  schema_map={"utterance": "schema/u.json"},
                  state_records={"utterance": {"sk": "default:utterance:m"}},
                  subquery_spec=self.subq, collection_id="default")
        kw.update(over)
        return self.mod._build_query_items(**kw)

    def test_restricted_plain_text_query_still_searches_ann(self):
        # the regression: with backfilled weights, '#utterance' + plain text
        # runs the scope's ANN even though FTS has no says:/quoted text
        items, dropped = self._items()
        self.assertEqual(dropped, [])
        self.assertEqual(len(items), 1)
        self.assertEqual([s["type"] for s in items[0]["sub_queries"]],
                         ["ann"])

    def test_zero_weights_drop_is_reported(self):
        items, dropped = self._items(weights={})
        self.assertEqual(items, [])
        self.assertEqual(dropped, [{"index_scope": "utterance",
                                    "skip_reason": "no_subqueries"}])

    def test_missing_state_record_is_reported(self):
        items, dropped = self._items(state_records={})
        self.assertEqual(items, [])
        self.assertEqual(dropped, [{"index_scope": "utterance",
                                    "skip_reason": "no_state_record"}])

    # ── the real call site: guard, backfill, and reporting must agree ──────
    def _decompose(self, search_term, slm_xml):
        mod = self.mod
        schema_map = {s: f"schema/{s}.json" for s in _SCHEMAS}
        with mock.patch.object(mod, "get_object_text",
                               return_value="# System Prompt\nsys\n"
                                            "# User Prompt\n{{search_query}}"), \
             mock.patch.object(mod, "call_slm_with_fallback",
                               return_value=(slm_xml, "m", False, None)), \
             mock.patch.object(mod, "embed", return_value=[0.0] * 4), \
             mock.patch.object(
                 mod, "get_json",
                 side_effect=lambda b, k: _SCHEMAS[k.split("/")[-1][:-5]]), \
             mock.patch.object(mod, "get_table_state",
                               side_effect=lambda ik: {"sk": ik}), \
             mock.patch.object(mod, "put_json"):
            return mod._decompose_text_search(
                search_term=search_term,
                workflow_cfg={"primaryModel": "m", "promptS3Key": "p.md"},
                collection={"collection_id": "default",
                            "retrieval_paths": _PATHS},
                all_schemas=list(_SCHEMAS),
                schema_map=schema_map,
                request_id="r", k=10, candidate_k=50)

    _VISUAL_XML = "<retrieval_paths>visual</retrieval_paths>"

    def test_call_site_user_scope_is_searched_despite_visual_paths(self):
        d = self._decompose("burn the coast #utterance", self._VISUAL_XML)
        self.assertEqual([q["index_scope"] for q in d["queries"]],
                         ["utterance"])
        self.assertEqual(d["queries"][0]["sub_queries"],
                         [{"type": "ann", "field": "vector", "weight": 0.4}])
        self.assertEqual(d["dropped_scopes"], [])

    def test_call_site_unrestricted_reports_no_drops(self):
        d = self._decompose("burn the coast", self._VISUAL_XML)
        self.assertEqual(sorted(q["index_scope"] for q in d["queries"]),
                         ["scene", "shot"])
        # utterance was merely de-emphasized by the classifier — that is
        # normal, not "never searched" noise
        self.assertEqual(d["dropped_scopes"], [])

    def test_call_site_unknown_scope_is_reported_not_silent(self):
        d = self._decompose("burn the coast #utterence", self._VISUAL_XML)
        self.assertEqual(d["queries"], [])
        self.assertEqual(d["dropped_scopes"],
                         [{"index_scope": "utterence",
                           "skip_reason": "unknown_scope"}])

    def test_call_site_slm_scopes_narrow_and_report_but_never_backfill(self):
        xml = ("<retrieval_paths>visual</retrieval_paths>"
               "<index_scopes>Utterance</index_scopes>")
        d = self._decompose("burn the coast", xml)
        # no user # token: the SLM restriction narrows the fan-out and its
        # zero-weight drop is REPORTED, but weights are never backfilled —
        # a hallucinated scope list must not neuter path classification
        self.assertEqual(d["queries"], [])
        self.assertEqual(d["dropped_scopes"],
                         [{"index_scope": "utterance",
                           "skip_reason": "no_subqueries"}])

    def test_rerank_surfaces_dropped_scopes_in_skipped_indices(self):
        import states.rerank_results.lambda_function as rr
        resp = rr.lambda_handler({
            "search_term": "burn the coast #utterance", "k": 10,
            "decomposition": {
                "collection_id": "default", "request_id": "",
                "vectors_s3_key": "", "queries": [],
                "dropped_scopes": [{"index_scope": "utterance",
                                    "skip_reason": "no_subqueries"}]},
            "search_results": [
                {"index_scope": "topic", "skipped": True, "results": []}],
        }, None)
        self.assertEqual(resp["total"], 0)
        self.assertEqual(sorted(resp["skipped_indices"]),
                         ["topic", "utterance"])


# ─────────────────────────────────────────────────────────────────────────────
# cfnstack.yaml — template invariants
#
# These are checked by PARSING the template, never by grepping the raw text:
# the template deliberately DOCUMENTS the things it omits (logs:CreateLogGroup,
# the non-existent !GetAtt ...VectorBucketName), so a text search for an
# "absent" property matches the comment explaining its absence and the test
# fails on the template's own documentation.
#
# PyYAML is a TEST-ONLY dependency (test/requirements.txt) — it is deliberately
# absent from the container's requirements.txt. Missing, these tests SKIP rather
# than take the other 28 down with them at import time.
# ─────────────────────────────────────────────────────────────────────────────

class _CfnLoader(yaml.SafeLoader if yaml else object):
    """SafeLoader that tolerates CFN short-form intrinsics (!Sub, !GetAtt, ...).

    Tags are preserved as ("!Tag", value) tuples so a test can assert on WHICH
    intrinsic was used, not merely on the resolved string.
    """


def _cfn_multi(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return ("!" + suffix, loader.construct_scalar(node))
    if isinstance(node, yaml.SequenceNode):
        return ("!" + suffix, loader.construct_sequence(node, deep=True))
    return ("!" + suffix, loader.construct_mapping(node, deep=True))


if yaml:
    _CfnLoader.add_multi_constructor("!", _cfn_multi)


@unittest.skipIf(yaml is None,
                 "PyYAML not installed — pip install -r test/requirements.txt")
class TestCfnTemplate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, "cfnstack.yaml")) as f:
            cls.tpl = yaml.load(f, Loader=_CfnLoader)  # nosec B506
        cls.res = cls.tpl["Resources"]

    def _of_type(self, cfn_type):
        return {k: v for k, v in self.res.items() if v.get("Type") == cfn_type}

    @staticmethod
    def _sub_body(node):
        """The format string out of a !Sub, in either form: a bare scalar, or
        [body, {substitution map}]."""
        _, val = node
        return val if isinstance(val, str) else val[0]

    def _depends_on(self, logical_id):
        dep = self.res[logical_id].get("DependsOn")
        return [dep] if isinstance(dep, str) else (dep or [])

    # ── log group coverage ──────────────────────────────────────────────────

    # The five functions whose log group is the canonical
    # /aws/lambda/{FunctionName} and must STAY that way: those groups are
    # already stack-owned in deployed stacks, and LogGroupName is a
    # createOnlyProperty, so renaming one REPLACES it and CFN deletes the old
    # group with every log in it (measured). See the template banner.
    _CANONICAL_GROUP_LAMBDAS = {
        "IndexerLambda", "RebuildLambda", "SearchLambda",
        "StreamHandlerLambda", "HandleExecutionFailureLambda",
    }

    def test_every_lambda_has_a_declared_log_group(self):
        """No Lambda may rely on an auto-created group.

        Auto-created groups have retentionInDays=None (kept and billed
        forever) and survive stack deletion.
        """
        groups = self._of_type("AWS::Logs::LogGroup")
        self.assertTrue(groups, "no log groups declared at all")
        for name, node in self._of_type("AWS::Lambda::Function").items():
            deps = self._depends_on(name)
            self.assertTrue(
                any(d in groups for d in deps),
                f"{name} does not DependsOn a declared log group — Lambda's "
                f"first invocation will race CFN and create an unmanaged one")

    def test_lambdas_outside_the_canonical_five_use_logging_config(self):
        """LoggingConfig decouples the group name from the function name.

        Needed for two different reasons, hence one rule: StartBuild and
        CopySchema are unnamed (CFN generates the name, so /aws/lambda/{name}
        is unknowable at author time), while GatewayTool and AdminTool are
        named but their canonical groups ALREADY EXIST — auto-created by Lambda
        before this template declared them, which fails the update at
        pre-flight validation with no per-resource event naming the culprit.
        """
        groups = self._of_type("AWS::Logs::LogGroup")
        for name, node in self._of_type("AWS::Lambda::Function").items():
            if name in self._CANONICAL_GROUP_LAMBDAS:
                continue
            props = node["Properties"]
            deps = self._depends_on(name)
            self.assertIn(
                "LoggingConfig", props,
                f"{name} must reach its group via LoggingConfig")
            self.assertEqual(
                props["LoggingConfig"].get("LogGroup"),
                ("!Ref", next(d for d in deps if d in groups)),
                f"{name}'s LoggingConfig must !Ref the group it DependsOn")

    def test_logging_config_groups_are_not_the_canonical_name(self):
        """A LoggingConfig group named /aws/lambda/{FunctionName} defeats the
        purpose: that name is exactly what collides with the auto-created group
        this pattern exists to sidestep. Caught a real update failure on m2cv4.
        """
        groups = self._of_type("AWS::Logs::LogGroup")
        for name, node in self._of_type("AWS::Lambda::Function").items():
            props = node["Properties"]
            if "LoggingConfig" not in props or "FunctionName" not in props:
                continue
            fn_body = self._sub_body(props["FunctionName"])
            grp = props["LoggingConfig"]["LogGroup"][1]
            declared = self._sub_body(groups[grp]["Properties"]["LogGroupName"])
            self.assertNotEqual(
                declared, f"/aws/lambda/{fn_body}",
                f"{name}'s declared group is the canonical name, which will "
                f"collide with the pre-existing auto-created group")

    def test_the_canonical_five_keep_their_group_names(self):
        """Renaming these deletes production logs — LogGroupName is
        createOnly, so a rename is a REPLACEMENT and the old group goes with
        it. Measured on a throwaway stack: nothing is left behind.
        """
        for name in self._CANONICAL_GROUP_LAMBDAS:
            # These use the !Sub [body, map] form, so compare the BODY: the
            # group's must be exactly "/aws/lambda/" + the function's.
            fn_body = self._sub_body(self.res[name]["Properties"]["FunctionName"])
            grp = self.res[name.replace("Lambda", "LambdaLogGroup")]
            grp_body = self._sub_body(grp["Properties"]["LogGroupName"])
            self.assertEqual(
                grp_body, f"/aws/lambda/{fn_body}",
                f"{name}'s group must keep the canonical name")
            self.assertNotIn(
                "LoggingConfig", self.res[name]["Properties"],
                f"{name} needs no LoggingConfig — its group IS the canonical "
                f"name, and adding one invites a rename that deletes logs")

    def test_every_state_machine_logs(self):
        machines = self._of_type("AWS::StepFunctions::StateMachine")
        self.assertTrue(machines, "no state machines found")
        for name, node in machines.items():
            self.assertIn("LoggingConfiguration", node["Properties"],
                          f"{name} has no LoggingConfiguration")

    def test_no_role_may_create_its_own_log_group(self):
        """logs:CreateLogGroup defeats the point of declaring the group.

        With the permission, a function that cannot reach its declared group
        silently creates an unmanaged one instead of failing visibly. CodeBuild
        is the one legitimate exception — the service creates its own group.
        """
        for name, node in self._of_type("AWS::IAM::Role").items():
            for policy in node["Properties"].get("Policies", []):
                for stmt in policy["PolicyDocument"]["Statement"]:
                    actions = stmt.get("Action", [])
                    actions = [actions] if isinstance(actions, str) else actions
                    if "logs:CreateLogGroup" not in actions:
                        continue
                    resource = json.dumps(stmt.get("Resource", ""))
                    self.assertIn(
                        "codebuild", resource,
                        f"{name} grants logs:CreateLogGroup on {resource}")

    # ── vector bucket ───────────────────────────────────────────────────────

    def test_vector_bucket_is_the_native_type_and_retained(self):
        """AWS::S3Vectors::VectorBucket, with BOTH retain policies.

        Re-embedding a corpus costs real Bedrock spend, so the bucket
        deliberately outlives the stack. The reason a plain native resource is
        safe here — despite the type being unable to ADOPT an existing bucket
        (a same-named bucket fails the create with 409 AlreadyExists and rolls
        back the whole stack) — is that ResourcePrefix is derived from the
        STACK ID in both deployment paths. A delete + re-create therefore
        produces a new stack id, a new bucket name, and a retained orphan; it
        can never collide with the bucket it just retained.
        """
        bucket = self.res["HybridSearchVectorBucket"]
        self.assertEqual(bucket["Type"], "AWS::S3Vectors::VectorBucket")
        self.assertEqual(bucket.get("DeletionPolicy"), "Retain")
        self.assertEqual(bucket.get("UpdateReplacePolicy"), "Retain")

    def test_vector_bucket_name_carries_account_and_region(self):
        """${ResourcePrefix}-${AccountId}-${Region}-vectors, mirroring how m2c
        names its ingest/proxy buckets."""
        self.assertEqual(
            self.res["HybridSearchVectorBucket"]["Properties"]["VectorBucketName"],
            ("!Sub", "${ResourcePrefix}-${AWS::AccountId}-${AWS::Region}-vectors"))

    def test_vector_bucket_name_fits_the_63_char_limit(self):
        """S3 Vectors enforces 3-63 chars, and THIS is the binding constraint
        on ResourcePrefix across the whole micro-workflow bundle.

        Worst case is the longest AWS region name (14 chars, ap-southeast-N)
        plus a 12-digit account id, so 36 characters are spoken for before
        ResourcePrefix contributes anything:

            ResourcePrefix <= 63 - 36 = 27

        That is TIGHTER than the Lambda FunctionName ceiling of 30 (64 minus
        the longest post-rename suffix, "-audiostem-separation-bucket-check").
        Since the bundle derives ResourcePrefix as "${Prefix}-${12-char
        stack-id slice}", the user-settable Prefix is capped at 27 - 13 = 14.
        """
        longest_region = "ap-southeast-7"          # 14 chars
        fixed = len(f"-{'9' * 12}-{longest_region}-vectors")
        self.assertEqual(fixed, 36)
        self.assertEqual(63 - fixed, 27)           # ResourcePrefix ceiling
        self.assertEqual(27 - len("-") - 12, 14)   # bundle Prefix ceiling
        # m2c passes ${solutionId}-${stackId}, e.g. so0050-0123456789ab.
        self.assertLessEqual(len("so0050-0123456789ab") + fixed, 63)

    def test_no_consumer_recomposes_the_vector_bucket_name(self):
        """Every reader derives the name from the resource, so it cannot drift.

        VectorBucketName is NOT a readOnlyProperty on this type, so
        !GetAtt ...VectorBucketName does not exist and !Ref returns the ARN.
        The name is therefore sliced out of the ARN. Recomposing it with a
        second !Sub would work today but gives the template two independent
        definitions of one string.
        """
        blob = json.dumps(self.tpl)
        self.assertNotIn("hybrid-search-vectors", blob,
                         "a stale literal vector bucket name survives")
        expected = ("!Select", [
            1, ("!Split", [
                "bucket/",
                ("!GetAtt", "HybridSearchVectorBucket.VectorBucketArn")])])
        env_refs = 0
        for node in self._of_type("AWS::Lambda::Function").values():
            env = node["Properties"].get("Environment", {}).get("Variables", {})
            if "VECTOR_BUCKET" in env:
                self.assertEqual(env["VECTOR_BUCKET"], expected)
                env_refs += 1
        self.assertEqual(env_refs, 6, "expected 6 Lambdas to read VECTOR_BUCKET")
        self.assertEqual(self.tpl["Outputs"]["VectorBucketName"]["Value"],
                         expected)
        # The ONE place the name is spelled out is the resource itself; the
        # !Sub above is the single definition all of the above derive from.
        self.assertEqual(
            sum(1 for _ in re.finditer(
                r"\$\{ResourcePrefix\}-\$\{AWS::AccountId\}-\$\{AWS::Region\}-vectors",
                blob)),
            1)

    def test_vector_bucket_arn_is_read_from_the_resource(self):
        """VectorBucketArn IS GetAtt-able (it is the primaryIdentifier), so the
        ARN — unlike the name — needs no slicing."""
        self.assertEqual(
            self.tpl["Outputs"]["VectorBucketArn"]["Value"],
            ("!GetAtt", "HybridSearchVectorBucket.VectorBucketArn"))


if __name__ == "__main__":
    unittest.main(verbosity=1)
