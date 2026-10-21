"""
media-toolkit-gateway unit tests (gateway_tool handlers).

This directory's existence also fixes a silent trap: before it existed,
`python3 -m unittest discover -s test` here resolved `test` to CPYTHON'S
OWN stdlib test package and ran 54 interpreter tests — a green "OK" that
validated nothing in this repo.

Run: python3 -m unittest discover -s test -p 'test_*.py'
"""
from __future__ import annotations

import os
import sys
import types
import unittest
from datetime import datetime, timezone
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "gateway_tool"))

os.environ.setdefault("AWS_DEFAULT_REGION", "eu-west-1")
os.environ.setdefault("INGEST_BUCKET", "test-ingest")
os.environ.setdefault("PROXY_BUCKET", "test-proxy")

import ingest_list  # noqa: E402
import upload_tool  # noqa: E402


def _ctx(tool):
    return types.SimpleNamespace(client_context=types.SimpleNamespace(
        custom={"bedrockAgentCoreToolName": f"media-ingest___{tool}"}))


def _up_ctx(tool):
    return types.SimpleNamespace(client_context=types.SimpleNamespace(
        custom={"bedrockAgentCoreToolName": f"media-uploads___{tool}"}))


_T = datetime(2026, 8, 31, 12, 0, 0, tzinfo=timezone.utc)



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


class FakeS3:
    """list_objects_v2 over a fixed key list, honouring Prefix, StartAfter,
    ContinuationToken and a small page size — enough to exercise every
    branch of the handler's pagination."""

    def __init__(self, keys, page_size=1000):
        self.keys = sorted(keys)
        self.page_size = page_size
        self.calls = []

    def list_objects_v2(self, **kw):
        self.calls.append(kw)
        keys = [k for k in self.keys if k.startswith(kw.get("Prefix", ""))]
        after = kw.get("StartAfter") or kw.get("ContinuationToken") or ""
        keys = [k for k in keys if k > after]
        page, rest = keys[:self.page_size], keys[self.page_size:]
        out = {"Contents": [{"Key": k, "Size": 10, "LastModified": _T}
                            for k in page]}
        if rest:
            out["IsTruncated"] = True
            out["NextContinuationToken"] = page[-1]
        return out


class TestListFiles(unittest.TestCase):
    def _run(self, keys, event, page_size=1000):
        fake = FakeS3(keys, page_size)
        old = ingest_list.s3
        ingest_list.s3 = fake
        try:
            return ingest_list.handler(event, _ctx("list_files")), fake
        finally:
            ingest_list.s3 = old

    def test_prefix_and_extension_filter_and_folder_markers_skipped(self):
        keys = ["a/one.mp4", "a/two.MOV", "a/sub/", "b/three.mp4"]
        r, _ = self._run(keys, {"prefix": "a/", "extension": ".mp4"})
        self.assertEqual([f["key"] for f in r["files"]], ["a/one.mp4"])
        self.assertEqual(r["files"][0]["s3uri"], "s3://test-ingest/a/one.mp4")
        self.assertEqual(r["count"], 1)
        self.assertNotIn("next_token", r)

    def test_extension_is_case_insensitive_and_accepts_a_list(self):
        keys = ["x/a.MP4", "x/b.mov", "x/c.wav"]
        r, _ = self._run(keys, {"extension": ["mp4", ".MOV"]})
        self.assertEqual([f["key"] for f in r["files"]],
                         ["x/a.MP4", "x/b.mov"])

    def test_default_page_is_20_and_next_token_resumes_exactly(self):
        keys = [f"k/{i:04d}.mp4" for i in range(45)]
        r1, _ = self._run(keys, {})
        self.assertEqual(r1["count"], 20)
        self.assertEqual(r1["next_token"], r1["files"][-1]["key"])
        r2, fake2 = self._run(keys, {"next_token": r1["next_token"]})
        self.assertEqual(fake2.calls[0]["StartAfter"], r1["next_token"])
        self.assertEqual(r2["count"], 20)
        r3, _ = self._run(keys, {"next_token": r2["next_token"]})
        self.assertEqual(r3["count"], 5)
        self.assertNotIn("next_token", r3)
        seen = ([f["key"] for f in r1["files"]]
                + [f["key"] for f in r2["files"]]
                + [f["key"] for f in r3["files"]])
        self.assertEqual(seen, sorted(keys))     # no gap, no repeat

    def test_max_results_is_clamped_both_ways(self):
        keys = [f"k/{i:04d}.mp4" for i in range(300)]
        r, _ = self._run(keys, {"max_results": 5000})
        self.assertEqual(r["count"], ingest_list._MAX_RESULTS)
        r, _ = self._run(keys, {"max_results": 0})
        self.assertEqual(r["count"], 1)

    def test_filter_walks_internal_pages_until_the_page_fills(self):
        # matches are sparse: 1 mp4 per 10 keys, S3 pages of 10 — the
        # handler must keep paging internally rather than return short
        keys = []
        for i in range(200):
            keys.append(f"k/{i:04d}.mp4" if i % 10 == 0 else f"k/{i:04d}.txt")
        r, fake = self._run(keys, {"extension": "mp4", "max_results": 15},
                            page_size=10)
        self.assertEqual(r["count"], 15)
        self.assertGreater(len(fake.calls), 1)
        self.assertIn("next_token", r)

    def test_full_s3_uri_prefix_is_accepted_only_for_the_ingest_bucket(self):
        keys = ["a/one.mp4"]
        r, _ = self._run(keys, {"prefix": "s3://test-ingest/a/"})
        self.assertEqual(r["count"], 1)
        r, _ = self._run(keys, {"prefix": "s3://other-bucket/a/"})
        self.assertTrue(r.get("error"))
        self.assertEqual(r["errorCode"], "INVALID_PARAMETER")

    def test_bad_parameters_refuse_with_named_errors(self):
        for event in ({"extension": 42}, {"extension": [""]},
                      {"max_results": "many"}, {"next_token": 7},  # nosec B105
                      {"prefix": 9}):
            r, _ = self._run(["a/x.mp4"], event)
            self.assertTrue(r.get("error"), event)
            self.assertEqual(r["errorCode"], "INVALID_PARAMETER", event)

    def test_empty_extension_means_no_filter(self):
        r, _ = self._run(["a/x.mp4", "a/y.mov"], {"extension": ""})
        self.assertEqual(r["count"], 2)

    def test_idempotency_token_is_ignored_never_rejected(self):
        r, _ = self._run(["a/x.mp4"], {"idempotency_token": "t-1"})  # nosec B105
        self.assertEqual(r["count"], 1)

    def test_unknown_tool_is_a_named_error(self):
        r = ingest_list.handler({}, _ctx("do_magic"))
        self.assertEqual(r["errorCode"], "UNKNOWN_TOOL")


if __name__ == "__main__":
    unittest.main()


# STRUCTURAL leak patterns only — naming the actual account id /
# buckets / usernames in this committed file would BE the leak these
# checks exist to prevent (mirrors the build script's kit check).
# Sharp deployment-specific literals live in the untracked
# .scrub-patterns.local (gitignored, one regex per line): loaded when
# present (maintainer machines), silently absent on customer checkouts.
_DOC_FAKE_IDS = ("111122223333", "123456789012", "000000000000")
_STRUCTURAL_LEAKS = (r"/Users/[a-zA-Z]|so0050-[0-9a-f]{12}|"
                     r"[a-z0-9]+-[0-9a-f]{12}-[0-9]{12}-")


def _deployment_value_leak(root, text):
    """Reason string when `text` carries a deployment value, else None."""
    import re
    for m in re.finditer(r"(?<![0-9])[0-9]{12}(?![0-9])", text):
        if m.group(0) not in _DOC_FAKE_IDS:
            return f"12-digit id {m.group(0)}"
    m = re.search(_STRUCTURAL_LEAKS, text)
    if m:
        return f"structural match {m.group(0)!r}"
    local = os.path.join(root, ".scrub-patterns.local")
    if os.path.exists(local):
        for pat in open(local).read().splitlines():
            if pat and re.search(pat, text):
                return "local scrub pattern"
    return None


class TestSkillToolPins(unittest.TestCase):
    """The shipped skills (skills/*/SKILL.md) hardcode gateway tool
    names. A tool rename in cfnstack.yaml would silently strand every
    recipe that references it — this pin makes the drift a test
    failure. Both directions are deliberate: names in skills must
    exist in the template; the reverse (unused tools) is fine."""

    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _template_tools(self):
        import re
        text = open(os.path.join(self._ROOT, "cfnstack.yaml")).read()
        tools = set()
        # every GatewayTarget block: Name: <target> ... InlinePayload
        # tool Names until the next target
        for tm in re.finditer(
                r"Type: AWS::BedrockAgentCore::GatewayTarget(.*?)"
                r"(?=Type: AWS::|Outputs:)", text, re.DOTALL):
            block = tm.group(1)
            target = re.search(r"Name: ([a-z0-9-]+)\n", block).group(1)
            for t in re.findall(r"- Name: ([a-z_]+)\n", block):
                tools.add(f"{target}___{t}")
        return tools

    def test_every_tool_named_in_a_skill_exists_in_the_template(self):
        import glob
        import re
        advertised = self._template_tools()
        self.assertGreater(len(advertised), 40, "template parse broke")
        referenced = {}
        paths = glob.glob(os.path.join(self._ROOT, "skills",
                                       "*", "**", "*.md"),
                          recursive=True)
        paths.append(os.path.join(self._ROOT, "README.md"))
        for path in paths:
            # NO \b on the left: an underscored-target typo like
            # media_processing___x must be CAPTURED (and then fail the
            # assertion), not silently skipped — the exact bug this
            # regex once had. Placeholders like <target>___<tool> and
            # the generic *___describe_tool are excluded by shape.
            for m in re.finditer(r"(?<![a-z0-9_<*-])([a-z][a-z0-9_-]*)___"
                                 r"([a-z][a-z0-9_]*)", open(path).read()):
                referenced.setdefault(m.group(0), set()).add(
                    os.path.relpath(path, self._ROOT))
        self.assertTrue(referenced, "no tool references found — the "
                        "scan pattern or skills/ layout changed")
        missing = {t: sorted(files) for t, files in referenced.items()
                   if t not in advertised}
        self.assertEqual(missing, {}, "skills reference tools the "
                         "gateway does not advertise")

    def test_skills_carry_no_deployment_values(self):
        """The skills ship to customers: no account ids, real buckets,
        or personal identifiers (the 2026-09-17 scrub, pinned)."""
        import glob
        hits = []
        for path in glob.glob(os.path.join(self._ROOT, "skills", "**",
                                           "*.*"), recursive=True):
            why = _deployment_value_leak(
                self._ROOT, open(path, errors="ignore").read())
            if why:
                hits.append((os.path.relpath(path, self._ROOT), why))
        self.assertEqual(hits, [])


class TestKitPublisher(unittest.TestCase):
    """kit_publisher: Create/Update put BOTH keys (versioned +
    stable alias) from the baked-in zip; Delete is a no-op; a failure
    reports FAILED to CFN instead of hanging the stack."""

    def _run(self, request_type, put_side_effect=None,
             existing_etag=None):
        import kit_publisher as kp
        from botocore.exceptions import ClientError
        calls, responses = [], []

        def fake_put(**kw):
            if put_side_effect:
                raise put_side_effect
            calls.append(kw)

        def fake_head(**kw):
            if existing_etag is None:
                raise ClientError({"Error": {"Code": "404",
                                             "Message": "nope"}},
                                  "HeadObject")
            return {"ETag": existing_etag}

        with mock.patch.object(kp.s3, "put_object", side_effect=fake_put), \
             mock.patch.object(kp.s3, "head_object",
                               side_effect=fake_head), \
             mock.patch.object(kp, "_send",
                               lambda e, c, s, d, reason="":
                               responses.append((s, reason))), \
             mock.patch.object(kp, "KIT_LOCAL", __file__):
            kp.handler({"RequestType": request_type,
                        "ResourceProperties":
                            {"Bucket": "proxy-b", "Version": "9.9.9"}},
                       None)
        return calls, responses

    def test_create_publishes_versioned_and_alias(self):
        calls, responses = self._run("Create")
        self.assertEqual(responses, [("SUCCESS", "")])
        self.assertEqual(
            [c["Key"] for c in calls],
            ["_toolkit/client-kit-9.9.9.zip", "_toolkit/client-kit.zip"])
        self.assertTrue(all(c["Bucket"] == "proxy-b" for c in calls))
        self.assertTrue(all(c["ContentType"] == "application/zip"
                            for c in calls))

    def test_update_republishes(self):
        calls, responses = self._run("Update")
        self.assertEqual(len(calls), 2)
        self.assertEqual(responses[0][0], "SUCCESS")

    def test_kit_env_injection_is_deterministic_and_preserves_members(self):
        # the published kit carries its OWN deployment's secret path
        # (kit.env) so configure.sh works on a derived GatewayName; the
        # bytes must be deterministic or the ETag idempotency skip breaks
        import io
        import zipfile

        import kit_publisher as kp
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("scripts/configure.sh", "echo hi\n")
        baked = buf.getvalue()
        one = kp._with_kit_env(baked, "/gw-abc123/dev/gateway-credentials",
                               "eu-west-1")
        two = kp._with_kit_env(baked, "/gw-abc123/dev/gateway-credentials",
                               "eu-west-1")
        self.assertEqual(one, two)
        with zipfile.ZipFile(io.BytesIO(one)) as z:
            self.assertEqual(sorted(z.namelist()),
                             ["kit.env", "scripts/configure.sh"])
            # mode 0644 explicitly — external_attr 0 extracts as 000 on
            # some tools, and an unreadable kit.env kills configure.sh
            self.assertEqual(
                z.getinfo("kit.env").external_attr >> 16, 0o644)
            env = z.read("kit.env").decode()
        self.assertIn("KIT_SECRET_NAME=/gw-abc123/dev/gateway-credentials",
                      env)
        self.assertIn("KIT_REGION=eu-west-1", env)
        # no SecretName property (an older template invoking newer code)
        # must leave the kit byte-identical to the baked zip
        self.assertEqual(kp._with_kit_env(baked, "", "eu-west-1"), baked)

    def test_publisher_secret_name_mirrors_the_secret(self):
        # kit.env must carry the SAME path the credentials secret is
        # created under — one expression, used twice (secret Name +
        # KitPublisher SecretName); a rename of either strands the kit
        text = open(os.path.join(_ROOT, "cfnstack.yaml")).read()
        self.assertEqual(
            text.count("/${GatewayName}/${Environment}/gateway-credentials"),
            3,
            "the gateway-credentials path expression must appear exactly "
            "three times — the secret's Name, ClientKitPublisher's "
            "SecretName, and the GatewayCredentialsSecretName output; a "
            "rename of any one strands the others")

    def test_delete_is_a_noop_success(self):
        calls, responses = self._run("Delete")
        self.assertEqual(calls, [])
        self.assertEqual(responses, [("SUCCESS", "")])

    def test_failure_reports_failed(self):
        calls, responses = self._run("Create",
                                     put_side_effect=RuntimeError("denied"))
        self.assertEqual(responses[0][0], "FAILED")
        self.assertIn("denied", responses[0][1])

    def test_byte_identical_republish_is_skipped(self):
        import hashlib
        etag = '"%s"' % hashlib.md5(open(__file__, "rb").read(), usedforsecurity=False).hexdigest()
        calls, responses = self._run("Update", existing_etag=etag)
        self.assertEqual(calls, [])
        self.assertEqual(responses, [("SUCCESS", "")])

    def test_changed_content_overwrites(self):
        calls, _ = self._run("Update", existing_etag='"different"')
        self.assertEqual(len(calls), 2)


class TestClientScripts(unittest.TestCase):
    """The kit's shell scripts must at least parse, and configure.sh
    must never learn a deployment-specific value beyond the documented
    secret-name default."""

    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_scripts_parse(self):
        import subprocess
        for name in ("configure.sh", "get_gateway_token.sh"):
            p = os.path.join(self._ROOT, "scripts", name)
            r = subprocess.run(["bash", "-n", p], capture_output=True,  # nosec B603
                               text=True)
            self.assertEqual(r.returncode, 0, f"{name}: {r.stderr}")

    def test_powershell_scripts_parse(self):
        """The PowerShell twins (Windows clients) must parse. Needs a
        local pwsh; SKIPS LOUDLY without one — a skip is not a pass, so
        CI hosts that validate the kit should carry PowerShell."""
        import shutil
        import subprocess
        if shutil.which("pwsh") is None:
            self.skipTest("pwsh not installed — PowerShell twins unparsed")
        for name in ("configure.ps1", "get_gateway_token.ps1"):
            p = os.path.join(self._ROOT, "scripts", name)
            r = subprocess.run(  # nosec B603
                ["pwsh", "-NoProfile", "-Command",
                 "$err = $null; [void][System.Management.Automation."
                 "Language.Parser]::ParseFile('" + p + "', [ref]$null, "
                 "[ref]$err); if ($err) { $err | Out-String; exit 1 }"],
                capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"{name}: {r.stdout}{r.stderr}")

    def test_configure_carries_no_deployment_values(self):
        text = open(os.path.join(self._ROOT, "scripts",
                                 "configure.sh")).read()
        # the secret NAME default (/media-toolkit-gateway/...) is
        # deliberately allowed — a path convention, not a credential
        self.assertIsNone(_deployment_value_leak(self._ROOT, text))
        # the client-side server key, pinned: skills prose and the
        # configs configure.sh writes must agree on this one string
        self.assertIn('SERVER_KEY="media-toolkit"', text)
        # the PowerShell twins carry the same guarantees
        for name in ("configure.ps1", "get_gateway_token.ps1"):
            ps = open(os.path.join(self._ROOT, "scripts", name)).read()
            self.assertIsNone(_deployment_value_leak(self._ROOT, ps), name)
        ps_cfg = open(os.path.join(self._ROOT, "scripts",
                                   "configure.ps1")).read()
        self.assertIn('$ServerKey = "media-toolkit"', ps_cfg)


class TestUploadTool(unittest.TestCase):
    """media-uploads: the control plane for presigned uploads. The 5 GiB
    cap (user decision 2026-10-14) bounds every multipart plan to <=50
    parts, which is what makes start_upload's full-URL-set response and
    the no-pagination contract of get_upload_part_urls possible."""

    def _s3(self, existing=None):
        fake = mock.Mock()
        fake.list_objects_v2.return_value = {
            "Contents": ([{"Key": existing, "Size": 123}] if existing else [])}
        fake.create_multipart_upload.return_value = {"UploadId": "UP1"}
        fake.list_parts.return_value = {
            "Parts": [{"PartNumber": 1, "Size": 100},
                      {"PartNumber": 2, "Size": 100}],
            "IsTruncated": False}
        fake.generate_presigned_url.side_effect = (
            lambda op, Params=None, ExpiresIn=None:
            f"https://signed/{Params.get('PartNumber', 'single')}")
        fake.exceptions = upload_tool.s3.exceptions
        return fake

    def _call(self, tool, event, s3=None):
        with mock.patch.object(upload_tool, "s3", s3 or self._s3()):
            return upload_tool.handler(dict(event), _up_ctx(tool))

    # ── start_upload ──────────────────────────────────────────────────

    def test_small_file_is_a_single_put_with_pinned_type_and_length(self):
        s3 = self._s3()
        out = self._call("start_upload",
                         {"key": "uploads/promo.mov", "size_bytes": 1024},
                         s3=s3)
        self.assertEqual(out["mode"], "single")
        self.assertEqual(out["content_type"], "video/quicktime")
        self.assertEqual(out["s3uri"], "s3://test-ingest/uploads/promo.mov")
        self.assertIn("url", out)
        self.assertEqual(out["size_bytes"], 1024)   # echoed for the uploader
        # the declared size is SIGNED — a different-sized body fails the PUT
        params = s3.generate_presigned_url.call_args.kwargs["Params"]
        self.assertEqual(params["ContentLength"], 1024)

    def test_large_file_plans_multipart_with_every_part_url(self):
        size = 2 * 1024**3                       # 2 GiB
        out = self._call("start_upload",
                         {"key": "uploads/ep.mxf", "size_bytes": size})
        self.assertEqual(out["mode"], "multipart")
        self.assertEqual(out["content_type"], "application/mxf")
        self.assertEqual(out["part_size"], 128 * 1024**2)
        self.assertEqual(out["part_count"], 16)
        self.assertEqual([p["part_number"] for p in out["parts"]],
                         list(range(1, 17)))
        self.assertEqual(out["upload_id"], "UP1")
        self.assertEqual(out["size_bytes"], size)   # echoed for the uploader

    def test_the_cap_refuses_by_name_and_the_plan_never_exceeds_50_parts(self):
        out = self._call("start_upload", {"key": "a.bin",
                                          "size_bytes": 5 * 1024**3 + 1})
        self.assertEqual(out.get("errorCode"), "FILE_TOO_LARGE")
        # a tiny requested part_size is floored so the URL set still fits
        # one response (and never below the S3 minimum)
        out = self._call("start_upload", {"key": "a.bin",
                                          "size_bytes": 5 * 1024**3,
                                          "part_size": 1024})
        self.assertLessEqual(out["part_count"], 50)
        self.assertGreaterEqual(out["part_size"], upload_tool.MIN_PART_SIZE)

    def test_mime_overlay_beats_stdlib_and_unknown_falls_back(self):
        self.assertEqual(upload_tool._guess_mime("x/a.mxf"), "application/mxf")
        self.assertEqual(upload_tool._guess_mime("a.mp4"), "video/mp4")
        # documents/images ride the stdlib table
        self.assertEqual(upload_tool._guess_mime("a.pdf"), "application/pdf")
        self.assertEqual(upload_tool._guess_mime("a.json"), "application/json")
        self.assertEqual(upload_tool._guess_mime("a.png"), "image/png")
        # sidecar formats the stdlib misses come from the overlay
        self.assertEqual(upload_tool._guess_mime("a.jsonl"),
                         "application/jsonl")
        self.assertEqual(upload_tool._guess_mime("a.vtt"), "text/vtt")
        self.assertEqual(upload_tool._guess_mime("a.itt"),
                         "application/ttml+xml")
        self.assertEqual(upload_tool._guess_mime("a.weird"),
                         "application/octet-stream")

    def test_explicit_content_type_overrides_detection(self):
        out = self._call("start_upload",
                         {"key": "u/export.bin", "size_bytes": 10,
                          "content_type": "application/vnd.custom+json"})
        self.assertEqual(out["content_type"], "application/vnd.custom+json")

    def test_existing_key_refused_unless_overwrite(self):
        s3 = self._s3(existing="uploads/promo.mov")
        out = self._call("start_upload", {"key": "uploads/promo.mov",
                                          "size_bytes": 10}, s3=s3)
        self.assertEqual(out.get("errorCode"), "KEY_EXISTS")
        out = self._call("start_upload", {"key": "uploads/promo.mov",
                                          "size_bytes": 10,
                                          "overwrite": True}, s3=s3)
        self.assertEqual(out["mode"], "single")
        # string-boolean coercion, same rule as the admin tools
        out = self._call("start_upload", {"key": "uploads/promo.mov",
                                          "size_bytes": 10,
                                          "overwrite": "false"}, s3=s3)
        self.assertEqual(out.get("errorCode"), "KEY_EXISTS")

    def test_key_sanitization(self):
        # omission has its own code (one vocabulary with upload_id)...
        out = self._call("start_upload", {"key": "", "size_bytes": 10})
        self.assertEqual(out.get("errorCode"), "MISSING_PARAMETER")
        # ...format problems are INVALID
        for bad in ("../etc/x", "a/../../b", "dir/",
                    "s3://other-bucket/k.mp4"):
            out = self._call("start_upload", {"key": bad, "size_bytes": 10})
            self.assertEqual(out.get("errorCode"), "INVALID_PARAMETER", bad)
        # a leading-'..' SEGMENT NAME is a legal key, not traversal
        out = self._call("start_upload", {"key": "..archive/f.mp4",
                                          "size_bytes": 10})
        self.assertEqual(out.get("mode"), "single")
        # an s3:// uri on the right bucket is accepted and stripped
        out = self._call("start_upload",
                         {"key": "s3://test-ingest/u/f.wav", "size_bytes": 1})
        self.assertEqual(out["s3uri"], "s3://test-ingest/u/f.wav")
        self.assertEqual(out["content_type"], "audio/wav")

    # ── the rest of the session ───────────────────────────────────────

    def test_part_url_refresh_validates_against_the_plan(self):
        plan = {"key": "u/f.mxf", "upload_id": "UP1",
                "size_bytes": 1024**3, "part_size": 128 * 1024**2}  # 8 parts
        out = self._call("get_upload_part_urls",
                         dict(plan, part_numbers=[3, 7]))
        self.assertEqual([p["part_number"] for p in out["parts"]], [3, 7])
        # beyond the plan, malformed, or over the per-call budget: refused —
        # the plan bound is what makes the 5 GiB cap enforced, not advisory
        for bad in ([], [0], [9], [10001], list(range(1, 52)), "3"):
            out = self._call("get_upload_part_urls",
                             dict(plan, part_numbers=bad))
            self.assertTrue(out.get("error"), bad)
        # missing plan fields refused by name
        out = self._call("get_upload_part_urls",
                         {"key": "u/f.mxf", "upload_id": "UP1",
                          "part_numbers": [1]})
        self.assertEqual(out.get("errorCode"), "INVALID_PARAMETER")

    def test_part_urls_are_signed_with_exact_content_length(self):
        s3 = self._s3()
        size = 300 * 1024**2                       # 2 full parts + 44 MiB
        self._call("start_upload", {"key": "u/f.mxf", "size_bytes": size},
                   s3=s3)
        lengths = [kw["Params"]["ContentLength"]
                   for args, kw in s3.generate_presigned_url.call_args_list
                   if kw.get("Params", {}).get("PartNumber")]
        self.assertEqual(lengths, [128 * 1024**2, 128 * 1024**2,
                                   44 * 1024**2])

    def test_a_presign_failure_mid_start_aborts_the_session(self):
        s3 = self._s3()
        s3.generate_presigned_url.side_effect = RuntimeError("throttle")
        with self.assertRaises(RuntimeError):
            self._call("start_upload",
                       {"key": "u/f.mxf", "size_bytes": 1024**3}, s3=s3)
        s3.abort_multipart_upload.assert_called_once()

    def test_complete_sorts_parts_and_returns_the_verified_object(self):
        s3 = self._s3(existing="u/f.mxf")
        s3.complete_multipart_upload.return_value = {"ETag": '"abc-16"'}
        out = self._call("complete_upload",
                         {"key": "u/f.mxf", "upload_id": "UP1",
                          "parts": [{"part_number": 2, "etag": '"e2"'},
                                    {"part_number": 1, "etag": "e1"}]},
                         s3=s3)
        self.assertEqual(out["s3uri"], "s3://test-ingest/u/f.mxf")
        self.assertEqual(out["etag"], "abc-16")
        self.assertEqual(out["size_bytes"], 123)
        sent = s3.complete_multipart_upload.call_args.kwargs[
            "MultipartUpload"]["Parts"]
        self.assertEqual([p["PartNumber"] for p in sent], [1, 2])
        # sorted by part number, quotes stripped from the client's ETags
        self.assertEqual([p["ETag"] for p in sent], ["e1", "e2"])

    def test_complete_enforces_the_cap_against_s3s_own_part_record(self):
        # plans are caller-reconstructed, so the SUM is only enforceable
        # here — against what S3 actually stored; over-cap aborts the session
        s3 = self._s3()
        s3.list_parts.return_value = {
            "Parts": [{"PartNumber": 1, "Size": 3 * 1024**3},
                      {"PartNumber": 2, "Size": 3 * 1024**3}],
            "IsTruncated": False}
        out = self._call("complete_upload",
                         {"key": "u/f.mxf", "upload_id": "UP1",
                          "parts": [{"part_number": 1, "etag": "e1"},
                                    {"part_number": 2, "etag": "e2"}]},
                         s3=s3)
        self.assertEqual(out.get("errorCode"), "FILE_TOO_LARGE")
        s3.abort_multipart_upload.assert_called_once()
        s3.complete_multipart_upload.assert_not_called()

    def test_booleans_are_not_part_numbers_and_s3_codes_go_snake(self):
        out = self._call("get_upload_part_urls",
                         {"key": "u/f.mxf", "upload_id": "UP1",
                          "size_bytes": 1024**3, "part_size": 128 * 1024**2,
                          "part_numbers": [True]})
        self.assertEqual(out.get("errorCode"), "INVALID_PARAMETER")
        self.assertEqual(upload_tool._snake_code("EntityTooSmall"),
                         "ENTITY_TOO_SMALL")
        self.assertEqual(upload_tool._snake_code("InvalidPart"),
                         "INVALID_PART")

    def test_complete_retry_after_lost_response_gets_the_designed_error(self):
        # first complete succeeded but the response was lost: the retry's
        # list_parts raises NoSuchUpload, which must surface as the designed
        # code, not a raw Lambda error
        s3 = self._s3()
        s3.list_parts.side_effect = upload_tool.s3.exceptions.NoSuchUpload(
            {"Error": {"Code": "NoSuchUpload"}}, "ListParts")
        out = self._call("complete_upload",
                         {"key": "u/f.mxf", "upload_id": "UP1",
                          "parts": [{"part_number": 1, "etag": "e1"}]},
                         s3=s3)
        self.assertEqual(out.get("errorCode"), "NO_SUCH_UPLOAD")

    def test_cap_sums_only_the_completing_subset(self):
        # stray parts under other numbers are discarded by S3 at complete —
        # they must not fail a legitimate assembly
        s3 = self._s3(existing="u/f.mxf")
        s3.list_parts.return_value = {
            "Parts": [{"PartNumber": 1, "Size": 1024},
                      {"PartNumber": 99, "Size": 6 * 1024**3}],  # stray
            "IsTruncated": False}
        s3.complete_multipart_upload.return_value = {"ETag": '"ok"'}
        out = self._call("complete_upload",
                         {"key": "u/f.mxf", "upload_id": "UP1",
                          "parts": [{"part_number": 1, "etag": "e1"}]},
                         s3=s3)
        self.assertEqual(out.get("etag"), "ok")

    def test_omitted_key_and_boolean_sizes_refused_by_the_right_codes(self):
        out = self._call("start_upload", {"size_bytes": 10})
        self.assertEqual(out.get("errorCode"), "MISSING_PARAMETER")
        out = self._call("start_upload", {"key": "u/f.mov",
                                          "size_bytes": True})
        self.assertEqual(out.get("errorCode"), "INVALID_PARAMETER")
        out = self._call("get_upload_part_urls",
                         {"key": "u/f.mov", "upload_id": "UP1",
                          "size_bytes": True, "part_size": 128 * 1024**2,
                          "part_numbers": [1]})
        self.assertEqual(out.get("errorCode"), "INVALID_PARAMETER")

    def test_complete_and_abort_refuse_malformed_input(self):
        out = self._call("complete_upload",
                         {"key": "u/f.mxf", "upload_id": "UP1",
                          "parts": [{"part_number": 1}]})
        self.assertEqual(out.get("errorCode"), "INVALID_PARAMETER")
        out = self._call("abort_upload", {"key": "u/f.mxf"})
        self.assertEqual(out.get("errorCode"), "MISSING_PARAMETER")

    def test_abort_succeeds_and_idempotency_token_is_ignored_not_rejected(self):
        s3 = self._s3()
        out = self._call("abort_upload",
                         {"key": "u/f.mxf", "upload_id": "UP1",
                          "idempotency_token": "tok"}, s3=s3)
        self.assertTrue(out["aborted"])
        s3.abort_multipart_upload.assert_called_once()


class TestSkillUploader(unittest.TestCase):
    """skills/upload-media/references/uploader.py — the skill's client-side
    data plane. No network: _put is replaced by an S3-faithful fake that
    reads the streamed body and answers its MD5 as the ETag (what a
    plain/SSE-S3 bucket does), so the etag==local_md5 parity, the plan
    validation branches, and the resume/rewrite guards are pinned at the
    exact error shapes SKILL.md branches on (errors.plan vs errors.<n>
    vs errors.fatal)."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        path = os.path.join(_ROOT, "skills", "upload-media", "references",
                            "uploader.py")
        spec = importlib.util.spec_from_file_location("skill_uploader", path)
        cls.up = importlib.util.module_from_spec(spec)
        # never drop a __pycache__ into the shipped skill tree — the .pyc
        # embeds local paths and fails test_skills_carry_no_deployment_values
        was = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(cls.up)
        finally:
            sys.dont_write_bytecode = was

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)

    def _file(self, size, seed=b"\x5a"):
        path = os.path.join(self.tmp, "data.bin")
        with open(path, "wb") as f:
            f.write((seed * size)[:size])
        return path

    def _run(self, plan, fail_urls=()):
        """Run main() against a fake _put; return (rc, parsed json line)."""
        import contextlib
        import hashlib
        import io
        import json
        import urllib.error
        calls = []

        def fake_put(url, body, length, content_type=None):
            data, chunk = b"", body.read()
            while chunk:
                data, chunk = data + chunk, body.read()
            body.close()
            calls.append({"url": url, "length": length, "ct": content_type})
            if url in fail_urls:
                if isinstance(fail_urls, dict):
                    raise fail_urls[url]
                raise urllib.error.HTTPError(url, 403, "expired", {}, None)
            return hashlib.md5(data).hexdigest()  # nosec B324 — S3 ETag law

        plan_path = os.path.join(self.tmp, "plan.json")
        with open(plan_path, "w") as f:
            json.dump(plan, f)
        buf = io.StringIO()
        with mock.patch.object(self.up, "_put", fake_put), \
                contextlib.redirect_stdout(buf):
            rc = self.up.main(plan_path)
        return rc, json.loads(buf.getvalue().strip()), calls

    @staticmethod
    def _resp(part_size, part_count, url="https://signed/p{n}"):
        return {"mode": "multipart", "part_size": part_size,
                "part_count": part_count,
                "parts": [{"part_number": n, "url": url.format(n=n)}
                          for n in range(1, part_count + 1)]}

    # ── plan validation: refusals name the condition as errors.plan ──────

    def test_response_shape_mutation_guard_uses_the_part_size_bucket(self):
        # response carries no size_bytes, so the guard is the bucket
        # inequality (want-1)*part_size < size <= want*part_size —
        # pinned at BOTH boundaries
        f = self._file(400)
        rc, out, _ = self._run({"file": f, "response": self._resp(200, 2)})
        self.assertEqual(rc, 0)                      # 400 == upper bound: ok
        f = self._file(200)
        rc, out, _ = self._run({"file": f, "response": self._resp(200, 2)})
        self.assertEqual(rc, 1)                      # 200 == lower bound: NOT ok
        self.assertIn("changed since", out["errors"]["plan"])
        f = self._file(201)
        rc, out, _ = self._run({"file": f, "response": self._resp(200, 2)})
        self.assertEqual(rc, 0)                      # lower bound + 1: ok

    def test_explicit_size_bytes_is_exact_in_both_modes(self):
        f = self._file(300)
        rc, out, _ = self._run({"file": f, "mode": "multipart",
                                "part_size": 150, "size_bytes": 299,
                                "parts": [{"part_number": 1, "url": "u1"},
                                          {"part_number": 2, "url": "u2"}]})
        self.assertEqual(rc, 1)
        self.assertIn("made for 299", out["errors"]["plan"])
        rc, out, _ = self._run({"file": f, "mode": "single", "url": "u",
                                "content_type": "a/b", "size_bytes": 299})
        self.assertEqual(rc, 1)
        self.assertIn("made for 299", out["errors"]["plan"])

    def test_single_without_content_type_is_refused_before_any_put(self):
        f = self._file(10)
        rc, out, calls = self._run({"file": f, "mode": "single", "url": "u"})
        self.assertEqual(rc, 1)
        self.assertIn("content_type", out["errors"]["plan"])
        self.assertEqual(calls, [])

    def test_missing_mode_and_missing_parts_are_named_not_keyerrors(self):
        f = self._file(10)
        rc, out, _ = self._run({"file": f})
        self.assertEqual(rc, 1)
        self.assertIn("no mode", out["errors"]["plan"])
        rc, out, _ = self._run({"file": f, "mode": "multipart"})
        self.assertEqual(rc, 1)
        self.assertIn("part_size and parts", out["errors"]["plan"])

    def test_duplicate_part_numbers_are_refused_even_across_types(self):
        f = self._file(300)
        rc, out, _ = self._run({"file": f, "mode": "multipart",
                                "part_size": 150, "size_bytes": 300,
                                "parts": [{"part_number": 1, "url": "a"},
                                          {"part_number": "1", "url": "b"},
                                          {"part_number": 2, "url": "c"}]})
        self.assertEqual(rc, 1)
        self.assertIn("duplicate part entry", out["errors"]["plan"])

    def test_coverage_check_counts_parts_union_done(self):
        # the documented minimal resume shape — fresh urls for the failed
        # parts only, done for the rest — must PASS validation
        f = self._file(300)
        plan = {"file": f, "response": self._resp(50, 6),
                "parts": [{"part_number": 3, "url": "https://fresh/p3"}],
                "done": [{"part_number": n, "etag": f"e{n}"}
                         for n in (1, 2, 4, 5, 6)]}
        rc, out, _ = self._run(plan)
        self.assertEqual(rc, 0)
        # a TRUNCATED verbatim paste (part_count says 6, parts carry 5)
        # is the transcription slip the guard exists for
        bad = self._resp(50, 6)
        bad["parts"] = bad["parts"][:-1]
        rc, out, _ = self._run({"file": f, "response": bad})
        self.assertEqual(rc, 1)
        self.assertIn("transcription slip", out["errors"]["plan"])

    # ── response/explicit precedence and the resume patch merge ──────────

    def test_top_level_parts_patch_into_response_parts_by_number(self):
        # the natural verbatim paste of get_upload_part_urls at TOP level
        # must be the correct resume edit: fresh url wins for its number,
        # stale urls stay for the rest, done parts are never PUT
        f = self._file(300)
        plan = {"file": f, "response": self._resp(50, 6),
                "parts": [{"part_number": 3, "url": "https://fresh/p3"}],
                "done": [{"part_number": n, "etag": f"e{n}"}
                         for n in (1, 2, 4, 5)]}
        rc, out, calls = self._run(plan)
        self.assertEqual(rc, 0)
        urls = sorted(c["url"] for c in calls)
        self.assertEqual(urls, ["https://fresh/p3", "https://signed/p6"])
        self.assertEqual([r["part_number"] for r in out["etags"]],
                         [1, 2, 3, 4, 5, 6])

    def test_patch_part_outside_the_response_plan_is_refused(self):
        f = self._file(300)
        rc, out, _ = self._run({"file": f, "response": self._resp(50, 6),
                                "parts": [{"part_number": 9, "url": "x"}]})
        self.assertEqual(rc, 1)
        self.assertIn("not in the", out["errors"]["plan"])

    def test_mixed_shape_is_refused_by_name_never_resolved_silently(self):
        # a stale response block beside fresh explicit fields (or a stray
        # per-mode key beside a response) has two copies of one fact and
        # no deterministic winner — the helper refuses rather than picks
        f = self._file(300)
        rc, out, calls = self._run(
            {"file": f, "mode": "multipart", "part_size": 150,
             "size_bytes": 300,
             "response": self._resp(50, 6, "https://stale/p{n}"),
             "parts": [{"part_number": 1, "url": "https://new/p1"},
                       {"part_number": 2, "url": "https://new/p2"}]})
        self.assertEqual(rc, 1)
        self.assertIn("keep exactly one shape", out["errors"]["plan"])
        self.assertEqual(calls, [])
        rc, out, _ = self._run({"file": f, "response": self._resp(50, 6),
                                "part_size": 100})
        self.assertEqual(rc, 1)
        self.assertIn("part_size", out["errors"]["plan"])

    def test_explicit_multipart_requires_size_bytes_as_its_mutation_guard(self):
        f = self._file(300)
        rc, out, calls = self._run(
            {"file": f, "mode": "multipart", "part_size": 150,
             "parts": [{"part_number": 1, "url": "a"},
                       {"part_number": 2, "url": "b"}]})
        self.assertEqual(rc, 1)
        self.assertIn("size_bytes", out["errors"]["plan"])
        self.assertEqual(calls, [])

    def test_done_list_is_validated_like_parts(self):
        f = self._file(300)
        base = {"file": f, "response": self._resp(50, 6)}
        rc, out, _ = self._run(dict(base, done=[
            {"part_number": 1, "etag": "old"},
            {"part_number": "1", "etag": "new"}]))
        self.assertEqual(rc, 1)
        self.assertIn("duplicate done entry", out["errors"]["plan"])
        # a stray done number is caught by the coverage check on BOTH
        # shapes (explicit derives part_count from its required size_bytes)
        rc, out, _ = self._run(dict(base, done=[
            {"part_number": 9, "etag": "e9"}]))
        self.assertEqual(rc, 1)
        self.assertIn("transcription slip", out["errors"]["plan"])
        rc, out, _ = self._run(
            {"file": f, "mode": "multipart", "part_size": 50,
             "size_bytes": 300,
             "parts": [{"part_number": n, "url": f"u{n}"}
                       for n in range(1, 7)],
             "done": [{"part_number": 9, "etag": "e9"}]})
        self.assertEqual(rc, 1)
        self.assertIn("transcription slip", out["errors"]["plan"])

    def test_explicit_truncated_plan_is_refused_like_the_response_shape(self):
        # explicit shape: part_count = ceil(size_bytes / part_size); a
        # pasted parts list missing one part must not silently assemble
        # a truncated object via complete_upload
        f = self._file(300)
        rc, out, calls = self._run(
            {"file": f, "mode": "multipart", "part_size": 50,
             "size_bytes": 300,
             "parts": [{"part_number": n, "url": f"u{n}"}
                       for n in range(1, 6)]})      # part 6 lost
        self.assertEqual(rc, 1)
        self.assertIn("transcription slip", out["errors"]["plan"])
        self.assertEqual(calls, [])

    def test_duplicate_patch_entries_are_refused_by_name(self):
        f = self._file(300)
        rc, out, _ = self._run(
            {"file": f, "response": self._resp(50, 6),
             "parts": [{"part_number": 3, "url": "https://fresh-a/p3"},
                       {"part_number": "3", "url": "https://fresh-b/p3"}]})
        self.assertEqual(rc, 1)
        self.assertIn("duplicate patch entry", out["errors"]["plan"])

    def test_rewrite_guard_stays_armed_across_consecutive_resumes(self):
        # verified done rows must RE-EMIT their local_md5, or the guard
        # silently disarms on the second resume of the same upload
        f = self._file(300)
        rc, first, _ = self._run({"file": f, "response": self._resp(200, 2)})
        self.assertEqual(rc, 0)
        rc, second, calls = self._run({"file": f,
                                       "response": self._resp(200, 2),
                                       "done": first["etags"]})
        self.assertEqual(rc, 0)
        self.assertEqual(calls, [])                   # nothing left to PUT
        for row in second["etags"]:
            self.assertIn("local_md5", row)           # verified → re-armed

    # ── transfer results: md5 parity, failure shape, string tolerance ────

    def test_etag_equals_local_md5_on_every_row_and_in_single_mode(self):
        f = self._file(300)
        rc, out, _ = self._run({"file": f, "response": self._resp(200, 2)})
        self.assertEqual(rc, 0)
        for row in out["etags"]:
            self.assertEqual(row["etag"], row["local_md5"])
        rc, out, calls = self._run({"file": self._file(10),
                                    "response": {"mode": "single", "url": "u",
                                                 "content_type": "a/b"}})
        self.assertEqual(rc, 0)
        self.assertEqual(out["etag"], out["local_md5"])
        self.assertEqual(calls[0]["ct"], "a/b")      # header echoed on single

    def test_failure_report_keeps_resume_state_with_string_part_numbers(self):
        # agent-transcribed string part numbers must not break the failure
        # report (sorted() over mixed types raised TypeError pre-fix) and
        # the etags carried back must be the ints the gateway accepts
        f = self._file(300)
        plan = {"file": f, "mode": "multipart", "part_size": 50,
                "size_bytes": 300,
                "parts": [{"part_number": str(n),
                           "url": ("https://dead/p2" if n == 2
                                   else f"https://ok/p{n}")}
                          for n in range(1, 7)]}
        rc, out, _ = self._run(plan, fail_urls={"https://dead/p2"})
        self.assertEqual(rc, 1)
        self.assertEqual(out["failed_parts"], [2])
        self.assertEqual(out["errors"]["2"], "HTTP 403")
        self.assertEqual([r["part_number"] for r in out["etags"]],
                         [1, 3, 4, 5, 6])
        self.assertTrue(all(isinstance(r["part_number"], int)
                            for r in out["etags"]))

    def test_done_rows_without_local_md5_are_trusted_and_not_reemitted_with_one(self):
        f = self._file(300)
        plan = {"file": f, "response": self._resp(50, 6),
                "done": [{"part_number": n, "etag": f"e{n}"} for n in (1, 2)]}
        rc, out, calls = self._run(plan)
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 4)              # parts 3..6 only
        by_n = {r["part_number"]: r for r in out["etags"]}
        self.assertNotIn("local_md5", by_n[1])       # not re-read this run
        self.assertIn("local_md5", by_n[3])

    def test_same_length_rewrite_between_runs_is_refused_by_the_done_md5(self):
        # a size check cannot see this; the done rows' local_md5 can
        f = self._file(300, seed=b"\x5a")
        plan = {"file": f, "response": self._resp(50, 6),
                "parts": [{"part_number": n, "url": f"https://ok/p{n}"}
                          for n in (3, 4, 5, 6)]}
        rc, first, _ = self._run({"file": f, "response": self._resp(50, 6)})
        self.assertEqual(rc, 0)
        self._file(300, seed=b"\xa5")                # rewrite, SAME length
        plan["done"] = first["etags"][:2]            # verbatim rows 1..2
        rc, out, calls = self._run(plan)
        self.assertEqual(rc, 1)
        self.assertEqual(calls, [])                  # refused before any PUT
        self.assertIn("changed since the previous run", out["errors"]["plan"])

    def test_response_pasted_as_json_text_is_parsed_not_fatal(self):
        # MCP results often arrive as JSON text; the verbatim paste of
        # that text must work, and non-JSON text must refuse by name
        import json as _json
        f = self._file(300)
        rc, out, _ = self._run({"file": f,
                                "response": _json.dumps(self._resp(200, 2))})
        self.assertEqual(rc, 0)
        rc, out, _ = self._run({"file": f, "response": "{not json"})
        self.assertEqual(rc, 1)
        self.assertIn("not valid JSON", out["errors"]["plan"])

    def test_missing_urls_are_plan_errors_not_transfer_errors(self):
        # SKILL step 3 routes errors.single to "re-run start_upload" and
        # errors.plan to "fix the plan" — a plan missing its url must
        # land in the second bucket or the agent loops on fresh URLs
        rc, out, _ = self._run({"file": self._file(10), "mode": "single",
                                "content_type": "a/b"})
        self.assertEqual(rc, 1)
        self.assertIn("url", out["errors"]["plan"])
        rc, out, _ = self._run({"file": self._file(300),
                                "mode": "multipart", "part_size": 150,
                                "size_bytes": 300,
                                "parts": [{"part_number": 1, "url": "u"},
                                          {"part_number": 2}]})
        self.assertEqual(rc, 1)
        self.assertIn("no url", out["errors"]["plan"])

    def test_part_count_is_a_clash_key_beside_a_response(self):
        f = self._file(300)
        rc, out, _ = self._run({"file": f, "response": self._resp(50, 6),
                                "part_count": 6})
        self.assertEqual(rc, 1)
        self.assertIn("keep exactly one shape", out["errors"]["plan"])

    def test_inconsistent_size_bytes_and_part_count_are_refused(self):
        # CONFIRMED repro from review: a re-stat'ed size_bytes beside a
        # stale response plan passed every guard and silently truncated
        f = self._file(320)
        rc, out, calls = self._run({"file": f, "size_bytes": 320,
                                    "response": self._resp(50, 6)})
        self.assertEqual(rc, 1)
        self.assertIn("means 7 parts", out["errors"]["plan"])
        self.assertEqual(calls, [])

    def test_empty_response_block_does_not_waive_explicit_requirements(self):
        f = self._file(300)
        rc, out, _ = self._run({"file": f, "response": {},
                                "mode": "multipart", "part_size": 150,
                                "parts": [{"part_number": 1, "url": "a"},
                                          {"part_number": 2, "url": "b"}]})
        self.assertEqual(rc, 1)
        self.assertIn("size_bytes", out["errors"]["plan"])

    def test_malformed_rows_are_plan_errors_not_fatals(self):
        f = self._file(300)
        base = {"file": f, "response": self._resp(50, 6)}
        rc, out, _ = self._run(dict(base, done=[{"part_number": "three",
                                                 "etag": "e"}]))
        self.assertEqual(rc, 1)
        self.assertIn("integer part_number", out["errors"]["plan"])
        rc, out, _ = self._run(dict(base, done=[{"part_number": 1,
                                                 "local_md5": "x"}]))
        self.assertEqual(rc, 1)
        self.assertIn("no etag", out["errors"]["plan"])
        rc, out, _ = self._run(dict(base, parts={"part_number": 1}))
        self.assertEqual(rc, 1)
        self.assertIn("must be a list", out["errors"]["plan"])

    def test_single_mode_transfer_failure_shape_is_errors_single(self):
        # SKILL step 3 routes errors.single to "re-run start_upload";
        # pin the key and that a failed single carries no etag/local_md5
        f = self._file(10)
        rc, out, _ = self._run({"file": f,
                                "response": {"mode": "single",
                                             "url": "https://dead/one",
                                             "content_type": "a/b"}},
                               fail_urls={"https://dead/one"})
        self.assertEqual(rc, 1)
        self.assertEqual(out["errors"]["single"], "HTTP 403")
        self.assertEqual(out["etags"], [])
        self.assertNotIn("etag", out)
        self.assertNotIn("local_md5", out)

    def test_response_size_bytes_echo_upgrades_the_guard_to_exact(self):
        # newer gateways echo size_bytes; a same-bucket size drift that
        # the bucket inequality cannot see must now refuse exactly
        f = self._file(290)                       # same 6x50 bucket as 300
        resp = dict(self._resp(50, 6), size_bytes=300)
        rc, out, calls = self._run({"file": f, "response": resp})
        self.assertEqual(rc, 1)
        self.assertIn("made for 300", out["errors"]["plan"])
        self.assertEqual(calls, [])

    def test_non_numeric_part_size_is_a_plan_error(self):
        f = self._file(300)
        rc, out, _ = self._run({"file": f, "mode": "multipart",
                                "part_size": "128MiB", "size_bytes": 300,
                                "parts": [{"part_number": 1, "url": "u"}]})
        self.assertEqual(rc, 1)
        self.assertIn("part_size must be a positive integer",
                      out["errors"]["plan"])

    def test_strict_scalars_refuse_floats_bools_and_bad_workers(self):
        f = self._file(300)
        base = {"file": f, "mode": "multipart", "part_size": 150,
                "parts": [{"part_number": 1, "url": "a"},
                          {"part_number": 2, "url": "b"}]}
        rc, out, _ = self._run(dict(base, size_bytes=300.5))
        self.assertEqual(rc, 1)
        self.assertIn("size_bytes must be a positive integer",
                      out["errors"]["plan"])
        rc, out, _ = self._run(dict(base, size_bytes=300, workers=-2))
        self.assertEqual(rc, 1)
        self.assertIn("workers must be a positive integer",
                      out["errors"]["plan"])

    def test_whole_refresh_result_pasted_as_the_patch_is_unwrapped(self):
        # the refresh reply is {parts: [...], expires_at: ...}; pasting it
        # verbatim must work as well as pasting just its parts list
        f = self._file(300)
        rc, out, calls = self._run(
            {"file": f, "response": self._resp(50, 6),
             "parts": {"parts": [{"part_number": 3,
                                  "url": "https://fresh/p3"}],
                       "expires_at": "2026-10-14T00:00:00Z"},
             "done": [{"part_number": n, "etag": f"e{n}"}
                      for n in (1, 2, 4, 5, 6)]})
        self.assertEqual(rc, 0)
        self.assertEqual([c["url"] for c in calls], ["https://fresh/p3"])

    def test_size_bytes_disagreeing_with_the_echo_is_refused_by_name(self):
        f = self._file(290)
        resp = dict(self._resp(50, 6), size_bytes=300)
        rc, out, calls = self._run({"file": f, "size_bytes": 290,
                                    "response": resp})
        self.assertEqual(rc, 1)
        self.assertIn("disagrees with the response's echo",
                      out["errors"]["plan"])
        self.assertEqual(calls, [])

    def test_single_mode_ignores_a_stray_workers_value(self):
        # SKILL says workers is multipart-only; a stray value must not
        # refuse a single-mode plan
        rc, out, _ = self._run({"file": self._file(10),
                                "response": {"mode": "single", "url": "u",
                                             "content_type": "a/b"},
                                "workers": "six"})
        self.assertEqual(rc, 0)

    def test_single_mode_unreadable_file_lands_in_errors_single(self):
        # the reader is constructed inside the transfer try, so an open
        # failure routes to the documented single-mode branch, not fatal
        import os as _os
        f = self._file(10)
        plan_path = os.path.join(self.tmp, "gone.bin")
        _os.rename(f, plan_path)

        real_getsize = _os.path.getsize

        def fake_getsize(p):
            if p == f:
                return 10          # stat succeeds, then the open races
            return real_getsize(p)

        with mock.patch.object(self.up.os.path, "getsize", fake_getsize):
            rc, out, _ = self._run({"file": f, "mode": "single",
                                    "url": "u", "content_type": "a/b"})
        self.assertEqual(rc, 1)
        self.assertIn("single", out["errors"])

    def test_workers_zero_or_false_refuses_instead_of_defaulting(self):
        f = self._file(300)
        for bad in (0, False):
            rc, out, _ = self._run({"file": f,
                                    "response": self._resp(200, 2),
                                    "workers": bad})
            self.assertEqual(rc, 1)
            self.assertIn("workers must be a positive integer",
                          out["errors"]["plan"])

    def test_non_object_plan_file_is_a_plan_error_not_fatal(self):
        rc, out, _ = self._run([{"file": "/nope"}])
        self.assertEqual(rc, 1)
        self.assertIn("JSON object", out["errors"]["plan"])

    def test_rehash_read_error_is_a_plan_error_and_cancels(self):
        f = self._file(300)
        real = self.up._BoundedReader
        state = {"first": True}

        def flaky(path, offset, length):
            if state["first"]:
                state["first"] = False
                raise OSError(5, "Input/output error")
            return real(path, offset, length)

        plan = {"file": f, "response": self._resp(50, 6),
                "done": [{"part_number": 1, "etag": "e1",
                          "local_md5": "0" * 32}]}
        with mock.patch.object(self.up, "_BoundedReader", flaky):
            rc, out, _ = self._run(plan)
        self.assertEqual(rc, 1)
        self.assertIn("cannot re-read part 1", out["errors"]["plan"])

    def test_json_syntax_error_in_the_plan_is_a_plan_error(self):
        # step 4 has the agent hand-edit the plan; a trailing comma must
        # route to "fix the plan", not errors.fatal
        import contextlib
        import io
        import json as _json
        plan_path = os.path.join(self.tmp, "broken.json")
        with open(plan_path, "w") as f:
            f.write('{"file": "/x",}')
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = self.up.main(plan_path)
        out = _json.loads(buf.getvalue().strip())
        self.assertEqual(rc, 1)
        self.assertIn("not valid JSON", out["errors"]["plan"])

    def test_workers_null_defaults_like_an_absent_key(self):
        f = self._file(300)
        rc, out, calls = self._run({"file": f,
                                    "response": self._resp(200, 2),
                                    "workers": None})
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 2)

    def test_unrecognized_mode_is_named_not_reported_as_missing(self):
        rc, out, _ = self._run({"file": self._file(10), "mode": "Single",
                                "url": "u", "content_type": "a/b"})
        self.assertEqual(rc, 1)
        self.assertIn("unrecognized mode", out["errors"]["plan"])

    def test_mangled_part_size_refuses_fast_without_materializing_ranges(self):
        # 5 GiB with part_size mangled to 8 would need ~671e6 part
        # numbers — the coverage check must refuse on count+bounds, never
        # allocate the range
        import time
        f = self._file(300)
        t0 = time.time()
        rc, out, _ = self._run({"file": f, "mode": "multipart",
                                "part_size": 8,
                                "size_bytes": 5 * 1024**3,
                                "parts": [{"part_number": 1, "url": "u"}]})
        self.assertEqual(rc, 1)
        self.assertIn("transcription slip", out["errors"]["plan"])
        self.assertLess(time.time() - t0, 2.0)

    def test_connection_error_on_a_part_lands_in_its_own_row(self):
        # SKILL step 4 routes connection-shaped part errors like 403s —
        # pin that they arrive as errors.<n> with the exception repr
        import urllib.error as _ue
        f = self._file(300)
        boom = _ue.URLError(BrokenPipeError(32, "Broken pipe"))
        rc, out, _ = self._run(
            {"file": f, "mode": "multipart", "part_size": 50,
             "size_bytes": 300,
             "parts": [{"part_number": n,
                        "url": ("https://dead/p2" if n == 2
                                else f"https://ok/p{n}")}
                       for n in range(1, 7)]},
            fail_urls={"https://dead/p2": boom})
        self.assertEqual(rc, 1)
        self.assertEqual(out["failed_parts"], [2])
        self.assertIn("URLError", out["errors"]["2"])
        self.assertIn("BrokenPipeError", out["errors"]["2"])

    def test_success_output_carries_a_paste_ready_parts_list(self):
        f = self._file(300)
        rc, out, _ = self._run({"file": f, "response": self._resp(200, 2)})
        self.assertEqual(rc, 0)
        self.assertEqual(out["parts"],
                         [{"part_number": r["part_number"],
                           "etag": r["etag"]} for r in out["etags"]])

    def test_done_row_with_empty_local_md5_refuses_by_name(self):
        f = self._file(300)
        rc, out, _ = self._run({"file": f, "response": self._resp(50, 6),
                                "done": [{"part_number": 1, "etag": "e1",
                                          "local_md5": ""}]})
        self.assertEqual(rc, 1)
        self.assertIn("empty local_md5", out["errors"]["plan"])

    def test_workers_above_the_bound_refuses_by_name(self):
        rc, out, _ = self._run({"file": self._file(300),
                                "response": self._resp(200, 2),
                                "workers": 6000})
        self.assertEqual(rc, 1)
        self.assertIn("workers must be 1..32", out["errors"]["plan"])

    def test_patch_pasted_as_json_text_is_parsed_like_the_response(self):
        import json as _json
        f = self._file(300)
        rc, out, calls = self._run(
            {"file": f, "response": self._resp(50, 6),
             "parts": _json.dumps({"parts": [{"part_number": 3,
                                              "url": "https://fresh/p3"}],
                                   "expires_at": "x"}),
             "done": [{"part_number": n, "etag": f"e{n}"}
                      for n in (1, 2, 4, 5, 6)]})
        self.assertEqual(rc, 0)
        self.assertEqual([c["url"] for c in calls], ["https://fresh/p3"])


    def test_plan_refusals_after_done_parsing_echo_resume_state(self):
        # the contract says the failure JSON's etags are what the agent
        # feeds back as done — a plan refusal must not wipe landed parts
        f = self._file(320)                       # exact-size mismatch
        done = [{"part_number": n, "etag": f"e{n}"} for n in (1, 2)]
        rc, out, _ = self._run({"file": f, "size_bytes": 300,
                                "mode": "multipart", "part_size": 50,
                                "parts": [{"part_number": n, "url": "u"}
                                          for n in range(3, 7)],
                                "done": done})
        self.assertEqual(rc, 1)
        self.assertIn("plan", out["errors"])
        self.assertEqual(out["etags"], done)

    def test_fatal_path_still_emits_the_etags_key(self):
        # the __main__ contract: one json line on EVERY path, and both
        # failure shapes carry "etags" (step 4 copies it unconditionally)
        import json as _json
        import subprocess
        import sys as _sys
        path = os.path.join(_ROOT, "skills", "upload-media", "references",
                            "uploader.py")
        r = subprocess.run([_sys.executable, path,
                            os.path.join(self.tmp, "missing.json")],
                           capture_output=True, text=True)
        out = _json.loads(r.stdout.strip())
        self.assertFalse(out["ok"])
        self.assertEqual(out["etags"], [])
        self.assertIn("fatal", out["errors"])
