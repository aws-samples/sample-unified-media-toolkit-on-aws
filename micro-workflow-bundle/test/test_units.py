"""Unit tests for the micro-workflow-bundle root template and build scripts.

Every test here encodes an invariant that a review had to derive by hand against
the child stacks, so the next review does not have to. Each docstring names
the bug it catches.

The bundle has no application code — its entire failure surface is
"does the root template agree with the children it nests, and does the
build script compose keys that actually exist in S3". Both are mechanically
checkable offline, which is what this suite does.

    cd micro-workflow-bundle && python3 -m unittest discover -s test
"""

import ast
import contextlib
import datetime
import importlib.util
import json
import os
import re
import sys
import types
import unittest

import yaml


BUNDLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCKER_ROOT = os.path.dirname(BUNDLE_DIR)

# logical resource name in cfnstack.yaml -> child directory under docker/
CHILDREN = {
    "MediaProcessingStack": "media-processing-on-aws",
    "AudiostemSeparationStack": "audiostem-separation-on-aws",
    "SpeechEnhancementStack": "speech-enhancement-on-aws",
    "SpeechDiarizationStack": "speech-diarization-on-aws",
    "ShotSegmentationStack": "shot-segmentation-on-aws",
    "FaceRecognitionStack": "face-recognition-on-aws",
    "HybridSearchStack": "hybrid-search-on-aws",
    "TtsStack": "tts-on-aws",
    "BedrockInferenceStack": "bedrock-inference-on-aws",
    "PromptRegistryStack": "prompt-registry-on-aws",
    "ConversationAnalysisStack": "conversation-analysis-on-aws",
    "ShotMetadataEnrichmentStack": "shot-metadata-enrichment-on-aws",
    "FrameInspectionStack": "frame-inspection-on-aws",
    "ProgramStructureAnalysisStack": "program-structure-analysis-on-aws",
    "AudioDescriptionStack": "audio-description-on-aws",
    "CastResolutionStack": "cast-resolution-on-aws",
    "ArtworkSelectionStack": "artwork-selection-on-aws",
    "M2cAgentToolsStack": "media-toolkit-gateway",
}

# Directories in this tree that are deployable stacks but deliberately NOT
# bundle children. EMPTY today: everything here ships in the bundle. A name added
# here is a statement, not a formality — it exempts that stack from every
# cross-wiring assertion in this suite.
# audio-description joined as the fifteenth child 2026-08-31 (user
# decision, same day its first live standalone run succeeded — ahead of
# the 28-episode corpus gate). Nothing is currently exempt.
NOT_BUNDLE_CHILDREN = frozenset()

# keyprefix token suffix per child, mirroring the WORKFLOWS table in
# build-deploy-s3-dist.sh
TOKEN_SUFFIXES = {
    "MediaProcessingStack": "MEDIA_PROCESSING",
    "AudiostemSeparationStack": "AUDIOSTEM_SEPARATION",
    "SpeechEnhancementStack": "SPEECH_ENHANCEMENT",
    "SpeechDiarizationStack": "SPEECH_DIARIZATION",
    "ShotSegmentationStack": "SHOT_SEGMENTATION",
    "FaceRecognitionStack": "FACE_RECOGNITION",
    "HybridSearchStack": "HYBRID_SEARCH",
    "TtsStack": "TTS",
    "BedrockInferenceStack": "BEDROCK_INFERENCE",
    "PromptRegistryStack": "PROMPT_REGISTRY",
    "ConversationAnalysisStack": "CONVERSATION_ANALYSIS",
    "ShotMetadataEnrichmentStack": "SHOT_METADATA_ENRICHMENT",
    "FrameInspectionStack": "FRAME_INSPECTION",
    "ProgramStructureAnalysisStack": "PROGRAM_STRUCTURE_ANALYSIS",
    "AudioDescriptionStack": "AUDIO_DESCRIPTION",
    "CastResolutionStack": "CAST_RESOLUTION",
    "ArtworkSelectionStack": "ARTWORK_SELECTION",
    "M2cAgentToolsStack": "M2C_AGENT_TOOLS",
}



@contextlib.contextmanager
def gateway_import_isolation(child_dir):
    """sys.path + sys.modules isolation for importing ONE child's gateway
    module — the ONE home (review, 2026-09-06: two loaders carried this
    ~20-line dance as near-identical copies and every new isolation
    requirement was a lockstep edit; missing one silently imported the
    PREVIOUS child's cached module — the stale-module failure found by
    mutation, not inspection).

    Path entries: gateway_tool/ (the flat zip's root at runtime),
    the child ROOT (bedrock-inference's gateway imports its own
    lib.identity/lib.input_schema package), and child_root/lib
    (cast-resolution's gateway imports its `pb_common` TOP-LEVEL, the way
    `zip -j` re-roots lib/ next to the dispatcher). Evicted around the
    import: `workflow_identity` (every child carries its own copy — the
    first one loaded must not shadow the rest), `pb_common` (the one
    flat-zip lib module a gateway imports top-level), and the `lib`
    package tree (a stale `lib` from a previous child would shadow this
    one's). Yields the gateway_tool directory."""
    gw_dir = os.path.join(DOCKER_ROOT, child_dir, "gateway_tool")
    child_root = os.path.join(DOCKER_ROOT, child_dir)
    child_lib = os.path.join(child_root, "lib")
    sys.path.insert(0, gw_dir)
    sys.path.insert(1, child_root)
    sys.path.insert(2, child_lib)
    cached = sys.modules.pop("workflow_identity", None)
    cached_pb = sys.modules.pop("pb_common", None)
    cached_lib = {k: sys.modules.pop(k) for k in list(sys.modules)
                  if k == "lib" or k.startswith("lib.")}
    try:
        yield gw_dir
    finally:
        sys.path.remove(gw_dir)
        sys.path.remove(child_root)
        sys.path.remove(child_lib)
        sys.modules.pop("workflow_identity", None)
        sys.modules.pop("pb_common", None)
        for k in list(sys.modules):
            if k == "lib" or k.startswith("lib."):
                sys.modules.pop(k)
        if cached is not None:
            sys.modules["workflow_identity"] = cached
        if cached_pb is not None:
            sys.modules["pb_common"] = cached_pb
        sys.modules.update(cached_lib)



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


class _CfnLoader(yaml.SafeLoader):
    """SafeLoader that tolerates CFN short-form intrinsics (!Sub, !GetAtt, ...).

    Tags are preserved as ("!Tag", value) tuples so a test can assert on the
    intrinsic that was used, not just on the resolved string.
    """


def _multi(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return ("!" + suffix, loader.construct_scalar(node))
    if isinstance(node, yaml.SequenceNode):
        return ("!" + suffix, loader.construct_sequence(node, deep=True))
    return ("!" + suffix, loader.construct_mapping(node, deep=True))


_CfnLoader.add_multi_constructor("!", _multi)


def _load(path):
    with open(path) as f:
        return yaml.load(f, Loader=_CfnLoader)  # nosec B506


def _read(path):
    with open(path) as f:
        return f.read()


class BundleTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root_path = os.path.join(BUNDLE_DIR, "cfnstack.yaml")
        cls.root = _load(cls.root_path)
        cls.root_text = _read(cls.root_path)
        cls.build = _read(os.path.join(BUNDLE_DIR, "build-deploy-s3-dist.sh"))
        cls.deploy = _read(os.path.join(BUNDLE_DIR, "create_update_stack.sh"))
        cls.child = {
            name: _load(os.path.join(DOCKER_ROOT, d, "cfnstack.yaml"))
            for name, d in CHILDREN.items()
        }
        cls.child_build = {
            name: _read(os.path.join(DOCKER_ROOT, d, "build-deploy-s3-dist.sh"))
            for name, d in CHILDREN.items()
        }

    def nested(self):
        """The AWS::CloudFormation::Stack resources only.

        The root is no longer nothing-but-children: it also declares the
        conditional proxy bucket and its policy. Tests that walk "every child"
        must filter by TYPE rather than assuming every resource is a child, or
        they crash looking up 'BundleProxyBucket' in the CHILDREN map.
        """
        return {
            name: node
            for name, node in self.root["Resources"].items()
            if node["Type"] == "AWS::CloudFormation::Stack"
        }


class TestNestedStackWiring(BundleTestCase):
    """The root template's whole job is passing the right values to the right
    children. Every failure mode here surfaces only at deploy time, and the
    cheapest one still costs media-processing's 90-minute CodeBuild first.
    """

    def test_every_deployable_stack_in_the_tree_is_a_registered_child(self):
        """The one direction nothing else covers. Every other test here iterates
        CHILDREN, so a new workflow that builds and deploys but never joins that
        map is INVISIBLE to the suite: it passes, the bundle simply does not ship
        the child, and the gap surfaces as a missing gateway target on a live
        deploy. Discovering from disk names the omission instead.
        """
        found = {
            name for name in os.listdir(DOCKER_ROOT)
            if name != os.path.basename(BUNDLE_DIR)
            and os.path.isfile(os.path.join(DOCKER_ROOT, name, "cfnstack.yaml"))
            and os.path.isfile(os.path.join(DOCKER_ROOT, name,
                                            "build-deploy-s3-dist.sh"))
        } - NOT_BUNDLE_CHILDREN
        self.assertEqual(
            set(CHILDREN.values()), found,
            "a deployable stack in this tree is not a registered bundle child — "
            "add it to CHILDREN, TOKEN_SUFFIXES, the WORKFLOWS table and the "
            "root template, or name it in NOT_BUNDLE_CHILDREN")

    def test_exactly_the_registered_children_are_nested(self):
        """'All of them, always' is a design decision, not a default. If a child
        were dropped, its resources would be deleted on the next stack update —
        and the children's DynamoDB tables carry no DeletionPolicy: Retain (by
        design), so the deletion is real. Asserted against CHILDREN as a set, so
        a new child needs no edit here beyond joining that map.
        """
        self.assertEqual(set(self.nested()), set(CHILDREN))

    def test_the_root_owns_nothing_but_children_and_the_proxy_bucket(self):
        """Kept as a deliberate inventory. The bundle's whole premise is that it
        composes children and owns almost nothing itself, so every non-child
        resource is a decision that should be visible here rather than
        accumulating quietly.
        """
        self.assertEqual(
            {"BundleProxyBucket", "BundleProxyBucketPolicy"},
            set(self.root["Resources"]) - set(CHILDREN),
        )

    def test_every_required_child_parameter_is_supplied(self):
        """A child parameter with no Default that the root fails to pass makes
        create-stack fail on that nested stack. Nearly every child defines NO
        defaults at all, so effectively every value must come from the root.
        """
        for name, node in self.nested().items():
            passed = set((node["Properties"].get("Parameters") or {}).keys())
            cparams = self.child[name].get("Parameters", {})
            required = {
                k for k, v in cparams.items() if "Default" not in (v or {})
            }
            self.assertEqual(
                set(), required - passed,
                f"{name}: root does not supply required {sorted(required - passed)}",
            )

    def test_no_unknown_parameter_is_passed_to_any_child(self):
        """Passing a parameter a child does not declare is a hard CFN validation
        error. This catches a child renaming or removing a parameter without the
        root following.
        """
        for name, node in self.nested().items():
            passed = set((node["Properties"].get("Parameters") or {}).keys())
            declared = set(self.child[name].get("Parameters", {}))
            self.assertEqual(
                set(), passed - declared,
                f"{name}: root passes undeclared {sorted(passed - declared)}",
            )

    def test_every_getatt_names_an_output_the_child_declares(self):
        """!GetAtt <X>Stack.Outputs.<Y> against an output the child does not
        declare fails at deploy time, not at validate time. This is the single
        most likely way a child edit silently breaks the bundle.
        """
        refs = re.findall(r"!GetAtt (\w+)\.Outputs\.(\w+)", self.root_text)
        self.assertGreaterEqual(len(refs), 19, "expected the full cross-wiring")
        for stack, output in refs:
            self.assertIn(stack, self.child, f"!GetAtt unknown stack {stack}")
            self.assertIn(
                output, self.child[stack].get("Outputs", {}),
                f"{stack} declares no output {output}",
            )

    def test_media_processing_arn_is_wired_from_the_output_not_a_parameter(self):
        """The MP parameter has no Default in any consumer, so media-processing
        is structurally required. Wiring it via !GetAtt is also what gives CFN
        the dependency edge that orders the deploy — there is no explicit
        DependsOn anywhere in this template.

        All three MP consumers converted to the gateway+callback sibling
        pattern (2026-09-18) and take the GATEWAY Lambda ARN instead of the
        machine ARN (MP's cost is read from its published
        usage_tracking.json, so the machine ARN parameter is gone entirely).
        The machine set is empty by design — kept so a future direct-machine
        consumer must be added HERE, visibly, rather than slipping in.
        """
        machine_arn_consumers = ()
        gateway_arn_consumers = ("AudiostemSeparationStack",
                                 "SpeechEnhancementStack",
                                 "SpeechDiarizationStack",
                                 "FaceRecognitionStack")
        for name in machine_arn_consumers:
            params = self.root["Resources"][name]["Properties"]["Parameters"]
            self.assertEqual(
                ("!GetAtt", "MediaProcessingStack.Outputs.StateMachineArn"),
                params["MediaProcessingStateMachineArn"], name,
            )
            self.assertNotIn("MediaProcessingGatewayLambdaArn", params, name)
        for name in gateway_arn_consumers:
            params = self.root["Resources"][name]["Properties"]["Parameters"]
            self.assertEqual(
                ("!GetAtt", "MediaProcessingStack.Outputs.GatewayToolLambdaArn"),
                params["MediaProcessingGatewayLambdaArn"], name,
            )
            self.assertNotIn("MediaProcessingStateMachineArn", params, name)

    def test_ingest_role_output_covers_exactly_the_ingest_reading_children(self):
        """IngestBucketReadRoleArns (2026-09-21): input for the ingest
        bucket owner's bucket policy. Three sets must agree, discovered
        rather than hand-kept so a child gaining or losing ingest access
        fails here instead of silently shipping a stale role list:

          A. children whose template has an IAM role whose Policies
             reference IngestBucket (the ground truth),
          B. children declaring the IngestBucketReadRoleArns output,
          C. children the root's aggregate output joins.
        """
        ground_truth = set()
        for name, tpl in self.child.items():
            for res in tpl.get("Resources", {}).values():
                if res.get("Type") != "AWS::IAM::Role":
                    continue
                pol = json.dumps(res.get("Properties", {}).get("Policies", []))
                if "IngestBucket" in pol:
                    ground_truth.add(name)
                    break
        declared = {name for name, tpl in self.child.items()
                    if "IngestBucketReadRoleArns" in tpl.get("Outputs", {})}
        joined = {m.group(1) for m in re.finditer(
            r"!GetAtt (\w+)\.Outputs\.IngestBucketReadRoleArns",
            self.root_text)}
        self.assertEqual(ground_truth, declared,
                         "a child's ingest access and its role-list output "
                         "disagree — add/remove IngestBucketReadRoleArns "
                         "in the child template")
        self.assertEqual(declared, joined,
                         "the root's IngestBucketReadRoleArns aggregate "
                         "does not join exactly the declaring children")

    def test_hybrid_search_admin_tool_is_wired_unconditionally(self):
        """Decided posture: a bundle-deployed gateway IS an admin gateway. The
        admin target grants delete_collection (full sweep of LanceDB tables, S3
        Vectors indexes, DDB records). An earlier draft proposed an
        EnableHybridSearchAdminTools parameter; that was rejected. If this ever
        becomes conditional, the security posture changed silently.
        """
        params = self.root["Resources"]["M2cAgentToolsStack"]["Properties"]["Parameters"]
        self.assertEqual(
            ("!GetAtt", "HybridSearchStack.Outputs.AdminToolLambdaArn"),
            params["HybridSearchAdminToolLambdaArn"],
        )

    def test_gateway_stack_receives_every_child_tool_arn(self):
        """media-toolkit-gateway' optional-target Conditions are permanently true
        under 'all children, always', which is exactly what lets every reference
        be a plain !GetAtt with no !If wrapper. A missing ARN would silently drop
        that target from the gateway — tools/list would just be short, and until
        2026-08-19 four targets were absent from the harness's TOOLS map, so
        'short' would not even have failed (TestGatewayHarnessCoverage).
        """
        params = self.root["Resources"]["M2cAgentToolsStack"]["Properties"]["Parameters"]
        arn_params = {k for k in params if k.endswith("ToolLambdaArn")}
        for k in arn_params:
            tag, value = params[k]
            self.assertEqual("!GetAtt", tag, k)
            self.assertTrue(value.endswith("ToolLambdaArn"), f"{k} -> {value}")
        # DERIVED, not counted: every gateway-tool ARN a child declares as an
        # output must be consumed here, and nothing else may be. A hardcoded
        # count needed an edit per new child and said nothing about WHICH one
        # was missing; this names it. media-outputs is absent on purpose — it is
        # a gateway target whose Lambda lives inside media-toolkit-gateway, so no ARN
        # crosses the boundary for it.
        referenced = {tuple(params[k][1].split(".Outputs."))
                      for k in arn_params}
        # NOT_GATEWAY_TARGETS is a POLICY statement, not a formality:
        # frame-inspection is internal plumbing invoked by sibling workflows
        # through its gateway Lambda and is deliberately NOT an agent-visible
        # target (a deliberate decision, recorded in README.md "What it
        # deploys").
        NOT_GATEWAY_TARGETS = {"FrameInspectionStack"}
        declared = {(name, out)
                    for name in CHILDREN
                    if name != "M2cAgentToolsStack"
                    and name not in NOT_GATEWAY_TARGETS
                    for out in self.child[name].get("Outputs", {})
                    if out.endswith("ToolLambdaArn")}
        self.assertEqual(
            declared, referenced,
            "a child's gateway-tool ARN output is not wired into the gateway "
            "(or the gateway is wired to one that no longer exists)")

    def test_no_explicit_dependson(self):
        """Ordering is derived from the !GetAtt references. An explicit DependsOn
        would be redundant, and a WRONG one (e.g. forcing a child to wait on
        something it does not need) would serialize builds that run in parallel
        today — turning a ~90 minute create into a much longer one.

        Asserted against the PARSED template, not the raw text: the word appears
        in a comment explaining this very decision, so a text search self-defeats.
        """
        for name, node in self.nested().items():
            self.assertNotIn("DependsOn", node, f"{name} has an explicit DependsOn")


class TestPromptRegistryAdminParity(BundleTestCase):
    """The prompt-registry-admin target duplicates the standard target's
    seven tool schemas (CFN YAML supports no anchors) — this pins the
    two InlinePayloads byte-identical modulo the target name, so a
    schema edit cannot half-land (the drift class the admin surface's
    own design note names)."""

    def test_admin_target_payload_matches_the_standard_target(self):
        # one loader, one parse: reuse the module's CFN-tolerant loader
        tpl = _load(os.path.join(DOCKER_ROOT, "media-toolkit-gateway",
                                 "cfnstack.yaml"))
        r = tpl["Resources"]
        def payload(name):
            return (r[name]["Properties"]["TargetConfiguration"]["Mcp"]
                    ["Lambda"]["ToolSchema"]["InlinePayload"])
        self.assertEqual(payload("PromptRegistryTarget"),
                         payload("PromptRegistryAdminTarget"))
        self.assertEqual(
            r["PromptRegistryAdminTarget"]["Properties"]["Name"],
            "prompt-registry-admin")


class TestGatewayHarnessCoverage(BundleTestCase):
    """`scripts/test_gateway.py`'s TOOLS map is the ONLY thing that can produce a
    MISSING verdict: `expected` is built from that map alone, so a target absent
    from it contributes nothing to `missing` and its tools land in the
    `EXTRA (unexpected, not a failure)` bucket instead — the harness prints PASS
    and exits 0 no matter what happened to them.

    Not hypothetical. Measured 2026-08-19: bedrock-inference, prompt-registry,
    conversation-analysis and shot-metadata-enrichment were all absent from the
    map, so **18 of the gateway's 58 tools were unverifiable** — a target that
    silently failed to create still passed `tools/list`, which is the one check
    built to catch exactly that. The whole LLM family, unpinned.

    The map drifted because nothing compared it against the template. This is
    that comparison.
    """

    HARNESS = os.path.join(DOCKER_ROOT, "media-toolkit-gateway", "scripts",
                           "test_gateway.py")

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Parsed, never imported: the harness builds no clients at import time
        # today, but a test that executes the thing it audits inherits every
        # future import-time side effect.
        consts = {}
        for node in ast.parse(_read(cls.HARNESS)).body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                try:
                    consts[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass          # not a literal — irrelevant here
        cls.harness_tools = consts["TOOLS"]
        cls.harness_optional = consts["OPTIONAL_TARGETS"]

    def _template_targets(self):
        """{target name: (MCP tool names, is_conditional)} straight off
        media-toolkit-gateway' template — the source of truth for what tools/list
        serves. MCP composes the advertised name as `{target}___{tool}`."""
        out = {}
        for res in self.child["M2cAgentToolsStack"]["Resources"].values():
            if not str(res.get("Type", "")).endswith("GatewayTarget"):
                continue
            props = res["Properties"]
            name = props["Name"]
            payload = (props["TargetConfiguration"]["Mcp"]["Lambda"]
                       ["ToolSchema"]["InlinePayload"])
            out[name] = ({f"{name}___{t['Name']}" for t in payload},
                         "Condition" in res)
        return out

    def test_routing_skill_covers_every_gateway_target(self):
        """The canonical routing skill (skills/video-processing/SKILL.md) is
        how agents pick a tool; a target absent from it is a workflow agents
        cannot be routed to — the Media Harness fork drifted exactly this way
        (no speech-enhancement row, a stale stem-to-diarization recipe).
        Every GatewayTarget name in the template must appear in the router,
        and every reference link in the router must resolve through the
        references→prompts symlink. Adding a micro-workflow now fails here
        until someone writes its router row."""
        skill_dir = os.path.join(DOCKER_ROOT, "media-toolkit-gateway",
                                 "skills", "video-processing")
        skill = _read(os.path.join(skill_dir, "SKILL.md"))
        for name in self._template_targets():
            self.assertIn(
                name, skill,
                f"gateway target {name!r} has no mention in the "
                "video-processing routing skill — route it (or state "
                "explicitly why it is not routed)")
        refs = re.findall(r"\((references/[^)]+\.md)\)", skill)
        self.assertTrue(refs, "router has no reference links")
        for rel in refs:
            self.assertTrue(
                os.path.isfile(os.path.join(skill_dir, rel)),
                f"router links {rel} but it does not resolve — the "
                "references symlink must point at the gateway prompts/")
        self.assertTrue(
            os.path.islink(os.path.join(skill_dir, "references")),
            "references must be a SYMLINK to ../../prompts — a copied "
            "directory is a fork, and forks drift")

    def test_every_tool_name_in_skills_and_prompts_is_advertised(self):
        """A `{target}___{tool}` string in a skill or prompt is an instruction
        an agent will execute verbatim; a typo'd verb is a guaranteed -32602
        at the worst time (measured 2026-10-13: the routing skill shipped
        `select_artworks` for the singular `select_artwork` — the same guess,
        made twice, caught only by a live battery). Extract every such token
        from the gateway's skills/ and prompts/ trees and pin membership in
        the template's advertised tool set."""
        advertised = set()
        for name, (tools, _) in self._template_targets().items():
            advertised |= tools
        gw = os.path.join(DOCKER_ROOT, "media-toolkit-gateway")
        pat = re.compile(r"\b([a-z][a-z0-9-]*___[a-z][a-z0-9_]*)\b")
        bad = {}
        for base in ("skills", "prompts"):
            for dirpath, _dirs, files in os.walk(os.path.join(gw, base)):
                if "references" in dirpath.split(os.sep):
                    continue          # symlinked prompts are walked once
                for fn in files:
                    if not fn.endswith(".md"):
                        continue
                    rel = os.path.relpath(os.path.join(dirpath, fn), gw)
                    for tok in pat.findall(_read(os.path.join(dirpath, fn))):
                        if tok not in advertised:
                            bad.setdefault(tok, set()).add(rel)
        self.assertEqual(
            bad, {},
            "tool names cited in docs but not advertised by the gateway "
            "template (typo'd verb or retired tool): " + repr(bad))

    def test_harness_tools_map_matches_the_gateway_template(self):
        """Both directions. A template tool missing from the map is unverifiable
        (the EXTRA bucket swallows it); a map tool missing from the template is a
        permanent MISSING that makes the harness fail on a correct deploy, which
        trains people to stop trusting it."""
        template = self._template_targets()
        self.assertEqual(
            set(template), set(self.harness_tools),
            "TOOLS targets drifted from media-toolkit-gateway/cfnstack.yaml's "
            "GatewayTargets — a registered target absent from TOOLS is a "
            "target tools/list cannot fail on")
        for name, (tools, _) in sorted(template.items()):
            with self.subTest(target=name):
                self.assertEqual(
                    tools, set(self.harness_tools[name]),
                    f"{name}'s tool names drifted; the harness pins what "
                    "tools/list must advertise")

    def test_optional_targets_is_exactly_the_conditional_ones(self):
        """Membership decides whether an absent target is a deployment choice or
        a failure, so it must track the template's Conditions in BOTH
        directions. A conditional target missing here fails a legitimate partial
        deploy (empty ARN parameter). An UNCONDITIONAL one listed here is worse
        and silent: a total outage of a target that always ships reads as
        'not deployed — skipping' and passes."""
        conditional = {n for n, (_, cond) in self._template_targets().items()
                       if cond}
        self.assertEqual(
            conditional, set(self.harness_optional),
            "OPTIONAL_TARGETS drifted from the template's target Conditions")

    def test_every_target_lambda_is_invokable_by_the_gateway_role(self):
        """AgentCore validates AT TARGET CREATION that the gateway's
        execution role can invoke the target's Lambda — the resource-based
        Lambda::Permission alone is NOT enough. Measured 2026-08-31: the
        media-ingest target shipped with its Permission but without an
        InvokeToolLambdas entry, and the m2cv6 update rolled back with
        'Gateway execution role lacks permission to invoke Lambda'."""
        res = self.child["M2cAgentToolsStack"]["Resources"]
        policies = res["GatewayExecutionRole"]["Properties"]["Policies"]
        pol = next(p for p in policies
                   if p["PolicyName"] == "InvokeToolLambdas")
        allowed = set()
        for st in pol["PolicyDocument"]["Statement"]:
            if isinstance(st, tuple) and st[0] == "!If":
                st = st[1][1]                    # the enabled branch
            resources = st["Resource"]
            if not isinstance(resources, list):
                resources = [resources]
            allowed.update(repr(r) for r in resources)
        targets = [r["Properties"] for r in res.values()
                   if str(r.get("Type", "")).endswith("GatewayTarget")]
        self.assertTrue(targets)
        for props in targets:
            arn = props["TargetConfiguration"]["Mcp"]["Lambda"]["LambdaArn"]
            with self.subTest(target=props["Name"]):
                self.assertIn(
                    repr(arn), allowed,
                    f"{props['Name']}'s Lambda is not in the gateway "
                    "execution role's InvokeToolLambdas policy — target "
                    "creation will fail the whole stack update")

    def test_every_conditional_target_is_wired_unconditionally_by_the_bundle(self):
        """Those Conditions are permanently TRUE under 'all children, always', so
        the harness must see every target when run against a bundle deploy. If
        the bundle ever stopped passing one of the ARNs, the target would vanish
        and `optional_targets` would correctly absolve it — that is the right
        behavior for a standalone gateway and the wrong signal for this bundle,
        which is why the ARN wiring is asserted here too."""
        params = self.root["Resources"]["M2cAgentToolsStack"]["Properties"]["Parameters"]
        supplied = {k for k, v in params.items()
                    if k.endswith("ToolLambdaArn") and v[0] == "!GetAtt"}
        # every conditional target except hybrid-search-admin takes its ARN from
        # a same-named parameter; admin's comes from HybridSearchStack's second
        # output, so it is named for the target rather than the child.
        # media-ingest and media-uploads are the targets gated by a BUCKET,
        # not a sibling ARN — their Lambdas live inside media-toolkit-gateway
        # itself, and their shared gate is the IngestBucket parameter,
        # required at the bundle root.
        for target in sorted(self.harness_optional):
            if target in ("media-ingest", "media-uploads"):
                with self.subTest(target=target):
                    self.assertEqual(
                        params.get("IngestBucket"), ("!Ref", "IngestBucket"),
                        "the bundle does not pass IngestBucket, so the "
                        f"{target} target will not be created")
                continue
            expected = "".join(p.capitalize() for p in target.split("-"))
            with self.subTest(target=target):
                self.assertIn(f"{expected}ToolLambdaArn", supplied,
                              f"the bundle does not pass {target}'s ARN, so "
                              "the gateway will not serve it")


class TestInputExamplesMatchTheGatewaySchemas(BundleTestCase):
    """Each workflow's gateway_tool/input_examples.json is served by its
    describe_tool as the INPUT twin of the output contract: per tool, a
    `required` list (prose entries like 'a OR b' state one-of rules the CFN
    SchemaDefinition cannot express) and a minimal valid `example`. The
    repo suites round-trip the examples through their own validators; what
    nothing else compares is the examples file against the GATEWAY
    template — the schema agents actually receive. This is that
    comparison: tool coverage both ways, and every field the template
    marks Required acknowledged by the repo's required list. A prose
    one-of entry satisfies a Required field it names; frame-inspection is
    not a gateway target and the admin targets serve no examples file, so
    neither appears here."""

    # GatewayTarget Name -> child repo directory carrying the examples.
    TARGET_REPOS = {
        "audio-stem": "audiostem-separation-on-aws",
        "speech-enhancement": "speech-enhancement-on-aws",
        "speech-diarization": "speech-diarization-on-aws",
        "media-processing": "media-processing-on-aws",
        "shot-segmentation": "shot-segmentation-on-aws",
        "face-recognition": "face-recognition-on-aws",
        "hybrid-search": "hybrid-search-on-aws",
        "text-to-speech": "tts-on-aws",
        "bedrock-inference": "bedrock-inference-on-aws",
        "prompt-registry": "prompt-registry-on-aws",
        "conversation-analysis": "conversation-analysis-on-aws",
        "shot-metadata-enrichment": "shot-metadata-enrichment-on-aws",
        "program-structure-analysis": "program-structure-analysis-on-aws",
        "audio-description": "audio-description-on-aws",
        "artwork-selection": "artwork-selection-on-aws",
        "cast-resolution": "cast-resolution-on-aws",
    }

    def _target_payloads(self):
        out = {}
        for res in self.child["M2cAgentToolsStack"]["Resources"].values():
            if not str(res.get("Type", "")).endswith("GatewayTarget"):
                continue
            props = res["Properties"]
            out[props["Name"]] = (props["TargetConfiguration"]["Mcp"]
                                  ["Lambda"]["ToolSchema"]["InlinePayload"])
        return out

    def test_every_examples_target_is_mapped(self):
        """TARGET_REPOS tracks the template in both directions, minus the
        deliberate exclusions (admin surfaces, the non-target plumbing
        tools)."""
        payloads = self._target_payloads()
        excluded = {"hybrid-search-admin", "prompt-registry-admin",
                    "media-outputs", "media-ingest", "media-uploads"}
        self.assertEqual(set(self.TARGET_REPOS),
                         set(payloads) - excluded)

    def test_required_lists_match_the_template_schemas(self):
        payloads = self._target_payloads()
        for target, repo in sorted(self.TARGET_REPOS.items()):
            path = os.path.join(DOCKER_ROOT, repo, "gateway_tool",
                                "input_examples.json")
            with self.subTest(target=target):
                with open(path) as f:
                    tools = json.load(f)["tools"]
                payload = {t["Name"]: t for t in payloads[target]}
                # coverage both ways, against the TEMPLATE
                self.assertEqual(sorted(tools), sorted(payload),
                                 f"{repo}: examples drifted from the "
                                 "template's tool set")
                for name, entry in tools.items():
                    # `alternatives` is optional: present only on one-of
                    # tools, one valid example per other form. There is
                    # deliberately NO `optional` key: the non-required
                    # vocabulary's single home is the InputSchema that
                    # tools/list already delivers (user decision
                    # 2026-10-07 — duplicating the names here bought
                    # nothing an agent did not already hold).
                    self.assertIn(
                        sorted(entry),
                        (["example", "required"],
                         ["alternatives", "example", "required"]),
                        f"{repo}:{name}")
                    schema = payload[name].get("InputSchema") or {}
                    schema_req = schema.get("Required") or []
                    for field in schema_req:
                        self.assertTrue(
                            any(field == r or field in r.split()
                                or field in r.split("|")
                                for r in entry["required"]),
                            f"{repo}:{name}: template marks {field!r} "
                            "Required but the examples' required list "
                            "does not acknowledge it")
                    # every PLAIN required name is demonstrated by the
                    # example (one-of prose entries exempt)
                    for r in entry["required"]:
                        if " " not in r and "|" not in r:
                            self.assertIn(
                                r, entry["example"],
                                f"{repo}:{name}: required field {r!r} "
                                "missing from its own example")
                    # every key the example and alternatives use must be
                    # schema-declared — an undeclared key would pass the
                    # Lambda round-trip while teaching agents a parameter
                    # the schema never advertises (and the AgentCore edge
                    # may reject)
                    props = set(schema.get("Properties") or {})
                    shown = set(entry["example"])
                    for alt in (entry.get("alternatives") or {}).values():
                        shown |= set(alt)
                    self.assertFalse(
                        shown - props,
                        f"{repo}:{name}: example/alternatives use keys "
                        f"the schema does not declare: "
                        f"{sorted(shown - props)}")


class TestS3NamingConvention(BundleTestCase):
    """One suffix per family, ratcheted (user decision 2026-10-07, made
    pre-publication while every caller was still in-repo): parameters and
    response fields naming an S3 location end `_s3uri` — never `_s3_uri` —
    and the prefix family keeps its single existing spelling (`s3_prefix`,
    `output_s3_prefix`; `s3prefix` never). The census that forced this:
    26 occurrences said `_s3uri` against 10 `_s3_uri`, and the mix had
    already shipped one real bug (PSA's schema declared transcript_s3_uri
    while its code read transcript_s3uri, silently dropping the parameter
    for every schema-conformant caller)."""

    def test_no_schema_name_uses_the_retired_spellings(self):
        for res in self.child["M2cAgentToolsStack"]["Resources"].values():
            if not str(res.get("Type", "")).endswith("GatewayTarget"):
                continue
            props = res["Properties"]
            payload = (props["TargetConfiguration"]["Mcp"]["Lambda"]
                       ["ToolSchema"]["InlinePayload"])
            for t in payload:
                for side in ("InputSchema", "OutputSchema"):
                    for name in (t.get(side) or {}).get("Properties") or {}:
                        with self.subTest(target=props["Name"],
                                          tool=t["Name"], field=name):
                            self.assertNotIn(
                                "s3_uri", name,
                                "retired spelling — the family suffix "
                                "is _s3uri")
                            self.assertNotIn(
                                "s3prefix", name,
                                "the prefix family spells it s3_prefix")


class TestRootParameters(BundleTestCase):
    """The point of the bundle is collapsing ~15 child parameters into 7."""

    def test_exactly_seven_parameters(self):
        """PromptSeedMode is deliberately NOT the eighth (user decision
        2026-08-18): the bundle collapses child knobs, and the prompt-registry
        dev backdoor must not be handed to a prod deployer — it is a HARDCODED
        pre-GA literal on the nested stack instead (see the test named for the
        GA flip below)."""
        self.assertEqual(7, len(self.root["Parameters"]), sorted(self.root["Parameters"]))

    def test_prompt_seed_mode_is_the_childs_knob_not_the_bundles(self):
        """User decision 2026-08-18: the bundle stays out of the seed-mode
        knob entirely — the CHILD's parameter Default carries the pre-GA
        posture (overwrite-builtins), and the GA-flip obligation is a test
        named for it in prompt-registry's own suite. The bundle passing the
        parameter would put the posture in two places that can disagree."""
        params = (self.root["Resources"]["PromptRegistryStack"]
                  ["Properties"]["Parameters"])
        self.assertNotIn("PromptSeedMode", params)

    def test_only_the_ingest_bucket_has_no_default(self):
        """IngestBucket alone, and the asymmetry is the point.

        ProxyBucket used to be mandatory too, as a guard: six child validators
        CREATE the proxy bucket on a 404, so a typo silently succeeded and every
        output landed in a stray bucket. That reasoning is now OBSOLETE, not
        merely relaxed — the bundle declares the bucket when the parameter is
        blank, so the hazard is gone rather than tolerated. There is nothing to
        mistype.

        IngestBucket must never gain a default by the same argument run in
        reverse: it holds SOURCE MEDIA the bundle did not produce and cannot
        recreate, so a wrong value deploys clean and then finds nothing to
        process.
        """
        no_default = {
            k for k, v in self.root["Parameters"].items() if "Default" not in v
        }
        self.assertEqual({"IngestBucket"}, no_default)
        self.assertEqual("", self.root["Parameters"]["ProxyBucket"]["Default"])

    def test_gateway_defaults_match_the_child_verbatim(self):
        """If the bundle's default and media-toolkit-gateway' own default diverge, a
        standalone deploy and a bundle deploy of the same gateway would target
        different Cognito domains and secret paths — a silent fork.
        GatewayName is the ONE deliberate divergence (2026-09-23): the bundle
        defaults it EMPTY and derives {Prefix}-{stack-id} at the child wiring
        (bGatewayNameGiven), while the standalone child keeps its friendly
        default — a standalone deploy has no bundle stack id to derive from.
        """
        child_params = self.child["M2cAgentToolsStack"]["Parameters"]
        for key in ("Environment", "OAuthScope", "SecretsKmsKeyId"):
            self.assertEqual(
                child_params[key]["Default"],
                self.root["Parameters"][key]["Default"],
                f"{key} default diverges from media-toolkit-gateway",
            )
        self.assertEqual("", self.root["Parameters"]["GatewayName"]["Default"])
        self.assertEqual("media-toolkit-gateway",
                         child_params["GatewayName"]["Default"])
        self.assertIn("bGatewayNameGiven", self.root["Conditions"])

    def test_environment_allowed_values_match_the_child(self):
        """A value the root accepts but the child rejects fails on the nested
        stack, after the seven workflow children have already built.
        """
        self.assertEqual(
            self.child["M2cAgentToolsStack"]["Parameters"]["Environment"]["AllowedValues"],
            self.root["Parameters"]["Environment"]["AllowedValues"],
        )

    def test_parameter_groups_cover_every_parameter_exactly_once(self):
        """A parameter missing from ParameterGroups still deploys but vanishes
        from the console form, so a deployer cannot set it — which for
        IngestBucket/ProxyBucket means they cannot deploy at all from the
        console.
        """
        declared = set(self.root["Parameters"])
        grouped = []
        for group in self.root["Metadata"]["AWS::CloudFormation::Interface"]["ParameterGroups"]:
            grouped += group["Parameters"]
        self.assertEqual(sorted(declared), sorted(grouped))
        self.assertEqual(len(grouped), len(set(grouped)), "duplicate in ParameterGroups")

    def test_every_parameter_has_a_label(self):
        labels = set(self.root["Metadata"]["AWS::CloudFormation::Interface"]["ParameterLabels"])
        self.assertEqual(set(self.root["Parameters"]), labels)

    def test_the_buckets_are_the_first_group_ingest_first(self):
        """Asserts the ORDERING INTENT, not a count: the console must surface the
        buckets first, and IngestBucket — now the only parameter a deployer MUST
        fill in — must come before the optional ProxyBucket. Grouping them
        together is still right even though only one is required, because they
        are the pair a deployer reasons about at the same moment.
        """
        first = self.root["Metadata"]["AWS::CloudFormation::Interface"]["ParameterGroups"][0]
        self.assertEqual(["IngestBucket", "ProxyBucket"], first["Parameters"])
        self.assertLess(
            first["Parameters"].index("IngestBucket"),
            first["Parameters"].index("ProxyBucket"),
            "the required bucket must be surfaced before the optional one",
        )


class TestOutputs(BundleTestCase):
    def test_no_output_is_exported(self):
        """An exported value can never change, and the producing stack cannot be
        deleted while any importer exists. Hard-locking a stack that owns eight
        nested stacks is the worst possible application of that. media-toolkit-gateway
        documents the same reasoning for itself.
        """
        for name, node in self.root.get("Outputs", {}).items():
            self.assertNotIn("Export", node, f"{name} must not be exported")

    def test_gateway_url_and_secret_are_surfaced(self):
        """The bundle exists so a deployer needs one console page. These two are
        the entire answer to 'what do I hand my agent?'.
        """
        outputs = self.root["Outputs"]
        self.assertIn("GatewayUrl", outputs)
        self.assertIn("GatewayCredentialsSecretName", outputs)


class TestKeyPrefixTokens(BundleTestCase):
    """Token substitution is where a build-time mistake becomes a deploy-time
    failure ~90 minutes into a create."""

    def test_one_keyprefix_token_per_child(self):
        """Each child carries its OWN .version, so one shared KeyPrefix cannot
        address all eight. A missing token means that child's TemplateURL is
        wrong or literal.
        """
        tokens = set(re.findall(r"%%KEYPREFIX_[A-Z0-9_]+%%", self.root_text))
        expected = {f"%%KEYPREFIX_{s}%%" for s in TOKEN_SUFFIXES.values()}
        self.assertEqual(expected, tokens)

    def test_build_script_substitutes_every_token_in_the_template(self):
        """The WORKFLOWS table in the build script and the Mappings in the
        template are two lists that must agree. A token in one and not the other
        is the failure this catches.
        """
        template_tokens = set(re.findall(r"%%([A-Z0-9_]+)%%", self.root_text))
        # BUCKET_NAME and VERSION are substituted by name; the rest come from
        # the WORKFLOWS table suffixes.
        table = re.findall(r'"\s*([a-z0-9-]+)\|([A-Z0-9_]+)\s*"', self.build)
        # derived from the set, not a tally (see README.md: pin sets,
        # never tallies)
        self.assertEqual(len(TOKEN_SUFFIXES), len(table),
                         f"WORKFLOWS table parsed as {table}")
        covered = {"BUCKET_NAME", "VERSION"} | {
            f"KEYPREFIX_{suffix}" for _, suffix in table
        }
        self.assertEqual(
            set(), template_tokens - covered,
            f"template tokens never substituted: {sorted(template_tokens - covered)}",
        )

    def test_workflows_table_matches_the_template_children(self):
        """The table's directory list and the template's nested stacks are the
        same eight things expressed twice. Drift means the build stages a child
        the template never references, or vice versa.
        """
        table = re.findall(r'"\s*([a-z0-9-]+)\|([A-Z0-9_]+)\s*"', self.build)
        self.assertEqual(sorted(CHILDREN.values()), sorted(d for d, _ in table))
        self.assertEqual(
            sorted(TOKEN_SUFFIXES.values()), sorted(s for _, s in table)
        )

    def test_media_processing_is_built_first_and_gateway_last(self):
        """Build order matches deploy order so that a break in the child three
        others depend on surfaces before seven other builds report success. And
        media-toolkit-gateway last because it consumes all eight ARNs.
        """
        table = re.findall(r'"\s*([a-z0-9-]+)\|[A-Z0-9_]+\s*"', self.build)
        self.assertEqual("media-processing-on-aws", table[0])
        self.assertEqual("media-toolkit-gateway", table[-1])

    def test_leftover_token_guard_accepts_digits(self):
        """THE bug this file exists for. Both children that carry a leftover-token
        guard use '%%[A-Z_]*%%', which cannot match a digit — so it silently does
        NOT catch %%KEYPREFIX_M2C_AGENT_TOOLS%% (the '2' in M2C falls outside the
        class). No child template has a digit-bearing token, so their pattern is
        accidentally sufficient there; this template is the first with one. With
        the narrow class, a mis-substituted M2C key would ship a literal
        TemplateURL and fail the stack ~90 minutes into a create.
        """
        self.assertIn("%%[A-Z0-9_]*%%", self.build)
        self.assertNotIn(
            "grep -o '%%[A-Z_]*%%'", self.build,
            "the narrow character class cannot match KEYPREFIX_M2C_AGENT_TOOLS",
        )
        # And prove it, rather than only asserting the string is present.
        token = "%%KEYPREFIX_M2C_AGENT_TOOLS%%"  # nosec B105
        self.assertIsNone(re.search(r"%%[A-Z_]*%%", token))
        self.assertIsNotNone(re.search(r"%%[A-Z0-9_]*%%", token))


class TestChildContractTheBundleDependsOn(BundleTestCase):
    """The bundle reads WORKFLOW_NAME and TEMPLATE_NAME out of each child's build
    script to compose S3 keys. That parsing is a contract with eight files this
    directory does not own."""

    def test_every_child_declares_parsable_workflow_and_template_names(self):
        """If the bundle cannot parse either value it composes a wrong key, and
        the head-object check fails the build. That is the designed outcome — but
        it should never happen from a formatting change alone, so pin the format.
        """
        for name, text in self.child_build.items():
            wf = re.search(r'^WORKFLOW_NAME="([^"]+)"', text, re.M)
            tpl = re.search(r'^TEMPLATE_NAME="([^"]+)"', text, re.M)
            self.assertIsNotNone(wf, f"{name}: no parsable WORKFLOW_NAME")
            self.assertIsNotNone(tpl, f"{name}: no parsable TEMPLATE_NAME")

    def test_template_name_convention_holds_for_every_child(self):
        """tts hardcoded 'tts-on-aws-stack.template' through alpha.0803c while its
        seven siblings derived it, which would have forced a special case into the
        bundle. It was fixed in tts instead. This asserts the convention the
        bundle's key composition assumes.
        """
        for name, text in self.child_build.items():
            wf = re.search(r'^WORKFLOW_NAME="([^"]+)"', text, re.M).group(1)
            tpl = re.search(r'^TEMPLATE_NAME="([^"]+)"', text, re.M).group(1)
            resolved = tpl.replace("${WORKFLOW_NAME}", wf)
            self.assertEqual(
                f"{wf}-stack.template", resolved,
                f"{name}: TEMPLATE_NAME '{tpl}' breaks the convention",
            )
            self.assertNotIn("$", resolved, f"{name}: unresolved variable")

    def test_every_child_has_a_version_file(self):
        """No --version means the bundle reads each child's .version to compose
        its key, so the file has to be there and has to hold something. A missing
        or empty one aborts the build by design; this catches it in the suite
        instead. The bundle's own .version is checked too — it stamps the root
        template's artifact key the same way.

        **Existence and non-emptiness only here — the release grammar lives
        in its own test below.** The value is an opaque S3 key segment and
        ECR ImageTag; nothing downstream slices or parses it, and `--version`
        on the command line overrides the file anyway.
        """
        for name, d in list(CHILDREN.items()) + [("bundle", "micro-workflow-bundle")]:
            path = os.path.join(DOCKER_ROOT, d, ".version")
            self.assertTrue(os.path.isfile(path), f"{name}: missing {path}")
            self.assertTrue(_read(path).strip(), f"{name}: empty .version")

    def test_every_version_matches_the_release_grammar(self):
        """The format WAS deliberately unpinned (every convention change had
        to be argued past a test); re-pinned 2026-10-07 by user decision as
        a publication-readiness gate, with a closed vocabulary:

            {alpha|beta|preview|vNN}.YYYYMMDDx   (NN 1-2 digits, x a
                                                  lowercase letter)
            MAJOR.MINOR.PATCH                     (common semver)

        The legacy 4-digit alpha.MMDDx form is retired — it never ordered
        across a year boundary, and no file in the tree still carries it.
        A change of vocabulary is a user decision, not a test edit.
        """
        grammar = re.compile(
            r"^(?:(?:alpha|beta|preview|v[0-9]{1,2})\.[0-9]{8}[a-z]"
            r"|[0-9]+\.[0-9]+\.[0-9]+)$")
        for name, d in list(CHILDREN.items()) + [("bundle", "micro-workflow-bundle")]:
            value = _read(os.path.join(DOCKER_ROOT, d, ".version")).strip()
            with self.subTest(child=name):
                self.assertRegex(
                    value, grammar,
                    f"{name}: .version {value!r} is outside the release "
                    "grammar — allowed: alpha|beta|preview|vNN"
                    ".YYYYMMDDx, or MAJOR.MINOR.PATCH semver")

    def test_every_child_accepts_the_flags_the_bundle_passes(self):
        """The loop passes --bucket/--solution/--version/--single-region/--acl to
        every child. A child missing one would ignore it silently — the argument
        parsers all end in a catch-all '*) shift ;;' rather than an error.
        """
        for name, text in self.child_build.items():
            for flag in ("--bucket", "--solution", "--version",
                         "--single-region", "--acl"):
                self.assertIn(flag, text, f"{name} does not accept {flag}")


class TestBucketResolution(BundleTestCase):
    """The finding that shaped this template: the children are only
    self-consistent under --single-region."""

    def test_children_substitute_the_basename_but_upload_to_the_regional_bucket(self):
        """Measured, not assumed. Every child seds %%BUCKET_NAME%% with
        ${BUCKET_NAME} (the basename) while uploading to
        ${BUCKET_NAME}-${BUCKET_REGION} when --single-region is absent. So a child
        staged WITHOUT --single-region bakes a CodeBuild source bucket that holds
        none of its artifacts. This test documents the latent inconsistency the
        bundle routes around; if a child is ever fixed, this test fails and the
        bundle's workaround can be revisited.
        """
        for name, text in self.child_build.items():
            self.assertIn(
                "s|%%BUCKET_NAME%%|${BUCKET_NAME}|g", text,
                f"{name}: no longer seds the bare basename",
            )
            self.assertRegex(
                text, r'deploy_to_bucket "\$\{BUCKET_NAME\}-\$\{[A-Z_]*REGION\}"',
                f"{name}: no longer appends the region on upload",
            )

    def test_bundle_always_passes_single_region_to_children(self):
        """The workaround: resolve the regional bucket name once, then hand every
        child '--bucket <resolved> --single-region' so its sed and its upload
        agree. Dropping --single-region here would reintroduce the mismatch for
        all eight at once.
        """
        self.assertRegex(
            self.build,
            r'--bucket "\$STAGING_BUCKET"[\s\S]{0,200}--single-region',
        )

    def test_root_template_has_no_single_region_condition(self):
        """deployment/media2cloud.yaml appends '-${AWS::Region}' to its bucket via
        a bSingleRegion condition. This template must NOT: the build script writes
        the already-resolved bucket name, so a condition here would be a second
        independent way for the composed TemplateURL to name the wrong bucket.

        Asserted structurally rather than by text search — the deviation is
        documented in a comment that names bSingleRegion, so grepping the raw
        text would fail on its own explanation.

        This used to assert the template had NO Conditions block at all, which
        was a proxy that stopped being available once the optional proxy bucket
        needed one. Pin the exact condition SET instead: that still fails the
        moment a region-splitting condition is added, without banning conditions
        outright.
        """
        # bGatewayNameGiven (2026-09-23) gates the gateway-name derivation,
        # not a region split — it belongs in the pinned set.
        self.assertEqual({"bCreateProxyBucket", "bGatewayNameGiven"},
                         set(self.root.get("Conditions", {})))
        self.assertEqual(
            {"Template", "Stack"}, set(self.root["Mappings"]["Solution"]),
            "an added mapping may be reintroducing a per-region bucket split",
        )
        # The mapped bucket must be used verbatim, never concatenated with a
        # region suffix inside the !Sub.
        for name, node in self.nested().items():
            tag, value = node["Properties"]["TemplateURL"]
            self.assertEqual("!Sub", tag, name)
            template, params = value
            self.assertEqual(
                "https://${bucket}.s3.${AWS::Region}.amazonaws.com/${key}",
                template, f"{name}: unexpected TemplateURL shape",
            )
            self.assertEqual(
                ("!FindInMap", ["Solution", "Template", "S3Bucket"]),
                params["bucket"], f"{name}: bucket is not the mapped value",
            )


class TestDeployScript(BundleTestCase):
    def _capabilities(self):
        """The CAPABILITIES assignment only — not the surrounding comment, which
        necessarily names the capability it explains NOT passing.
        """
        m = re.search(r'^CAPABILITIES="([^"]*)"', self.deploy, re.M)
        self.assertIsNotNone(m, "no CAPABILITIES assignment found")
        return m.group(1).split()

    def test_passes_iam_capabilities(self):
        """The children create IAM roles, so CAPABILITY_IAM is mandatory.
        NAMED_IAM is defensive — see the next test for why it is not strictly
        required today, and why it is still passed.
        """
        caps = self._capabilities()
        self.assertIn("CAPABILITY_IAM", caps)
        self.assertIn("CAPABILITY_NAMED_IAM", caps)

    def test_does_not_pass_auto_expand(self):
        """Measured against real CloudFormation, and counterintuitive enough to
        pin: `validate-template` on this root ASKS for CAPABILITY_AUTO_EXPAND, but
        it is not required. That response is a conservative superset for any
        template containing a nested stack — CFN cannot see inside a child at
        validate time. A minimal template whose only resource is an
        AWS::CloudFormation::Stack (no IAM, no transform) reports the same, yet
        create-stack on it succeeds with CAPABILITY_IAM alone AND with no
        capabilities at all. Without this test, someone runs validate-template,
        sees the mismatch, and "fixes" it by granting a capability nobody audited.
        """
        self.assertNotIn("CAPABILITY_AUTO_EXPAND", self._capabilities())

    def test_no_child_declares_a_transform(self):
        """The precondition for the test above. AUTO_EXPAND is required only for
        transforms and macros; if a child ever adds one, that test's premise dies
        and the bundle must start passing it.
        """
        for name, d in CHILDREN.items():
            text = _read(os.path.join(DOCKER_ROOT, d, "cfnstack.yaml"))
            self.assertNotRegex(text, r"^Transform:", f"{name} declares Transform")
            self.assertNotIn("Fn::Transform", text, f"{name} uses Fn::Transform")
            self.assertNotIn("AWS::Serverless", text, f"{name} uses SAM")

    def test_named_iam_is_defensive_not_load_bearing(self):
        """Pins a fact I got wrong once by reading grep counts too quickly: NO
        child declares RoleName / ManagedPolicyName / InstanceProfileName, and
        none declares a standalone AWS::IAM::Policy. Every named-looking hit is an
        inline PolicyName inside a role's Policies list — verified against real
        CFN, that shape needs only CAPABILITY_IAM; NAMED_IAM is returned only for
        an explicit RoleName. So NAMED_IAM is passed for future-proofing and to
        match the children's own scripts, NOT because a child requires it.

        If this test ever fails, the capability became load-bearing — which is
        fine, but update the deploy script's comment so it stops claiming
        otherwise.
        """
        for name, d in CHILDREN.items():
            text = _read(os.path.join(DOCKER_ROOT, d, "cfnstack.yaml"))
            self.assertNotRegex(
                text, r"^\s+(RoleName|ManagedPolicyName|InstanceProfileName):",
                f"{name} now declares a named IAM resource",
            )
            for t in ("AWS::IAM::Policy", "AWS::IAM::ManagedPolicy",
                      "AWS::IAM::User", "AWS::IAM::Group"):
                self.assertNotIn(f"Type: {t}", text, f"{name} declares {t}")

    def test_help_states_the_gateway_uniqueness_rule(self):
        """The rule changed 2026-09-23: an OMITTED --gateway-name derives
        {prefix}-{stack-id}, so default-parameter bundles never collide —
        gateway stack included. Only a PINNED name carries a uniqueness duty,
        and it is per ACCOUNT (the Cognito domain namespace is global, so the
        old per-region phrasing understated it). --help is where a deployer
        looks, so it must state the new rule, not the retired one.
        """
        self.assertIn("{prefix}-{stack-id}", self.deploy)
        self.assertIn("unique per ACCOUNT", self.deploy)
        self.assertNotIn("You MUST vary --gateway-name", self.deploy)

    def test_derives_template_url_from_the_built_template(self):
        """Reading dist/ rather than cfnstack.yaml is the safety property: the
        source carries %%BUCKET_NAME%%, so an unbuilt template fails loudly on a
        bucket literally named '%%BUCKET_NAME%%' instead of resolving to a
        wrong-but-plausible one.
        """
        self.assertIn("dist/global-s3-assets/micro-workflow-bundle-stack.template",
                      self.deploy)
        self.assertIn('"${T_BUCKET#%%}" != "$T_BUCKET"', self.deploy)


class TestBuildScriptSafety(BundleTestCase):
    def test_verifies_every_composed_key_exists_before_writing_the_root(self):
        """Without this, a child renaming WORKFLOW_NAME/TEMPLATE_NAME, or a
        --single-region mismatch against an earlier staging run, surfaces as a
        failed TemplateURL fetch ~90 minutes into a create.
        """
        self.assertIn("head-object", self.build)
        self.assertIn("verify_staged_children", self.build)

    def test_verification_runs_even_when_children_are_skipped(self):
        """--skip-children is the case that MOST needs the check: it assumes the
        eight keys are already in S3 from an earlier run.
        """
        main = self.build[self.build.index("# Main"):]
        self.assertIn("verify_staged_children", main)
        skip_branch = main[main.index("SKIP_CHILDREN"):]
        self.assertIn("verify_staged_children", skip_branch)

    def test_aborts_the_run_on_any_child_failure(self):
        """A partially staged bundle whose root points at a template that was
        never uploaded fails at deploy time instead, after real CodeBuild minutes.
        """
        self.assertIn("aborting the bundle", self.build)

    def test_checks_each_child_path_explicitly(self):
        """The zip lesson generalizes: a command exiting 0 is not evidence that
        all its inputs existed. Check the directory, the build script, and the
        template before trusting the loop.
        """
        self.assertIn('[ ! -d "$cpath" ]', self.build)
        self.assertIn('[ ! -f "$cscript" ]', self.build)

    def test_reads_each_child_version_once_and_forwards_it(self):
        """If the bundle reads .version for the mapping and the child reads it
        again for its own KEY_PREFIX, the two reads can straddle an edit and the
        root template ends up pointing at a key that was never staged.
        """
        self.assertRegex(self.build, r'--version "\$cver"')

    def test_dirty_tree_requires_an_explicit_flag(self):
        """A dirty build's artifact is indistinguishable from a clean one by its
        S3 key alone. Building dirty must stay possible (it is the normal dev
        loop) but must be stated.
        """
        self.assertIn("--allow-dirty", self.build)
        self.assertIn("ALLOW_DIRTY", self.build)

    def test_a_failed_git_status_is_not_treated_as_clean(self):
        """`git status --porcelain` prints nothing when it FAILS as well as when
        the tree is clean, so empty output alone is not evidence of cleanliness.
        --allow-dirty is now the guard's only escape, so a build run outside a
        git checkout must not slip past it by reading as clean.
        """
        self.assertIn("GIT_STATUS_RC", self.build)
        self.assertIn("[ $GIT_STATUS_RC -ne 0 ]", self.build)


class TestBuildOnlyAndSourcePackage(BundleTestCase):
    """The build-time packaging contract (2026-10-08, revised same day):
    every build script runs its full local pipeline under --build-only with
    no AWS calls, and the REPO-ROOT build script packages the TRACKED TREE
    (git ls-files) into the single source zip. Per-folder source zips are
    retired — hand-maintained allowlists drifted (routers, test suites and
    deploy scripts measured missing), and the tracked tree is the curated
    artifact by construction.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.root_build = _read(os.path.join(DOCKER_ROOT,
                                            "build-deploy-s3-dist.sh"))

    def test_every_build_script_supports_build_only(self):
        """The root packaging builds everything through the bundle, so a
        child without the flag would drag an AWS dependency back into the
        offline path."""
        for name, text in {**self.child_build,
                           "bundle": self.build}.items():
            self.assertIn("--build-only)", text,
                          f"{name}: no --build-only case in the arg parser")

    def test_no_build_script_still_builds_a_source_code_zip(self):
        """Retired 2026-10-08 (user decision): the root git-based package is
        the ONE source distribution. A resurrected per-folder source zip
        would be an incomplete second answer to the same question."""
        for name, text in {**self.child_build,
                           "bundle": self.build}.items():
            self.assertNotIn("_source_code.zip", text,
                             f"{name}: still builds a per-folder source zip")

    def test_the_root_script_builds_through_the_bundle_then_packages(self):
        """The repo-root script owns the PACKAGING; the bundle folder owns
        the BUILD. Delegation first (same option surface), packaging after —
        and the root must stay template-less, or the child-discovery test
        would see the repo root itself as an unregistered child."""
        self.assertIn(
            '(cd "$ROOT_DIR/micro-workflow-bundle" && '
            'bash ./build-deploy-s3-dist.sh "$@")',
            self.root_build)
        self.assertLess(self.root_build.index("micro-workflow-bundle"),
                        self.root_build.index("git ls-files"))
        self.assertFalse(
            os.path.isfile(os.path.join(DOCKER_ROOT, "cfnstack.yaml")),
            "a repo-root cfnstack.yaml would make the root script look "
            "like a bundle child to the discovery test")

    def test_the_package_members_are_the_tracked_tree(self):
        """git ls-files is the member list — complete and drift-proof, and
        untracked/ignored files can never join. The count check is the
        guard against a silent shortfall (zip does not fail reliably when
        some listed files are missing from the working tree)."""
        self.assertIn("git ls-files -z | xargs -0 zip", self.root_build)
        self.assertIn("TRACKED_COUNT", self.root_build)
        self.assertIn("ZIPPED_COUNT", self.root_build)
        self.assertIn("!= tracked file count", self.root_build)

    def test_the_source_package_is_leak_swept(self):
        """A belt over .gitignore: the package listing is checked for the
        LOCAL-ONLY artifact classes that must never ship. A hit means
        something slipped past .gitignore into the index. Asserts each
        pattern inside the sweep's grep, not merely somewhere in the file."""
        start = self.root_build.index("leaks=$(")
        sweep = self.root_build[start:self.root_build.index("|| true", start)]
        for pattern in ("GOTCHAS", "do-not-commit", "mcp\\.json",
                        "gateway_headers", "__pycache__", "pyc"):
            self.assertIn(pattern, sweep,
                          f"source-package leak sweep lost: {pattern}")

    def test_the_leak_sweep_patterns_are_gitignored(self):
        """The sweep is the belt; .gitignore is the suspenders. Every file
        class the sweep hunts must also be ignored, or the normal add/commit
        path can track one and the sweep becomes the only defense."""
        gitignore = _read(os.path.join(DOCKER_ROOT, ".gitignore"))
        for pattern in ("GOTCHAS.md", "do-not-commit/", ".mcp.json",
                        "gateway_headers.", "__pycache__/", "*.pyc"):
            self.assertIn(pattern, gitignore,
                          f".gitignore lost the swept class: {pattern}")

    def test_the_staging_values_have_one_writer(self):
        """The bundle resolves the staging bucket ONCE and records it in
        dist/.staging.env; the root script sources that record. The root
        re-deriving the bucket would be a second independent way to name the
        wrong one — the exact failure the bundle's resolution exists to
        prevent."""
        self.assertIn("dist/.staging.env", self.build)
        self.assertIn('STAGING_BUCKET="$STAGING_BUCKET"', self.build)
        self.assertIn("micro-workflow-bundle/dist/.staging.env",
                      self.root_build)
        for aws_call in ("sts get-caller-identity", "get-bucket-location"):
            self.assertNotIn(aws_call, self.root_build,
                             "the root script must not resolve the bucket "
                             "itself — source the bundle's record")

    def test_build_only_builds_children_without_bucket_flags(self):
        """Under the flag, children get --build-only and must NOT get
        --bucket/--single-region — a placeholder bucket passed as a real one
        would substitute into their templates as a plausible name."""
        fn = self.build[self.build.index("function build_children()"):]
        fn = fn[:fn.index("\n}")]
        start = fn.index('if [ "$BUILD_ONLY" == "true" ]')
        bo_branch = fn[start:fn.index("else", start)]
        self.assertIn("--build-only", bo_branch)
        self.assertNotIn("--bucket", bo_branch)
        self.assertNotIn("--single-region", bo_branch)

    def test_build_only_skips_every_aws_touchpoint(self):
        """verify_staged_children (head-object) and deploy_root_template
        (s3 cp) are AWS-only; the bundle's build-only branch must reach
        neither while still producing the template — and the root script's
        upload must sit behind the same flag."""
        main = self.build[self.build.index("# Main"):]
        start = main.index('if [ "$BUILD_ONLY" == "true" ]')
        bo = main[start:main.index("else", start)]
        self.assertIn("build_cloudformation_template", bo)
        for fn in ("verify_staged_children", "deploy_root_template",
                   "on_complete"):
            self.assertNotIn(fn, bo)
        upload = self.root_build.index("aws s3 cp")
        guard = self.root_build.index('if [ "$BUILD_ONLY" == "true" ]')
        self.assertLess(guard, upload,
                        "the root script's upload is not behind --build-only")


class TestChildLogGroupHygiene(BundleTestCase):
    """Cross-child log-group invariants (README.md §P10 item 10).

    A Lambda with no declared group still gets one: Lambda auto-creates
    /aws/lambda/{name} with retentionInDays=None, so the logs are billed
    forever AND the group survives stack deletion. For a NAMED function that
    orphan then blocks the next create-stack with "already exists"; 16 orphans
    from CFN-generated StartBuildLambda names were measured in eu-west-1 before
    this was fixed.

    These live in the bundle suite rather than per-child because four of the
    five affected children have no unittest suite at all — their test/ holds
    live-AWS scripts. The bundle already parses every child template, so it is
    the only place the invariant can be enforced for all of them at once.
    """

    # Every child that runs a CodeBuild image build has an UNNAMED
    # StartBuildLambda. Unnamed means /aws/lambda/{FunctionName} is unknowable
    # at author time, so the group is reached via LoggingConfig instead.
    #
    # TtsStack joined this set when its naming was conformed: it had a NAMED
    # StartBuildFunction with a canonical StartBuildLogGroup, which worked but
    # was a seventh spelling of one pattern (the stack was ported from a
    # separate upstream repo). All seven are now identical.
    _START_BUILD_CHILDREN = {
        "MediaProcessingStack", "AudiostemSeparationStack",
        "SpeechEnhancementStack", "SpeechDiarizationStack",
        "ShotSegmentationStack", "FaceRecognitionStack",
        "HybridSearchStack", "TtsStack", "BedrockInferenceStack",
    }

    @staticmethod
    def _sub_body(node):
        """The format string out of a !Sub, in either form: a bare scalar, or
        [body, {substitution map}]."""
        if not isinstance(node, tuple):
            return node
        _, val = node
        return val if isinstance(val, str) else val[0]

    def _resources(self, child):
        return self.child[child].get("Resources", {})

    def test_start_build_lambdas_are_unnamed_and_use_logging_config(self):
        """An unnamed function CANNOT declare /aws/lambda/{FunctionName}, so
        LoggingConfig is the only way it reaches a stack-owned group. If one
        ever gains a FunctionName, this test should fail loudly rather than
        letting the LoggingConfig quietly become redundant."""
        for child in sorted(self._START_BUILD_CHILDREN):
            fn = self._resources(child).get("StartBuildLambda")
            self.assertIsNotNone(fn, f"{child} has no StartBuildLambda")
            props = fn["Properties"]
            self.assertNotIn(
                "FunctionName", props,
                f"{child}'s StartBuildLambda gained a FunctionName — either "
                "declare /aws/lambda/{name} directly or keep it unnamed, but "
                "do not leave both mechanisms in play")
            self.assertEqual(
                props.get("LoggingConfig", {}).get("LogGroup"),
                ("!Ref", "StartBuildLambdaLogGroup"),
                f"{child}'s StartBuildLambda must route to its declared group "
                "via LoggingConfig, or Lambda auto-creates an unmanaged one")

    def test_start_build_log_groups_are_declared_with_retention(self):
        """A declared group with no RetentionInDays is never-expire too — the
        declaration alone fixes the orphan half, not the cost half."""
        for child in sorted(self._START_BUILD_CHILDREN):
            group = self._resources(child).get("StartBuildLambdaLogGroup")
            self.assertIsNotNone(group, f"{child} has no StartBuildLambdaLogGroup")
            self.assertEqual(group["Type"], "AWS::Logs::LogGroup")
            self.assertIn("RetentionInDays", group["Properties"], child)

    def test_start_build_lambdas_depend_on_their_log_group(self):
        """StartBuildLambda is invoked by a custom resource during stack
        create. Without the ordering edge, Lambda's first invocation can beat
        CFN to the group and create the unmanaged one anyway."""
        for child in sorted(self._START_BUILD_CHILDREN):
            fn = self._resources(child)["StartBuildLambda"]
            depends = fn.get("DependsOn")
            depends = [depends] if isinstance(depends, str) else (depends or [])
            self.assertIn("StartBuildLambdaLogGroup", depends, child)

    def test_no_child_role_may_create_its_own_log_group(self):
        """logs:CreateLogGroup alongside a DECLARED group is actively harmful:
        a function that cannot reach its declared group silently creates an
        unmanaged one instead of failing visibly. Checked only on the roles
        this remediation touched — the tier-function roles predate the pattern
        and still carry the permission, which is tracked separately."""
        REMEDIATED = ("StartBuildLambdaRole", "ProxyBucketValidatorRole")
        for child in sorted(CHILDREN):
            for logical, body in self._resources(child).items():
                if logical not in REMEDIATED:
                    continue
                for policy in body["Properties"].get("Policies", []):
                    for stmt in policy["PolicyDocument"]["Statement"]:
                        action = stmt["Action"]
                        actions = action if isinstance(action, list) else [action]
                        self.assertNotIn(
                            "logs:CreateLogGroup", actions,
                            f"{child}.{logical} may not create log groups — "
                            "scope logs to the declared group's ARN instead")

    def test_no_declared_group_lambda_role_carries_the_basic_execution_policy(self):
        """AWSLambdaBasicExecutionRole grants logs:CreateLogGroup on EVERY group,
        which is the same harm as the inline grant above — but the test above
        cannot see it, because a managed policy ARN carries no Statement list.
        That blind spot is not hypothetical: media-toolkit-gateway declared two log
        groups while both its roles still carried the managed policy, so both
        functions could create unmanaged groups instead of failing visibly, and
        that is exactly how the two orphans blocking its create came to exist.

        Scoped the same way test_no_child_role_may_create_its_own_log_group is:
        to roles whose function routes to its group through LoggingConfig, plus
        the two remediated role names. tts's TtsLambdaRole and GatewayToolRole
        still carry the managed policy alongside declared canonical-name groups
        and are DELIBERATELY not covered here — that is the same pre-existing
        tier-role debt the sibling test excludes, and widening the boundary in
        passing would fail a correct change and train people to edit the test.
        """
        BASIC = ("arn:aws:iam::aws:policy/service-role/"
                 "AWSLambdaBasicExecutionRole")
        REMEDIATED = ("StartBuildLambdaRole", "ProxyBucketValidatorRole")
        for child in sorted(CHILDREN):
            resources = self._resources(child)
            # Roles whose function explicitly decouples its group name via
            # LoggingConfig — the pattern this test defends.
            guarded = set(REMEDIATED)
            for body in resources.values():
                if body["Type"] != "AWS::Lambda::Function":
                    continue
                props = body["Properties"]
                if "LoggingConfig" not in props:
                    continue
                role = props.get("Role")
                if not (isinstance(role, tuple) and role[0] == "!GetAtt"):
                    continue
                target = role[1]
                if isinstance(target, str):
                    guarded.add(target.split(".")[0])
                elif isinstance(target, list) and target:
                    guarded.add(target[0])
            for logical, body in resources.items():
                if body["Type"] != "AWS::IAM::Role" or logical not in guarded:
                    continue
                managed = body["Properties"].get("ManagedPolicyArns", [])
                self.assertNotIn(
                    BASIC, managed,
                    f"{child}.{logical} carries AWSLambdaBasicExecutionRole "
                    "while the stack declares log groups — that managed policy "
                    "grants logs:CreateLogGroup on every group, so the function "
                    "can silently create an unmanaged group instead of failing "
                    "visibly. Scope logs:CreateLogStream/PutLogEvents to the "
                    "declared group's ARN in an inline policy instead.")

    def test_validator_functions_are_named_bucket_check(self):
        """The old -proxy-bucket-validator name put audiostem-separation's
        function at exactly 64 chars, the hard FunctionName limit, so one more
        character of ResourcePrefix would have failed the deploy. The rename is
        also what makes declaring the canonical /aws/lambda/{FunctionName}
        group safe: the new name has never existed, so nothing pre-exists to
        collide with at pre-flight validation."""
        for child, directory in sorted(CHILDREN.items()):
            fn = self._resources(child).get("ProxyBucketValidatorFunction")
            if fn is None:                      # hybrid-search, media-toolkit-gateway
                continue
            name = self._sub_body(fn["Properties"]["FunctionName"])
            self.assertTrue(
                name.endswith("-bucket-check"),
                f"{child} validator is {name!r} — must end in -bucket-check")
            group = self._resources(child)["ProxyBucketValidatorLogGroup"]
            self.assertEqual(
                self._sub_body(group["Properties"]["LogGroupName"]),
                "/aws/lambda/" + name,
                f"{child}'s validator group must track the function name")

    def test_child_function_names_fit_64_chars_at_the_prefix_ceiling(self):
        """FunctionName's hard limit is 64 (measured: 65 is rejected with an
        explicit length constraint; at exactly 64 the name validates and the
        call fails only on the role). ResourcePrefix's ceiling is 27, set by
        hybrid-search's S3 Vectors bucket name — so every name must fit at 27,
        not merely at the 20-char prefix deployed today. This is the assertion
        that makes (d) safe to raise the prefix."""
        CEILING = 27
        for child in sorted(CHILDREN):
            for logical, body in self._resources(child).items():
                if body.get("Type") != "AWS::Lambda::Function":
                    continue
                raw = body["Properties"].get("FunctionName")
                if raw is None:
                    continue                    # unnamed: CFN generates it
                name = self._sub_body(raw)
                if "${ResourcePrefix}" not in name:
                    continue                    # named off something else
                # Substitution-map vars (${name}, ${variant}) resolve from
                # Mappings at deploy time; keep the literal, which is at least
                # as long as any real value here.
                rendered = name.replace("${ResourcePrefix}", "P" * CEILING)
                self.assertLessEqual(
                    len(rendered), 64,
                    f"{child}.{logical} renders to {len(rendered)} chars at "
                    f"ResourcePrefix={CEILING} ({rendered!r})")


class TestProxyBucketOwnership(BundleTestCase):
    """README.md §P10 item 11 — the bundle declares the proxy bucket when the
    parameter is blank, instead of forcing the parameter to guard against the
    child validators' create-on-404 path.

    The invariant that actually matters is the one a single missed edit breaks:
    all EIGHT children must receive the !If, because a bare !Ref ProxyBucket
    would pass "" to that child and send one workflow's outputs to a bucket
    named empty-string.
    """

    _IF = ("!If", ["bCreateProxyBucket", ("!Ref", "BundleProxyBucket"),
                   ("!Ref", "ProxyBucket")])

    def test_every_child_gets_the_conditional_bucket_not_a_bare_ref(self):
        """The single-missed-call-site bug. Seven children with the !If and one
        with a bare !Ref ProxyBucket deploys clean when a bucket IS supplied and
        breaks only in the blank case — i.e. it survives exactly the testing a
        deployer is most likely to do."""
        for name, node in self.nested().items():
            self.assertEqual(
                self._IF, node["Properties"]["Parameters"]["ProxyBucket"],
                f"{name} must receive the conditional bucket, not a bare !Ref")

    def test_the_condition_is_blankness_not_mere_presence(self):
        """!Equals against "" is what makes 'omitted' mean 'create one'. A
        condition on anything else (e.g. a sentinel like 'AUTO') would leave the
        empty string as a legal value that reaches the children."""
        self.assertEqual(
            ("!Equals", [("!Ref", "ProxyBucket"), ""]),
            self.root["Conditions"]["bCreateProxyBucket"])

    def test_the_bucket_retains_on_both_delete_and_replace(self):
        """UpdateReplacePolicy is the one that actually fires on a Prefix change:
        BucketName is replacement-forcing, so without it an edit to Prefix would
        DELETE every workflow output. DeletionPolicy covers stack delete.
        Measured on a live probe stack: flipping the condition off reported
        DELETE_SKIPPED, and a full stack delete left the bucket in place."""
        bucket = self.root["Resources"]["BundleProxyBucket"]
        self.assertEqual("Retain", bucket.get("DeletionPolicy"))
        self.assertEqual("Retain", bucket.get("UpdateReplacePolicy"))
        self.assertEqual("bCreateProxyBucket", bucket.get("Condition"))

    def test_the_bucket_and_its_policy_share_the_condition(self):
        """An unconditional resource referencing a conditional one fails the
        whole template at validate time, not just for deployers who omit the
        parameter."""
        for logical in ("BundleProxyBucket", "BundleProxyBucketPolicy"):
            self.assertEqual(
                "bCreateProxyBucket",
                self.root["Resources"][logical].get("Condition"), logical)

    def test_the_bucket_name_follows_the_m2c_convention_and_fits_63_chars(self):
        """Same shape as m2c's own proxy bucket and hybrid-search's -vectors
        bucket. The length check is the reason item 9's arithmetic was got
        backwards once: -{account 12}-{region 14}-proxy spends 34, so this
        constraint allows ResourcePrefix ≤ 29 — LOOSER than the 27 that
        -vectors imposes, therefore NOT binding. Asserting it anyway means a
        future rename that makes it binding fails here instead of silently
        moving the ceiling."""
        name = self.root["Resources"]["BundleProxyBucket"]["Properties"]["BucketName"]
        tag, (body, subs) = name
        self.assertEqual("!Sub", tag)
        self.assertEqual(
            "${prefix}-${stackId}-${AWS::AccountId}-${AWS::Region}-proxy", body)
        self.assertEqual(("!Ref", "Prefix"), subs["prefix"])
        # Worst case: 12-char account, 14-char region (ap-southeast-N), and a
        # ResourcePrefix at the 27 that hybrid-search's -vectors bucket allows.
        rendered = (body
                    .replace("${prefix}-${stackId}", "P" * 27)
                    .replace("${AWS::AccountId}", "9" * 12)
                    .replace("${AWS::Region}", "a" * 14))
        self.assertLessEqual(len(rendered), 63, rendered)

    def test_the_bucket_carries_the_security_floor(self):
        """Public-access-block + AES256 + DenyInsecureTransport, matching what
        the child validators' _create() already applies — so a bundle-owned
        bucket is never weaker than a validator-created one. m2c's extras
        (versioning, CORS, acceleration, access logging, _ephemeraldata expiry)
        are deliberately skipped; see the template comment for each reason."""
        props = self.root["Resources"]["BundleProxyBucket"]["Properties"]
        self.assertEqual(
            {"BlockPublicAcls": True, "BlockPublicPolicy": True,
             "IgnorePublicAcls": True, "RestrictPublicBuckets": True},
            props["PublicAccessBlockConfiguration"])
        self.assertEqual(
            "AES256",
            props["BucketEncryption"]["ServerSideEncryptionConfiguration"][0]
                 ["ServerSideEncryptionByDefault"]["SSEAlgorithm"])
        stmts = (self.root["Resources"]["BundleProxyBucketPolicy"]
                 ["Properties"]["PolicyDocument"]["Statement"])
        deny = [s for s in stmts if s.get("Sid") == "DenyInsecureTransport"]
        self.assertEqual(1, len(deny), "no DenyInsecureTransport statement")
        self.assertEqual("Deny", deny[0]["Effect"])
        self.assertIs(False, deny[0]["Condition"]["Bool"]["aws:SecureTransport"])

    def test_the_bucket_name_is_surfaced_as_an_output(self):
        """The bucket is RETAINED on delete and its name embeds a stack-id
        slice, so this output is the only record of what to sweep — or to pass
        back as ProxyBucket to recover the outputs. Losing it means finding the
        bucket by guesswork."""
        outputs = self.root["Outputs"]
        # Assert presence BEFORE subscripting: a missing key raises KeyError,
        # which unittest reports as errors=1, not failures=1 — and an error
        # reads like a broken test rather than a caught regression.
        for key in ("ProxyBucketName", "ProxyBucketIsStackOwned",
                    "ResourcePrefix"):
            self.assertIn(key, outputs,
                          f"{key} output is the only record of a RETAINED "
                          "resource's name once the stack is gone")
        self.assertEqual(self._IF, outputs["ProxyBucketName"]["Value"])

    def test_the_validators_are_still_wired_and_still_needed(self):
        """The bundle does NOT make the child validators redundant: a child
        deployed standalone still needs one, and inside the bundle the
        ownership/region check still fires. What changes is that the CREATE path
        becomes unreachable, because CFN orders the bucket first via the !If's
        reference (measured on a probe: bucket CREATE_COMPLETE at 09:13:06.764,
        child CREATE_IN_PROGRESS at 09:13:07.092). If a validator ever
        disappears, the standalone deploy path lost its guard."""
        validators = {
            child for child in CHILDREN
            if "ProxyBucketValidatorFunction" in self.child[child].get("Resources", {})
        }
        self.assertEqual(
            {"MediaProcessingStack", "AudiostemSeparationStack",
             "SpeechEnhancementStack", "SpeechDiarizationStack",
             "ShotSegmentationStack", "FaceRecognitionStack",
             "TtsStack"},
            validators)


class TestPrefixBudget(BundleTestCase):
    """README.md §P10 item 10(d) — Prefix is the user-settable part, and
    ResourcePrefix is {Prefix}-{12-char stack-id slice}.

    The stack-id slice is what makes the retained -vectors bucket safe: the
    native S3Vectors type CANNOT adopt an existing bucket (measured: 409
    AlreadyExists rolls back the whole stack), so a stable operator-supplied
    prefix would make a re-create impossible after one delete.
    """

    def test_prefix_is_capped_at_14(self):
        """Arithmetic, not taste: 14 + 1 hyphen + 12 stack-id chars = 27, which
        is exactly what hybrid-search's -{account}-{region}-vectors name leaves
        inside S3's 63-char limit. Verified live — CFN rejects 15 with
        "must contain at most 14 characters" and accepts 14, rendering
        'aaaaaaaaaaaaaa-0234d9037121' (27 chars)."""
        prefix = self.root["Parameters"]["Prefix"]
        self.assertEqual(14, prefix["MaxLength"])
        self.assertEqual(1, prefix["MinLength"])

    def test_prefix_is_constrained_to_bucket_safe_characters(self):
        """It lands in S3 bucket names, where uppercase and underscores are
        illegal. Without the pattern the failure surfaces as an S3 error on a
        nested stack rather than as parameter validation before anything is
        built. All three rejections verified live (uppercase, underscore,
        leading hyphen)."""
        self.assertEqual(
            "^[a-z0-9][a-z0-9-]*$",
            self.root["Parameters"]["Prefix"]["AllowedPattern"])

    def test_no_child_receives_the_bare_prefix(self):
        """Every workflow child must get {Prefix}-{stack-id}, never !Ref Prefix.
        A child handed the bare prefix would name its resources without the
        stack-id slice, so two bundles would collide and — worse — the retained
        -vectors bucket could never be re-created after a delete."""
        expected = ("!Sub", [
            "${prefix}-${stackId}",
            {"prefix": ("!Ref", "Prefix"),
             "stackId": ("!Select", [4, ("!Split", ["-", ("!Select", [
                 2, ("!Split", ["/", ("!Ref", "AWS::StackId")])])])])},
        ])
        for name, node in self.nested().items():
            params = node["Properties"]["Parameters"]
            if "ResourcePrefix" not in params:
                # media-toolkit-gateway has no ResourcePrefix — it names everything
                # off GatewayName/Environment; since 2026-09-23 an empty
                # GatewayName derives this same form (the test below).
                self.assertEqual("M2cAgentToolsStack", name)
                continue
            self.assertEqual(expected, params["ResourcePrefix"], name)

    def test_empty_gateway_name_derives_the_prefixed_form(self):
        """An empty GatewayName (the default, 2026-09-23) must hand
        media-toolkit-gateway the SAME {Prefix}-{stack-id} derivation the children
        get as ResourcePrefix — that is what makes two default-parameter
        bundles non-colliding, gateway stack included. A set name passes
        through untouched (pinned identity: the secret path follows it, so
        a re-created stack reconnects every configured client)."""
        derived = ("!Sub", [
            "${prefix}-${stackId}",
            {"prefix": ("!Ref", "Prefix"),
             "stackId": ("!Select", [4, ("!Split", ["-", ("!Select", [
                 2, ("!Split", ["/", ("!Ref", "AWS::StackId")])])])])},
        ])
        wired = (self.nested()["M2cAgentToolsStack"]["Properties"]
                 ["Parameters"]["GatewayName"])
        self.assertEqual(
            ("!If", ["bGatewayNameGiven", ("!Ref", "GatewayName"), derived]),
            wired)

    def test_the_deploy_script_renamed_the_flag_and_rejects_the_old_one(self):
        """--resource-prefix is gone. The argument parser ends in a catch-all
        '*) shift ;;', so without an explicit rejection the old flag would be
        SILENTLY IGNORED and the stack would deploy under the default prefix —
        a second bundle meant to be isolated would collide, ~90 minutes in."""
        self.assertIn("--prefix)", self.deploy)
        self.assertRegex(
            self.deploy,
            r"--resource-prefix\)\s*\n\s*echo \"error:",
            "the renamed flag must be rejected explicitly, not fall through")
        self.assertIn("ParameterKey=Prefix,ParameterValue=", self.deploy)

    def test_the_deploy_script_always_sends_proxybucket(self):
        """It must be in the base PARAMETERS array, not appended conditionally
        like the other optionals. On an UPDATE, an omitted parameter keeps the
        value already on the stack — so dropping --proxy-bucket from a stack
        created with an explicit bucket would silently keep that bucket instead
        of switching to a stack-owned one."""
        m = re.search(r"^PARAMETERS=\((.*?)^\)", self.deploy, re.M | re.S)
        self.assertIsNotNone(m, "no PARAMETERS array found")
        self.assertIn("ParameterKey=ProxyBucket,ParameterValue=", m.group(1))
        self.assertNotRegex(
            self.deploy,
            r'\[ -n "\$PROXY_BUCKET" \].*PARAMETERS\+=',
            "ProxyBucket must not be appended conditionally")

    def test_the_deploy_script_no_longer_requires_proxy_bucket(self):
        """Omitting it is the RECOMMENDED path now. A leftover required-check
        would make the whole feature unreachable from the script."""
        self.assertNotIn('error: --proxy-bucket is required', self.deploy)
        self.assertIn('[ -z "$INGEST_BUCKET" ]', self.deploy)


class TestBucketValidatorParity(BundleTestCase):
    """gateway_tool/bucket_validator.py is COPIED into six children, not
    imported. The gateway zip is flat (`zip -j`, no lib/), so there is nowhere
    shared to import from — the copies ARE the distribution mechanism, and
    nothing but a test can keep them in sync.

    This is not a style rule. The five sibling copies read
    ResourceProperties["AccountId"] OUTSIDE their try block, and CFN's Delete
    event does not carry it: the KeyError escaped, _send() never ran, and the
    stack sat in DELETE_IN_PROGRESS for a full hour before giving up — twice,
    each time orphaning the validator Lambda whose surviving FunctionName then
    blocked the next create-stack. tts-on-aws was fixed and the siblings were
    not, so the bug sat in five templates behind a file that LOOKED shared.
    """

    _VALIDATOR_CHILDREN = {
        "MediaProcessingStack", "AudiostemSeparationStack",
        "SpeechEnhancementStack", "SpeechDiarizationStack",
        "ShotSegmentationStack", "FaceRecognitionStack", "TtsStack",
    }

    def _validator_src(self, child):
        path = os.path.join(DOCKER_ROOT, CHILDREN[child],
                            "gateway_tool", "bucket_validator.py")
        self.assertTrue(os.path.exists(path), f"{child} has no {path}")
        return _read(path)

    def test_the_children_that_declare_a_validator_are_exactly_these(self):
        """Guards the set above: a child that gains a validator function
        without gaining the file (or vice versa) makes every other test in
        this class silently skip it."""
        declaring = {
            child for child in CHILDREN
            if "ProxyBucketValidatorFunction" in self.child[child].get(
                "Resources", {})
        }
        self.assertEqual(
            declaring, self._VALIDATOR_CHILDREN,
            "a child's validator appeared or disappeared — update "
            "_VALIDATOR_CHILDREN and confirm its bucket_validator.py matches")

    def test_every_copy_is_byte_identical(self):
        """Copies that drift are worse than no sharing at all: the file name
        implies they are the same, so a fix applied to one reads as a fix
        applied to all."""
        srcs = {child: self._validator_src(child)
                for child in sorted(self._VALIDATOR_CHILDREN)}
        canonical = srcs["MediaProcessingStack"]
        for child, src in srcs.items():
            self.assertEqual(
                src, canonical,
                f"{child}'s bucket_validator.py has drifted from "
                "media-processing's — copy one over the others")

    def test_no_property_read_precedes_the_try_block(self):
        """THE bug, pinned structurally rather than behaviourally: a
        behavioural test only catches it if the test event happens to omit the
        key that gets hoisted. Any subscript before `try:` can raise, and an
        exception escaping the handler means CFN never gets a response."""
        for child in sorted(self._VALIDATOR_CHILDREN):
            src = self._validator_src(child)
            body = src.split("def handler(", 1)[1]
            head = body.split("try:", 1)[0]
            self.assertNotIn(
                '["', head,
                f"{child}: subscript before try: — an escaping KeyError here "
                "is an hour of DELETE_IN_PROGRESS, not a fast failure")

    def test_the_handler_is_named_handler_everywhere(self):
        """tts wired Handler: bucket_validator.lambda_handler while the five
        siblings wired .handler — the same file could not have served both.
        Template and file must agree, per child."""
        for child in sorted(self._VALIDATOR_CHILDREN):
            self.assertIn("def handler(event, context):",
                          self._validator_src(child), child)
            fn = self.child[child]["Resources"]["ProxyBucketValidatorFunction"]
            self.assertEqual(fn["Properties"]["Handler"],
                             "bucket_validator.handler", child)

    def test_the_failure_path_can_always_resolve_a_physical_id(self):
        """A FAILED response still has to carry a PhysicalResourceId. Deriving
        it from ResourceProperties means the failure path can fail for the
        same reason the success path did."""
        for child in sorted(self._VALIDATOR_CHILDREN):
            self.assertIn('event.get("PhysicalResourceId")',
                          self._validator_src(child),
                          f"{child}: no PhysicalResourceId fallback")


class TestCustomResourceTypeUniformity(BundleTestCase):
    """A custom resource's Custom:: TYPE is part of its identity — changing
    one is a REPLACEMENT that fires Delete on the old resource. So these
    spellings have deploy-time consequences and are worth pinning.

    tts-on-aws carried Custom::ProxyBucketValidation and
    Custom::BuildAndSeedModels because it was ported from a separate upstream
    repo; the bundle's own docs had to tell readers to grep for both spellings.
    """

    def _typed(self, child, prefix):
        return {logical: body["Type"]
                for logical, body in self.child[child].get(
                    "Resources", {}).items()
                if str(body.get("Type", "")).startswith(prefix)}

    def test_every_validator_custom_resource_uses_one_type(self):
        for child in sorted(TestBucketValidatorParity._VALIDATOR_CHILDREN):
            types = self._typed(child, "Custom::ProxyBucket")
            self.assertEqual(
                sorted(types.values()), ["Custom::ProxyBucketValidator"],
                f"{child} must use exactly one Custom::ProxyBucketValidator "
                f"resource; found {types}")

    def test_every_start_build_custom_resource_uses_one_type(self):
        """The buildspec's post_build phase sends the final CFN response, not
        the Lambda — which is what DelayResponse records. tts spelled it
        BuildAndSeedModels."""
        for child in sorted(
                TestChildLogGroupHygiene._START_BUILD_CHILDREN):
            types = self._typed(child, "Custom::StartBuild")
            self.assertEqual(
                sorted(types.values()), ["Custom::StartBuildDelayResponse"],
                f"{child} must use exactly one "
                f"Custom::StartBuildDelayResponse resource; found {types}")


# ─────────────────────────────────────────────────────────────────────────────
# Memory-tier vocabulary
#
# Six children size their workflow Lambdas from a Mappings.Lambda block of
# named tiers. The NAMES are the interface: re-pointing a state at another tier
# during testing only tells you something comparable if `3-cores` means the
# same 5376 MB in every workflow. Nothing enforced that — it held by hand.
#
# The count per child is deliberately NOT pinned. Children declare different
# subsets (7, 7, 7, 5, 2, and tts 4 → 7 on 2026-08-05) and declaring a tier no
# state routes to is the norm, not a defect: media-processing routes to 3 of
# its 7, audiostem-separation 4 of 7, speech-diarization 5 of 7. What must not
# drift is the memory/name pair behind a tier name.
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryTierVocabulary(BundleTestCase):
    _PAIRS = {
        "HalfCore": (896, "half-core"),
        "OneCore": (1792, "1-core"),
        "TwoCores": (3584, "2-cores"),
        "ThreeCores": (5376, "3-cores"),
        "FourCores": (7168, "4-cores"),
        "FiveCores": (8960, "5-cores"),
        "SixCores": (10240, "6-cores"),
    }

    # hybrid-search sizes its Lambdas per-role (Indexer/Search/Rebuild/...),
    # not per-core, and media-toolkit-gateway has no Mappings.Lambda at all; the three
    # gateway-only children (prompt-registry, conversation-analysis,
    # shot-metadata-enrichment) declare Lambda sizes but no core tiers. Listing
    # the children that DO use core tiers rather than skipping on absence, so a
    # child that silently loses its Lambda mapping fails instead of passing.
    _TIERED_CHILDREN = {
        "MediaProcessingStack", "AudiostemSeparationStack",
        "SpeechEnhancementStack", "SpeechDiarizationStack",
        "ShotSegmentationStack", "FaceRecognitionStack", "TtsStack",
        "BedrockInferenceStack",
    }

    def _tiers(self, child):
        lambda_map = self.child[child].get("Mappings", {}).get("Lambda", {})
        return {k: v for k, v in lambda_map.items()
                if k in self._PAIRS}

    def test_the_tiered_children_are_exactly_these(self):
        """hybrid-search sizes per-role, media-toolkit-gateway has no tiers and the
        gateway-only children declare none, so the set is the list above. A new
        child using core tiers must join it."""
        using = {name for name in CHILDREN if self._tiers(name)}
        self.assertEqual(using, self._TIERED_CHILDREN)

    def test_every_declared_tier_uses_the_house_memory_and_name(self):
        """A tier named 3-cores at anything but 5376 MB makes a cross-workflow
        comparison silently wrong — the swap looks like a like-for-like
        re-point and is not."""
        for child in sorted(self._TIERED_CHILDREN):
            for tier, body in sorted(self._tiers(child).items()):
                want_mem, want_name = self._PAIRS[tier]
                self.assertEqual(
                    (body.get("Memory"), body.get("Name")),
                    (want_mem, want_name),
                    f"{child}'s {tier} must be {want_mem} MB named "
                    f"{want_name!r}; found {body.get('Memory')} MB named "
                    f"{body.get('Name')!r}")

    def test_no_child_invents_a_tier_name_outside_the_vocabulary(self):
        """Catches the other direction: a `SevenCores` or `TwoAndAHalfCores`
        added to one child only, which the pair check above cannot see."""
        allowed = set(self._PAIRS) | {"Function", "GatewayTool"}
        for child in sorted(self._TIERED_CHILDREN):
            declared = set(self.child[child]["Mappings"]["Lambda"])
            self.assertLessEqual(
                declared, allowed,
                f"{child} declares Lambda mapping keys outside the house "
                f"vocabulary: {sorted(declared - allowed)}")


class TestWorkflowIdentityParity(BundleTestCase):
    """gateway_tool/workflow_identity.py is COPIED into every child that derives
    a workflow_id, not imported. The gateway zips are flat (`zip -j`, no lib/),
    so there is nowhere shared to import from — the copies ARE the distribution
    mechanism, exactly like bucket_validator.py.

    This exists because the derivation it replaces was hand-copied into nine
    gateways and had already drifted three ways. Drift here is worse than
    elsewhere: the workflow_id IS the execution name and the output prefix, so a
    preimage change silently re-runs every request at a fresh prefix and nothing
    errors. See README.md §C2.
    """

    # Children that DERIVE a workflow_id. hybrid-search is deliberately absent:
    # index_content names its execution uuid4() on purpose (accumulating write to
    # a shared collection) and derives no workflow_id at all, so shipping the
    # module there would be dead code in the zip.
    _IDENTITY_CHILDREN = {
        "MediaProcessingStack", "AudiostemSeparationStack",
        "SpeechEnhancementStack",
        "SpeechDiarizationStack", "ShotSegmentationStack",
        "FaceRecognitionStack", "TtsStack", "BedrockInferenceStack",
        "ConversationAnalysisStack", "ShotMetadataEnrichmentStack",
        "FrameInspectionStack", "ProgramStructureAnalysisStack",
        "AudioDescriptionStack", "CastResolutionStack",
        "ArtworkSelectionStack",
    }

    def _identity_path(self, child):
        return os.path.join(DOCKER_ROOT, CHILDREN[child],
                            "gateway_tool", "workflow_identity.py")

    def _identity_src(self, child):
        path = self._identity_path(child)
        self.assertTrue(os.path.exists(path), f"{child} has no {path}")
        return _read(path)

    def test_the_children_that_derive_a_workflow_id_are_exactly_these(self):
        """Guards the set: a child that starts deriving an id without gaining
        the file (or vice versa) makes every other test here silently skip it."""
        having = {child for child in CHILDREN
                  if os.path.exists(self._identity_path(child))}
        self.assertEqual(
            having, self._IDENTITY_CHILDREN,
            "a child's workflow_identity.py appeared or disappeared — update "
            "_IDENTITY_CHILDREN and confirm the copy is byte-identical")

    def test_every_copy_is_byte_identical(self):
        """Copies that drift are worse than no sharing at all: the file name
        implies they are the same, so a fix applied to one reads as a fix
        applied to all. Byte-identity is only achievable because per-gateway
        variation lives in the CALL (`exclude=`, `extra_material=`), never in
        the module — keep it that way."""
        srcs = {child: self._identity_src(child)
                for child in sorted(self._IDENTITY_CHILDREN)}
        canonical = srcs["MediaProcessingStack"]
        for child, src in srcs.items():
            self.assertEqual(
                src, canonical,
                f"{child}'s workflow_identity.py has drifted from "
                "media-processing's — copy one over the others")

    def test_no_gateway_still_hand_rolls_the_derivation(self):
        """The point of the module is that there is ONE preimage. A gateway that
        keeps its own uuid5 call is drift that the byte-identity test cannot
        see, because that test only compares the copies to each other.

        Walks the AST rather than searching the text: every gateway DOCUMENTS
        the derivation in prose ("workflow_id = uuid5(canonical JSON of ...)"),
        so a text search fails on the docstring that explains the very thing
        being asserted. `ast` makes comments and strings unreachable by
        construction instead of stripping them with a regex that then has to be
        kept correct.
        """
        for child in sorted(self._IDENTITY_CHILDREN):
            path = os.path.join(DOCKER_ROOT, CHILDREN[child],
                                "gateway_tool", "lambda_function.py")
            tree = ast.parse(_read(path))
            called = {
                node.func.attr if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", None)
                for node in ast.walk(tree) if isinstance(node, ast.Call)
            }
            self.assertNotIn(
                "uuid5", called,
                f"{child}'s gateway still calls uuid5 directly — it must go "
                "through compute_workflow_id()")
            imported = {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                and node.module == "workflow_identity"
                for alias in node.names
            }
            # bedrock-inference derives through the workflow's OWN lib.identity
            # wrapper instead (EXCLUDED_KEYS, the records digest, and token
            # folding live there, and its states recompute the hash through the
            # SAME wrapper — two call shapes would be a hash that silently
            # disagrees). The wrapper's delegation to
            # workflow_identity.compute_workflow_id — and its not hand-rolling
            # uuid5 — is pinned by that repo's own suite.
            delegates = any(
                isinstance(node, ast.ImportFrom) and node.module == "lib"
                and any(alias.name == "identity" for alias in node.names)
                for node in ast.walk(tree))
            self.assertTrue(
                "compute_workflow_id" in imported or delegates,
                f"{child}'s gateway neither imports compute_workflow_id nor "
                "delegates through its workflow's lib.identity wrapper")

    def test_the_module_ships_in_the_gateway_zip(self):
        """A copy that is not zipped is not distributed: the Lambda would fail
        at import with ModuleNotFoundError on its first invocation, which is
        deploy-time-green and runtime-broken.

        Parses the `members=(...)` array rather than searching the file text.
        Every build script also NAMES the file in the comment explaining why the
        existence guard is there, so a text search passes even when the member
        has been deleted — measured: removing it from shot-segmentation's array
        left this test green until it was rewritten this way. This is the
        repo's documented "assert against the parsed thing, not the raw text"
        trap, arriving from the opposite direction: here the comment creates a
        false PASS rather than a false failure.
        """
        for child in sorted(self._IDENTITY_CHILDREN):
            build = self.child_build[child]
            match = re.search(r"members=\(([^)]*)\)", build)
            self.assertIsNotNone(
                match,
                f"{CHILDREN[child]}'s build script has no gateway-zip "
                "members=() array — it may have reverted to listing files "
                "inline on the zip call, where a missing file cannot be "
                "guarded (zip exits 0 when only SOME members are missing)")
            members = match.group(1).split()
            self.assertIn(
                "workflow_identity.py", members,
                f"{CHILDREN[child]}'s gateway zip members are {members} — "
                "workflow_identity.py is missing, so the Lambda would fail at "
                "import on its first invocation")
            # The array is only worth having if every member is checked for
            # existence before the zip runs; that guard is what turns a dropped
            # file into a build failure instead of a broken Lambda.
            # legacy inline guard OR cast-resolution's parameterized
            # verify_members guard (2026-09-06, two-directional)
            self.assertTrue(
                'echo "error: gateway_tool/$f not found"' in build
                or ('echo "error: $dir/$f not found"' in build
                    and "verify_members gateway_tool" in build),
                f"{CHILDREN[child]}'s build script lists members but does not "
                "verify each exists before zipping")


class TestGatewayTokenContract(unittest.TestCase):
    """The idempotency-token contract, asserted against every gateway that has
    one — by IMPORTING each and calling it, not by reading its text.

    This lives in the bundle suite because four of the six children
    (media-processing, audiostem-separation, shot-segmentation,
    speech-diarization) have no unittest suite at all: their test/ directories
    hold stress scripts and probes. Their gateways would otherwise have zero
    coverage, and "Ran 0 tests ... OK" reads exactly like a pass.

    Three properties, each of which was a real defect somewhere in this
    repo's history:

      1. The retired `uuid` parameter is REJECTED, not ignored. It meant BOTH
         "which asset" and "same request"; silently ignoring it would start a
         fresh run for a caller who believed they were protected from paying
         twice.
      2. The token NEVER reaches the state machine. It rides in the hash
         preimage as extra material, so this holds by construction — but only
         until someone "helpfully" adds it to sfn_input.
      3. The token is NEVER echoed. The echo was the entire blast radius of the
         2026-08-05 face-recognition bug, where a poll read the key from the
         execution OUTPUT (which no state writes) and so disagreed with its own
         submit response.

    The gateway zips are FLAT (`zip -j`), so `workflow_identity` is a top-level
    module at runtime — each gateway_tool/ therefore goes on sys.path around
    the import, or the module's own import fails here while working in Lambda.
    """

    # child dir -> (submit function, env, extra submit args)
    #
    # The env values are what each gateway reads at IMPORT time; the ones that
    # use os.environ[...] (not .get) raise without them.
    GATEWAYS = {
        "media-processing-on-aws": (
            "_process_media",
            {"MEDIA_PROCESSING_STATE_MACHINE_ARN":
             "arn:aws:states:us-east-1:123456789012:stateMachine:mp"},
            {"asset_s3uri": "s3://in/a.mp4"}),
        "audiostem-separation-on-aws": (
            "_separate_audio_stems",
            {"AUDIOSTEM_STATE_MACHINE_ARN":
             "arn:aws:states:us-east-1:123456789012:stateMachine:as"},
            {"asset_s3uri": "s3://in/a.mp4"}),
        "speech-enhancement-on-aws": (
            "_enhance_speech",
            {"SPEECH_ENHANCEMENT_STATE_MACHINE_ARN":
             "arn:aws:states:us-east-1:123456789012:stateMachine:se"},
            {"asset_s3uri": "s3://in/a.mp4"}),
        "shot-segmentation-on-aws": (
            "_segment_shots",
            {"STATE_MACHINE_ARN":
             "arn:aws:states:us-east-1:123456789012:stateMachine:ss"},
            {"asset_s3uri": "s3://in/a.mp4"}),
        "speech-diarization-on-aws": (
            "_diarize_speech",
            {"DIARIZATION_STATE_MACHINE_ARN":
             "arn:aws:states:us-east-1:123456789012:stateMachine:sd"},
            {"asset_s3uri": "s3://in/a.mp4"}),
        "face-recognition-on-aws": (
            "_recognize_faces",
            {"STATE_MACHINE_ARN":
             "arn:aws:states:us-east-1:123456789012:stateMachine:fr"},
            {"asset_s3uri": "s3://in/a.mp4"}),
        "tts-on-aws": (
            "_synthesize_speech",
            {"SYNTHESIS_STATE_MACHINE_ARN":
             "arn:aws:states:us-east-1:123456789012:stateMachine:tts",
             "REGISTER_VOICE_STATE_MACHINE_ARN":
             "arn:aws:states:us-east-1:123456789012:stateMachine:ttsr",
             "VOICE_TABLE_NAME": "voices"},
            {"voice_id": "v", "transcripts": ["hi"]}),
        "bedrock-inference-on-aws": (
            "_run_inference",
            {"INFERENCE_STATE_MACHINE_ARN":
             "arn:aws:states:us-east-1:123456789012:stateMachine:bi"},
            {"records": [{"record_id": "r0",
                          "messages": [{"role": "user",
                                        "content": [{"text": "Hi"}]}]}]}),
        "conversation-analysis-on-aws": (
            "_analyze_conversation",
            {"STATE_MACHINE_ARN":
             "arn:aws:states:us-east-1:123456789012:stateMachine:ca"},
            {"transcript_s3uri": "s3://test-proxy/micro-workflow/"
             "speech-diarization/x/transcripts/final_transcript.json"}),
        "shot-metadata-enrichment-on-aws": (
            "_enrich_shot_metadata",
            {"STATE_MACHINE_ARN":
             "arn:aws:states:us-east-1:123456789012:stateMachine:sme"},
            {"shot_segmentation_s3uri": "s3://test-proxy/micro-workflow/"
             "shot-segmentation/x/embeddings/shots.json"}),
        "frame-inspection-on-aws": (
            "_inspect_frames",
            {"STATE_MACHINE_ARN":
             "arn:aws:states:us-east-1:123456789012:stateMachine:fi"},
            {"shot_segmentation_s3uri": "s3://test-proxy/micro-workflow/"
             "shot-segmentation/x", "category": "credits-read"}),
        "program-structure-analysis-on-aws": (
            "_analyze_program_structure",
            {"STATE_MACHINE_ARN":
             "arn:aws:states:us-east-1:123456789012:stateMachine:psa"},
            {"shot_segmentation_s3uri": "s3://test-proxy/micro-workflow/"
             "shot-segmentation/x",
             "shot_metadata_s3uri": "s3://test-proxy/micro-workflow/"
             "shot-metadata-enrichment/x"}),
        "audio-description-on-aws": (
            "_generate_ad",
            {"STATE_MACHINE_ARN":
             "arn:aws:states:us-east-1:123456789012:stateMachine:ad"},
            {"shot_segmentation_s3uri": "s3://test-proxy/micro-workflow/"
             "shot-segmentation/x",
             "shot_metadata_s3uri": "s3://test-proxy/micro-workflow/"
             "shot-metadata-enrichment/x",
             "speech_diarization_s3uri": "s3://test-proxy/micro-workflow/"
             "speech-diarization/x",
             "conversation_analysis_s3uri": "s3://test-proxy/micro-workflow/"
             "conversation-analysis/x",
             "face_recognition_s3uri": "s3://test-proxy/micro-workflow/"
             "face-recognition/x",
             "voice": "test-voice"}),
        "cast-resolution-on-aws": (
            "_resolve_cast",
            {"STATE_MACHINE_ARN":
             "arn:aws:states:us-east-1:123456789012:stateMachine:pb"},
            {"speech_diarization_s3uri": "s3://test-proxy/micro-workflow/"
             "speech-diarization/x",
             "face_recognition_s3uri": "s3://test-proxy/micro-workflow/"
             "face-recognition/x"}),
        "artwork-selection-on-aws": (
            "_select_artwork",
            {"STATE_MACHINE_ARN":
             "arn:aws:states:us-east-1:123456789012:stateMachine:aw"},
            {"shot_segmentation_s3uri": "s3://test-proxy/micro-workflow/"
             "shot-segmentation/x",
             "shot_metadata_s3uri": "s3://test-proxy/micro-workflow/"
             "shot-metadata-enrichment/x"}),
    }

    # The two tools NOT tied to a single asset, where an absent token means
    # "do not deduplicate me" rather than "dedup on the parameters". See
    # README.md §C2.
    _GENERATES_A_TOKEN = {"face-recognition-on-aws", "tts-on-aws"}

    def _load(self, child):
        from unittest import mock
        env = dict(self.GATEWAYS[child][1],
                   PROXY_BUCKET="test-proxy", TARGET_NAME=child,
                   TOOL_VERSION="test.0")
        # path + module isolation is gateway_import_isolation's, once
        with gateway_import_isolation(child) as gw_dir:
            spec = importlib.util.spec_from_file_location(
                f"bundle_gw_{child.replace('-', '_')}",
                os.path.join(gw_dir, "lambda_function.py"))
            gw = importlib.util.module_from_spec(spec)
            with mock.patch.dict(os.environ, env), mock.patch("boto3.client"):
                spec.loader.exec_module(gw)
        gw._sfn = mock.MagicMock()
        gw._sfn.start_execution.return_value = {"executionArn": "arn:x"}
        return gw

    def _submit(self, child, **kwargs):
        gw = self._load(child)
        fn, _, base = self.GATEWAYS[child]
        return gw, getattr(gw, fn)(dict(base, **kwargs))

    def _sent(self, gw):
        return json.loads(gw._sfn.start_execution.call_args.kwargs["input"])

    def test_every_gateway_rejects_the_retired_uuid_parameter(self):
        for child in sorted(self.GATEWAYS):
            with self.subTest(child=child):
                gw, out = self._submit(child, uuid="tok-1")
                self.assertIs(out.get("error"), True,
                              f"{child} accepted the retired `uuid`")
                self.assertEqual(out.get("errorCode"), "INVALID_PARAMETER")
                self.assertIn("idempotency_token", out.get("message", ""))
                gw._sfn.start_execution.assert_not_called()

    def test_no_gateway_forwards_the_token_to_the_state_machine(self):
        for child in sorted(self.GATEWAYS):
            with self.subTest(child=child):
                gw, out = self._submit(child, idempotency_token="tok-1")  # nosec B106
                self.assertNotIn("error", out, f"{child}: {out}")
                sent = self._sent(gw)
                self.assertNotIn("idempotency_token", sent)
                self.assertNotIn("correlation_uuid", sent)
                self.assertNotIn(
                    "tok-1", json.dumps(sent),
                    f"{child} leaked the token into the SFN input")

    def test_no_gateway_echoes_the_token(self):
        for child in sorted(self.GATEWAYS):
            with self.subTest(child=child):
                _, out = self._submit(child, idempotency_token="tok-1")  # nosec B106
                self.assertNotIn("uuid", out, f"{child} echoed `uuid`")
                self.assertNotIn("idempotency_token", out)
                self.assertTrue(out.get("workflow_id"),
                                f"{child} returned no workflow_id")

    def test_the_event_uuid_still_flows_and_names_the_execution(self):
        """Load-bearing and deliberately NOT renamed: state-side
        lib/input_schema.py validators in audiostem, shot-segmentation and
        speech-diarization fail the execution outright on a missing
        `event["uuid"]`, and face-recognition rebuilds its semaphore key and
        sandbox collection name from it."""
        for child in sorted(self.GATEWAYS):
            with self.subTest(child=child):
                gw, out = self._submit(child, idempotency_token="tok-1")  # nosec B106
                sent = self._sent(gw)
                self.assertEqual(sent.get("uuid"), out["workflow_id"])
                self.assertEqual(
                    gw._sfn.start_execution.call_args.kwargs["name"],
                    out["workflow_id"])

    def test_a_bad_token_is_rejected_everywhere(self):
        for child in sorted(self.GATEWAYS):
            with self.subTest(child=child):
                gw, out = self._submit(child, idempotency_token="not valid!")  # nosec B106
                self.assertIs(out.get("error"), True)
                self.assertEqual(out.get("errorCode"), "INVALID_PARAMETER")
                gw._sfn.start_execution.assert_not_called()

    def test_the_same_token_with_different_parameters_still_re_runs(self):
        """The token NARROWS dedup rather than replacing the parameter hash, so
        a reused token can never hand back a different request's results."""
        variation = {
            "media-processing-on-aws":     {"width": 1280},
            "audiostem-separation-on-aws": {"asset_s3uri": "s3://in/b.mp4"},
            "speech-enhancement-on-aws": {"asset_s3uri": "s3://in/b.mp4"},
            "shot-segmentation-on-aws":    {"asset_s3uri": "s3://in/b.mp4"},
            "speech-diarization-on-aws":   {"asset_s3uri": "s3://in/b.mp4"},
            "face-recognition-on-aws":     {"asset_s3uri": "s3://in/b.mp4"},
            "tts-on-aws":                  {"transcripts": ["different"]},
            # max_tokens is RETIRED (stripped before the digest — varying it
            # must NOT re-run), so vary a live parameter instead:
            "bedrock-inference-on-aws":    {"temperature": 0.7},
            "conversation-analysis-on-aws": {"use_batch": True},
            # use_batch defaults TRUE here, so vary it the other way:
            "shot-metadata-enrichment-on-aws": {"use_batch": False},
            # use_batch defaults TRUE here too (pooled catalog sweeps):
            "frame-inspection-on-aws": {"use_batch": False},
            # PSA has no inference knobs at all; vary the mode:
            "program-structure-analysis-on-aws": {"mode": "narrative"},
            # AD is prompt-bound (prompt bytes are NOT in the hash — a
            # rebuild re-runs via a fresh token); vary the voice:
            "audio-description-on-aws": {"voice": "another-voice"},
            # use_batch defaults TRUE (batch witness reads); vary it:
            "cast-resolution-on-aws": {"use_batch": False},
            # use_batch defaults FALSE here (single-record rounds):
            "artwork-selection-on-aws": {"use_batch": True},
        }
        for child in sorted(self.GATEWAYS):
            with self.subTest(child=child):
                _, a = self._submit(child, idempotency_token="tok-1")  # nosec B106
                _, b = self._submit(child, idempotency_token="tok-1",  # nosec B106
                                    **variation[child])
                self.assertNotEqual(a["workflow_id"], b["workflow_id"])

    def test_the_token_changes_the_id_only_where_that_is_the_design(self):
        """Two groups, and the split is the substance of the design.

        The four asset-shaped tools dedup on parameters, so a token NARROWS
        that: with and without it are different runs, and tokenless resubmits
        still collapse. face-recognition and tts are not tied to one asset (a
        persistent face collection, a mutable voice catalog), so a tokenless
        call GENERATES a token — two identical tokenless calls are two runs."""
        for child in sorted(self.GATEWAYS):
            with self.subTest(child=child):
                _, with_tok = self._submit(child, idempotency_token="tok-1")  # nosec B106
                _, bare_a = self._submit(child)
                _, bare_b = self._submit(child)
                self.assertNotEqual(with_tok["workflow_id"],
                                    bare_a["workflow_id"])
                if child in self._GENERATES_A_TOKEN:
                    self.assertNotEqual(
                        bare_a["workflow_id"], bare_b["workflow_id"],
                        f"{child} deduplicated a caller who supplied no token")
                else:
                    self.assertEqual(
                        bare_a["workflow_id"], bare_b["workflow_id"],
                        f"{child} failed to dedup on parameters alone")

    # ── Phase 3: tools that cannot honour a token ACCEPT it and log ───────────
    #
    # The opposite of the submit path above, and deliberately so. Rejecting is
    # right where the token changes what runs — a malformed one would become an
    # execution name, and a retired `uuid` might mean either of two things. It
    # is wrong on a poll: the caller is an LLM, a hallucinated parameter name is
    # a routine failure mode, and erroring would break polling *mid-job*, which
    # is the worst possible place to fail a caller whose request was otherwise
    # correct. Neither key could have changed what a poll returns anyway.

    def _poll(self, child, **kwargs):
        """Poll through `lambda_handler`, not `_get_job_status`.

        The advisory warn lives at the dispatcher — once, before routing, so a
        tool added later is covered by default rather than by someone
        remembering. Calling the handler function directly would bypass it and
        this whole group would pass against no implementation at all.
        """
        from unittest import mock
        gw = self._load(child)
        gw._sfn.describe_execution.return_value = {
            "executionArn": ("arn:aws:states:us-east-1:123456789012:"
                             "execution:sm:wf-1"),
            "status": "RUNNING", "name": "wf-1", "input": "{}",
        }
        gw._sfn.get_execution_history.return_value = {"events": []}
        job_id = ("arn:aws:states:us-east-1:123456789012:execution:sm:wf-1")
        ctx = mock.MagicMock()
        ctx.client_context.custom = {
            "bedrockAgentCoreToolName": f"{child}___get_job_status"}
        with mock.patch.object(gw.logger, "warning") as warn:
            out = gw.lambda_handler(dict({"job_id": job_id}, **kwargs), ctx)
        return out, warn

    def test_a_poll_accepts_a_token_instead_of_failing_the_caller(self):
        for child in sorted(self.GATEWAYS):
            with self.subTest(child=child):
                out, _ = self._poll(child, idempotency_token="tok-1")  # nosec B106
                self.assertNotIn(
                    "error", out,
                    f"{child} failed a poll over an ignorable field")

    def test_a_poll_accepts_the_retired_uuid_too(self):
        """Rejected on submit, accepted here — the asymmetry is the design.

        On submit, `uuid` is ambiguous between two retired meanings, so guessing
        is worse than erroring. On a poll it is unambiguous: it means nothing at
        all, because `job_id` already identifies the execution.
        """
        for child in sorted(self.GATEWAYS):
            with self.subTest(child=child):
                out, _ = self._poll(child, uuid="tok-1")
                self.assertNotIn(
                    "error", out,
                    f"{child} rejected a poll carrying the retired `uuid`")

    def test_an_ignored_token_is_logged_not_silent(self):
        """Accepting is only defensible if it leaves a trace — otherwise the
        person debugging "why did this run twice?" has nothing to read."""
        for child in sorted(self.GATEWAYS):
            for field in ("idempotency_token", "uuid"):
                with self.subTest(child=child, field=field):
                    _, warn = self._poll(child, **{field: "tok-1"})
                    self.assertTrue(
                        warn.called,
                        f"{child} ignored {field} on a poll without logging it")

    def test_a_clean_poll_logs_no_token_warning(self):
        """The warning must be conditional. A warning on every poll is noise,
        and noise is what makes the real one unreadable."""
        for child in sorted(self.GATEWAYS):
            with self.subTest(child=child):
                _, warn = self._poll(child)
                msgs = " ".join(str(c) for c in warn.call_args_list)
                self.assertNotIn("idempotency_token", msgs,
                                 f"{child} warned about a token nobody sent")

    def test_the_warn_helper_never_returns_an_error_shape(self):
        """`warn_idempotency_token` returns a LOG LINE, never an error dict —
        a caller who wired it like `read_idempotency_token` (which does return
        an error) would silently reintroduce rejection."""
        for child in sorted(self.GATEWAYS):
            with self.subTest(child=child):
                gw = self._load(child)
                for event in ({"idempotency_token": "tok"}, {"uuid": "tok"}):  # nosec B105
                    note = gw.warn_idempotency_token(event)
                    self.assertIsInstance(note, str)
                    self.assertTrue(note)
                self.assertIsNone(gw.warn_idempotency_token({}))
                self.assertIsNone(
                    gw.warn_idempotency_token({"idempotency_token": ""}))  # nosec B105

    def test_the_warning_makes_no_claim_about_starting_an_execution(self):
        """The message must describe the TOKEN, never what the tool does.

        It read "this tool starts no execution, so every call runs and reflects
        current state" until 2026-08-06. One line, shared by six gateways, and
        false for FIVE of the tools it was emitted for: face-recognition's
        `batch_index_faces`, `batch_delete_faces` and `delete_collection` all
        start executions, hybrid-search's `index_content` starts one, and tts's
        `register_voice` both starts one AND dedups it on the parameter hash —
        so "every call runs" was precisely backwards there.

        Why that mattered more than a wrong log line usually does: the sentence
        argued *against* `register_voice`'s exemption, in the one place a
        maintainer reads before changing it. The exemption is load-bearing —
        `write_voice_record` mints `voice_id = uuid4()` under an unconditional
        `put_item` and no tool deletes a voice, so a duplicate registration is a
        permanent, unremovable catalog entry.

        The set of tools that ignore a token is NOT the set that starts no
        execution, and this asserts the message never conflates them again.
        Phrase-level rather than semantic on purpose: this is a wording
        regression, and wording is what has to be pinned.
        """
        forbidden = (
            "starts no execution",
            "start no execution",
            "every call runs",
        )
        for child in sorted(self.GATEWAYS):
            with self.subTest(child=child):
                gw = self._load(child)
                for event in ({"idempotency_token": "tok"}, {"uuid": "tok"}):  # nosec B105
                    note = gw.warn_idempotency_token(event).lower()
                    for phrase in forbidden:
                        self.assertNotIn(
                            phrase, note,
                            f"{child}: the ignore-warning claims {phrase!r}, "
                            "which is false for the index tools, "
                            "index_content and register_voice — say what is "
                            "true of the TOKEN instead")

    def test_a_tool_that_ignores_a_token_may_still_start_an_execution(self):
        """The counter-example, executed rather than asserted in prose.

        tts's `register_voice` is the sharpest case in the repo: it ignores a
        token, starts an execution, and deduplicates that execution on the
        parameter hash alone. Two identical registrations therefore collapse to
        ONE run — the opposite of the retired warning's claim. Pinned here so
        the message and the behavior can never drift apart again.
        """
        from unittest import mock
        gw = self._load("tts-on-aws")
        ev = {"ref_s3uri": "s3://test-proxy/ref.wav", "ref_transcript": "hi",
              "name": "n", "gender": "f", "language": "en"}
        ctx = mock.MagicMock()
        ctx.client_context.custom = {
            "bedrockAgentCoreToolName": "text-to-speech___register_voice"}

        with mock.patch.object(gw.logger, "warning") as warn:
            a = gw.lambda_handler(dict(ev, idempotency_token="tok-1"), ctx)  # nosec B106
        self.assertNotIn("error", a, f"register_voice rejected a token: {a}")
        self.assertTrue(warn.called, "the ignored token was not logged")

        b = gw.lambda_handler(dict(ev), ctx)
        self.assertEqual(
            a["workflow_id"], b["workflow_id"],
            "register_voice must dedup on parameters alone — the token is "
            "ignored, so supplying one cannot change the run")
        self.assertTrue(
            gw._sfn.start_execution.called,
            "register_voice starts an execution; a warning that says "
            "otherwise is the bug this test exists for")


class TestSubmitReturnsAHandleNotResults(TestGatewayTokenContract):
    """The submit/poll contract, EXECUTED against every gateway that starts an
    execution — four of which have no unittest suite of their own.

    Reuses TestGatewayTokenContract's loader (a real import of each
    gateway_tool/lambda_function.py with boto3 mocked), so these are the actual
    shipped code paths, not a re-implementation of what they should do.

    The contract, agreed 2026-08-07:

      submit, fresh              job_id, workflow_id, status=RUNNING, + per-target
                                 handle fields (current_state, asset_s3uri,
                                 output_s3_prefix)
      submit, absorbed RUNNING   the same, + message
      submit, absorbed SUCCEEDED the same, + message              (NO results)
      submit, absorbed terminal  the same, + message + error      (NO results)
      get_job_status             the ONLY place results come from

    Why it is shaped this way. Before this, an absorbed resubmit called
    _parse_succeeded_output(existing) and handed back the finished run's entire
    result payload — up to 23 fields on speech-diarization, against 5 declared in
    cfnstack.yaml. That is three defects in one:

      * The reachable field set could not be declared, because it was the union
        of the handle shape and every result shape. ~70 fields were undeclared
        across the six submits.
      * The agent needed two readers for one response, and no way to know in
        advance which it would get.
      * A submit could dump an unbounded results payload (tts: one entry per
        transcript) into an agent that asked for a handle.

    Making the absorbed response a handle whose ONLY variable is `status` fixes
    all three by construction: the reachable field set IS the declared field set,
    and there is one rule — "a submit names a run; get_job_status reports it".

    `message` is the sole exception, and it is load-bearing: every other field is
    identical to a fresh start's, so without it nothing distinguishes "I started
    your run" from "your run already existed". `error` is the second, present only
    on a terminal absorb, because the agent has to know WHY to choose between
    changing a parameter and changing the idempotency_token — a plain retry
    cannot clear a failure that is sticky for as long as Step Functions reserves
    the execution name.
    """

    # Fields that may appear on a submit response beyond the fresh-start set.
    # error_code joined 2026-10-15: a terminal absorb carries the stable code
    # (the raised exception's class name) beside the compacted error, exactly
    # like get_job_status — the absorbed resubmit is the one response an agent
    # gets when it DOES resubmit a failed run.
    _ABSORB_ONLY = {"message", "error", "error_code"}

    # The one field a fresh start carries that a STOPPED absorb legitimately
    # does not: `current_state` names the state an execution is sitting in, which
    # is meaningless once it has stopped — and reading it costs a
    # GetExecutionHistory paging pass, so returning "unknown" would be paying for
    # a non-answer. tts has no current_state on any path (two state machines, no
    # helper), so its fresh set simply never contains it.
    _RUNNING_ONLY = {"current_state"}

    # Result-shaped fields that must NEVER appear on a submit response. Each was
    # reachable from a submit before this change.
    _RESULT_FIELDS = {
        "outputs", "batch_qc_summary", "voice", "usage_uri", "usage_key",
        "pipeline_wall_sec", "total_estimated_cost", "shots_key",
        "embeddings_key", "centroids_key", "recurrence_key", "n_shots",
        "n_recurrence_groups", "total_frames", "actual_duration_sec",
        "speech_key", "music_key", "sfx_key", "metrics_key",
        "transcript_json_key", "transcript_vtt_key", "speaker_manifest_key",
        "global_speakers_key", "num_sentences", "num_speakers",
        "clustering_method", "single_speaker_detected", "duration_sec",
        "has_video", "has_audio", "is_all_intra", "output_keys",
        "probe_s3_key", "num_people_recognized", "media_processing_arn",
        "execution_arn",
    }

    def _absorbed(self, child, existing):
        """Submit into an existing execution: start_execution raises
        ExecutionAlreadyExists and describe_execution returns `existing`."""
        from unittest import mock
        gw = self._load(child)
        fn, _, base = self.GATEWAYS[child]
        from botocore.exceptions import ClientError
        # Five gateways catch botocore's ClientError and switch on
        # response["Error"]["Code"]; tts catches the client's own typed
        # ExecutionAlreadyExists. One exception object has to satisfy both
        # readings, so raise a ClientError subclass and register that subclass
        # as the client's typed exception.
        already = type("ExecutionAlreadyExists", (ClientError,), {})
        err = already({"Error": {"Code": "ExecutionAlreadyExists",
                                 "Message": "dup"}}, "StartExecution")
        gw._sfn.exceptions.ExecutionAlreadyExists = already
        gw._sfn.start_execution.side_effect = err
        gw._sfn.describe_execution.return_value = existing
        # _get_current_state pages execution history; a RUNNING absorb calls it.
        gw._sfn.get_execution_history.return_value = {"events": []}
        return getattr(gw, fn)(dict(base))

    def _fresh(self, child):
        gw, out = self._submit(child)
        return out

    _EXISTING_ARN = "arn:aws:states:us-east-1:123456789012:execution:sm:e"

    def _existing(self, status, **extra):
        return dict({
            "executionArn": self._EXISTING_ARN,
            "status": status,
            "input": json.dumps({"input": {"s3uri": "s3://in/a.mp4"},
                                 "output": {"s3uri": "s3://test-proxy/out"}}),
        }, **extra)

    def test_a_fresh_submit_returns_a_handle(self):
        """job_id and workflow_id on every fresh start, status RUNNING, and no
        result field anywhere. RUNNING specifically: tts used to answer
        "STARTED", a seventh status word for a state Step Functions and every
        sibling call RUNNING — which only worked while a fresh start was the one
        path that could produce it."""
        for child in sorted(self.GATEWAYS):
            with self.subTest(child=child):
                out = self._fresh(child)
                self.assertNotIn("error", out, out)
                self.assertIn("job_id", out)
                self.assertIn("workflow_id", out)
                self.assertEqual(out["status"], "RUNNING")
                self.assertFalse(
                    self._RESULT_FIELDS & set(out),
                    f"{child}: a fresh submit returned result fields "
                    f"{sorted(self._RESULT_FIELDS & set(out))}")

    def test_an_absorbed_resubmit_differs_from_a_fresh_start_only_in_status(self):
        """The invariant the whole change exists to create, for all three
        existing-run states. One reader serves every submit response."""
        for child in sorted(self.GATEWAYS):
            fresh = set(self._fresh(child))
            for status in ("RUNNING", "SUCCEEDED", "FAILED"):
                with self.subTest(child=child, existing=status):
                    out = self._absorbed(child, self._existing(status))
                    expected = fresh if status == "RUNNING" else (
                        fresh - self._RUNNING_ONLY)
                    self.assertEqual(
                        set(out) - self._ABSORB_ONLY, expected,
                        f"{child}: an absorbed resubmit of a {status} run must "
                        "carry exactly a fresh start's fields, plus message "
                        "(and error on a terminal failure); current_state is "
                        "dropped once the run has stopped")
                    self.assertEqual(out["status"], status)

    def test_current_state_is_reported_while_running_and_dropped_once_stopped(self):
        """The one deliberate asymmetry, stated positively so neither half can
        be silently lost: a RUNNING absorb pays for GetExecutionHistory because
        "which state is it in" is the question the caller has; a stopped one does
        not, because the answer would be a meaningless "unknown" bought with a
        paging pass."""
        for child in sorted(self.GATEWAYS):
            if "current_state" not in self._fresh(child):
                continue          # tts: no current_state on any path
            with self.subTest(child=child, existing="RUNNING"):
                out = self._absorbed(child, self._existing("RUNNING"))
                self.assertIn("current_state", out)
            for status in ("SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"):
                with self.subTest(child=child, existing=status):
                    out = self._absorbed(child, self._existing(status))
                    self.assertNotIn("current_state", out)

    # The MP consumers whose gateways compact an absorbed terminal failure
    # the way their get_job_status does (2026-10-15, the fail-by-name set):
    # stable code in error_code, 'code: message' in error, never a raw
    # stack-trace blob. The other gateways still return the raw cause there —
    # extend this set when one of them adopts the compaction.
    _COMPACT_ABSORB_CHILDREN = (
        "audiostem-separation-on-aws", "speech-diarization-on-aws",
        "speech-enhancement-on-aws", "face-recognition-on-aws")

    def test_an_absorbed_terminal_failure_carries_the_stable_code(self):
        """The absorbed resubmit is the one response an agent gets when it
        DOES resubmit a failed run — the schema's 'match on error_code, do
        not resubmit' instruction must be followable there, not only on a
        poll."""
        cause = json.dumps({
            "errorType": "NoAudioStream",
            "errorMessage": "the source has no audio stream",
            "stackTrace": ["x" * 400] * 8})
        for child in self._COMPACT_ABSORB_CHILDREN:
            with self.subTest(child=child):
                out = self._absorbed(child, self._existing(
                    "FAILED", error="NoAudioStream", cause=cause))
                self.assertEqual(out["error_code"], "NoAudioStream")
                self.assertIn("no audio stream", out["error"])
                self.assertNotIn("stackTrace", out["error"])
                self.assertLessEqual(len(out["error"]), 600)

    def test_an_absorbed_resubmit_never_returns_the_existing_runs_results(self):
        """The specific regression: absorbed-SUCCEEDED used to inline
        _parse_succeeded_output(existing). A full result payload on a submit is
        undeclarable and unbounded."""
        succeeded = self._existing("SUCCEEDED", output=json.dumps({
            "uuid": "u",
            "output": {"s3uri": "s3://test-proxy/out"},
            "data": {
                "shots_key": "k", "n_shots": 7, "embeddings_key": "e",
                "audiostem_sep_output": {"speech_key": "s", "metrics_key": "m"},
                "final_transcript_key": "t", "num_speakers": 2,
                "duration_sec": 12.0, "has_video": True,
                "output_bucket": "test-proxy", "proxy_bucket": "test-proxy",
                "proxy_prefix": "out",
                "iterators": [{"out_bucket": "b", "outputs": [
                    {"wav_key": "k0", "duration_s": 1.0}]}],
                "usage_tracking": {"pipeline_wall_sec": 1.0,
                                   "total_estimated_cost": 0.01,
                                   "usage_key": "u.json"},
            },
        }))
        for child in sorted(self.GATEWAYS):
            with self.subTest(child=child):
                out = self._absorbed(child, succeeded)
                leaked = self._RESULT_FIELDS & set(out)
                self.assertFalse(
                    leaked,
                    f"{child}: an absorbed resubmit leaked result fields "
                    f"{sorted(leaked)} — results belong to get_job_status")

    def test_an_absorbed_resubmit_says_it_did_not_resubmit(self):
        """`message` is the ONLY absorption signal now that every other field
        matches a fresh start. Before this change only tts set it, so on the
        other five an absorbed call was indistinguishable from a new run unless
        the caller happened to notice a non-RUNNING status.

        The wording is pinned byte-identically across all six, because it is the
        text an agent reads to decide whether it just spent money."""
        expected = (
            "identical request already submitted — this call did not resubmit; "
            "call get_job_status with this job_id for status and results")
        for child in sorted(self.GATEWAYS):
            for status in ("RUNNING", "SUCCEEDED", "FAILED"):
                with self.subTest(child=child, existing=status):
                    out = self._absorbed(child, self._existing(status))
                    self.assertEqual(out.get("message"), expected)

    def test_an_absorbed_terminal_failure_carries_its_cause(self):
        """A terminal failure is STICKY: Step Functions keeps the execution name
        reserved, so this same request keeps returning this same verdict. The
        escape is a changed parameter or a different idempotency_token, never a
        plain retry — and the agent cannot choose between those two without the
        cause."""
        for child in sorted(self.GATEWAYS):
            for status in ("FAILED", "TIMED_OUT", "ABORTED"):
                with self.subTest(child=child, existing=status):
                    out = self._absorbed(
                        child, self._existing(status, cause="boom"))
                    self.assertEqual(out["status"], status)
                    self.assertEqual(out["error"], "boom")

    def test_an_absorbed_success_carries_no_error(self):
        """The mirror of the above: `error` on a SUCCEEDED absorb would read as
        a failed run to any agent that checks for the key rather than the
        status."""
        for child in sorted(self.GATEWAYS):
            for status in ("RUNNING", "SUCCEEDED"):
                with self.subTest(child=child, existing=status):
                    out = self._absorbed(
                        child, self._existing(status, cause="stale"))
                    self.assertNotIn("error", out)

    def test_every_poll_branch_names_the_run(self):
        """`job_id` and `workflow_id` on EVERY get_job_status branch, including
        the fallback for a status Step Functions has not shipped yet.

        Before this, five of the six omitted job_id on the RUNNING and terminal
        branches and tts returned a bare {"status": "RUNNING"} — so an agent
        that resumed a conversation holding one poll response could not name the
        run it had just read."""
        from unittest import mock
        arn = self._EXISTING_ARN
        for child in sorted(self.GATEWAYS):
            gw = self._load(child)
            gw._sfn.get_execution_history.return_value = {"events": []}
            for status in ("RUNNING", "SUCCEEDED", "FAILED", "TIMED_OUT",
                           "ABORTED", "PENDING_REDRIVE"):
                with self.subTest(child=child, status=status):
                    gw._sfn.describe_execution.return_value = self._existing(
                        status, output=json.dumps({"data": {}}))
                    out = gw._get_job_status({"job_id": arn})
                    self.assertEqual(out.get("job_id"), arn,
                                     f"{child}/{status}: no job_id")
                    self.assertEqual(out.get("workflow_id"), "e",
                                     f"{child}/{status}: no workflow_id")

    def test_the_redundant_execution_arn_echo_is_gone(self):
        """tts and speech-diarization each returned an `execution_arn` that was
        byte-identical to the `job_id` the caller had just passed in — one value
        under two names, which invites an agent to treat them as two different
        handles. Removed; `media_processing_arn` stays, because that one names a
        genuinely different (child) execution."""
        arn = self._EXISTING_ARN
        for child in sorted(self.GATEWAYS):
            gw = self._load(child)
            gw._sfn.describe_execution.return_value = self._existing(
                "SUCCEEDED",
                output=json.dumps({"execution_arn": arn, "data": {}}))
            with self.subTest(child=child):
                out = gw._get_job_status({"job_id": arn})
                self.assertNotIn("execution_arn", out)


class TestWorkflowIdGoldenVectors(BundleTestCase):
    """Pins the HASH PREIMAGE with known-answer vectors.

    This is the test the nine hand-copied derivations never had, and the reason
    their drift went unnoticed. Every other property of a workflow_id
    (determinism, key-order independence) is self-consistent — it holds equally
    well for a WRONG preimage. Only a fixed input mapped to a fixed output
    string catches "the preimage changed", which is the one failure mode here
    that is both silent and expensive.

    A failure means one of two things:
      - an unintended preimage change  -> fix the code, not the vector;
      - an intended one                -> update the vector AND bump every
                                          affected .version, because old and
                                          new code now disagree about the
                                          identity of the same request.
    """

    TOOL_VERSION = "golden.1"

    def _module(self):
        path = os.path.join(DOCKER_ROOT, "media-processing-on-aws",
                            "gateway_tool", "workflow_identity.py")
        spec = importlib.util.spec_from_file_location("wf_identity_golden", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    # (label, payload, exclude, expected workflow_id)
    #
    # One vector per SHAPE in scope, not per gateway: the six copies are
    # byte-identical (TestWorkflowIdentityParity), so what varies between
    # gateways is the CALL. Three distinct calls exist today — the asset-shaped
    # six, face-recognition's token fold, and tts's `uuid` exclusion.
    VECTORS = (
        ("asset-shaped, no token",
         {"input": {"s3uri": "s3://ingest/a.mp4",
                    "input_config": {"frame_interval_sec": 2}},
          "output": {"s3uri": "s3://p/micro-workflow/t/DERIVED",
                     "output_groups": [{"group_name": "g"}]},
          "data": {}},
         ("output.s3uri",),
         "6fef13a0-4432-52ca-afa4-419b3a093b7c"),
        ("face-recognition, token folded in by the caller",
         {"input": {"s3uri": "s3://ingest/a.mp4",
                    "input_config": {"collection_id": "c1"}},
          "output": {"s3uri": "s3://p/micro-workflow/face-recognition/DERIVED"},
          "idempotency_token": "tok-1",  # nosec B105
          "data": {}},
         ("output.s3uri",),
         "cac9ad1d-48fe-5330-a4ea-38c9a47fe6ea"),
        ("face-recognition, same request WITHOUT the token",
         {"input": {"s3uri": "s3://ingest/a.mp4",
                    "input_config": {"collection_id": "c1"}},
          "output": {"s3uri": "s3://p/micro-workflow/face-recognition/DERIVED"},
          "data": {}},
         ("output.s3uri",),
         "22dcc3a2-ab5a-5058-97dc-63007c18a319"),
        ("tts, uuid IS the workflow_id so it is excluded",
         {"uuid": "SHOULD-NOT-MATTER",
          "input": {"voice_id": "v1", "transcripts": [{"text": "hello"}]},
          "output": {"s3uri": "s3://p/micro-workflow/text-to-speech/DERIVED"},
          "data": {}},
         ("uuid",),
         "ad6e5623-e9c6-5e17-a5b8-dc7f795aaa8d"),
    )

    def test_the_preimage_is_pinned(self):
        mod = self._module()
        for label, payload, exclude, expected in self.VECTORS:
            got = mod.compute_workflow_id(
                payload, tool_version=self.TOOL_VERSION, exclude=exclude)
            self.assertEqual(got, expected, f"preimage changed for {label}")

    def test_the_token_narrows_dedup_rather_than_replacing_it(self):
        """Vectors 2 and 3 are the same request differing only by the token, so
        they must differ — and vector 2 must still differ from a run with the
        same token and DIFFERENT parameters. That second half is what makes a
        reused token safe: it can never hand back another asset's results."""
        mod = self._module()
        _, with_token, exclude, _ = self.VECTORS[1]
        _, without_token, _, _ = self.VECTORS[2]

        def wid(payload):
            return mod.compute_workflow_id(
                payload, tool_version=self.TOOL_VERSION, exclude=exclude)

        self.assertNotEqual(wid(with_token), wid(without_token))
        other_asset = json.loads(json.dumps(with_token))
        other_asset["input"]["s3uri"] = "s3://ingest/b.mp4"
        self.assertNotEqual(
            wid(with_token), wid(other_asset),
            "the same token collapsed two different assets onto one run")

    def test_the_derived_output_prefix_cannot_influence_the_id(self):
        """THE circularity invariant: output.s3uri is DERIVED from the id, so it
        cannot be in the preimage. Enforced via `exclude` rather than by
        assigning it after the hash — the old ordering was an invariant held
        only by statement order, invisible to a reader and broken by any edit
        that moves a line."""
        mod = self._module()
        base = {"input": {"s3uri": "s3://ingest/a.mp4"}, "output": {}, "data": {}}
        without = mod.compute_workflow_id(
            base, tool_version=self.TOOL_VERSION, exclude=("output.s3uri",))
        for prefix in ("s3://p/micro-workflow/t/abc", "s3://other/x", ""):
            with_prefix = dict(base, output={"s3uri": prefix})
            self.assertEqual(
                without,
                mod.compute_workflow_id(with_prefix,
                                        tool_version=self.TOOL_VERSION,
                                        exclude=("output.s3uri",)),
                "output.s3uri reached the preimage")

    def test_a_version_bump_re_runs_everything(self):
        """tool_version is appended so changed shaping cannot return a result
        produced by different code."""
        mod = self._module()
        payload = {"input": {"s3uri": "s3://ingest/a.mp4"}}
        self.assertNotEqual(
            mod.compute_workflow_id(payload, tool_version="v1"),
            mod.compute_workflow_id(payload, tool_version="v2"))

    def test_key_order_and_whitespace_do_not_matter(self):
        """canonical_json sorts keys and strips whitespace, so two callers that
        build the same dict differently must collide."""
        mod = self._module()
        a = {"b": 2, "a": {"y": 1, "x": [1, 2]}}
        b = {"a": {"x": [1, 2], "y": 1}, "b": 2}
        self.assertEqual(mod.compute_workflow_id(a, tool_version="v"),
                         mod.compute_workflow_id(b, tool_version="v"))

    def test_extra_material_is_content_addressing(self):
        """Reserved for the out-of-scope gateways (ETags, records digest): the
        same parameters over CHANGED content must re-run."""
        mod = self._module()
        payload = {"input": {"s3uri": "s3://ingest/a.mp4"}}
        self.assertNotEqual(
            mod.compute_workflow_id(payload, tool_version="v",
                                    extra_material=("etag-1",)),
            mod.compute_workflow_id(payload, tool_version="v",
                                    extra_material=("etag-2",)))


# ---------------------------------------------------------------------------
# usage_tracking cost parity — see README.md §C10
# ---------------------------------------------------------------------------

# Every docker/ stack whose usage_tracking state walks Step Functions execution
# history to price its own Lambda cost. Deliberately NOT the CHILDREN map:
# hybrid-search and media-toolkit-gateway have no usage_tracking state at all. The bug
# this class pins was identical in seven copies, so the set that matters here is
# "has the code", not "is nested by the bundle".
#
# The set is what is PRESENT IN THIS TREE, and `_load_usage_module` executes each
# entry's module — so an entry for a directory that is not checked out is not a
# skipped assertion, it is an ERROR in setUpClass that takes the whole class down
# and with it the seven stacks that ARE here. `conversation-analysis` and
# `deepfilternet3-on-aws` were listed and are absent; both are gone from the tuple
# for that reason. If either is checked out later,
# `test_the_usage_tracking_stacks_are_exactly_these` fails and says to add it back
# — which is that test working, not breaking.
USAGE_TRACKING_STACKS = (
    "audiostem-separation-on-aws",
    # Added when bedrock-inference-on-aws absorbed the former service-quotas-on-aws
    # and grew an orchestrator to route between its two children. Its usage state is
    # the one exception to the shape below: it walks TWO histories, its own and
    # whichever child ran, because the orchestrator's own history is three short
    # Lambda calls and one `.sync:2` span while every Converse retry and the whole
    # lease wait live in the child.
    "bedrock-inference-on-aws",
    "face-recognition-on-aws",
    "media-processing-on-aws",
    "shot-segmentation-on-aws",
    "speech-diarization-on-aws",
    "speech-enhancement-on-aws",
    "tts-on-aws",
)

# usage_tracking states that are AGGREGATORS, not history billers: the
# llm-prompt workflows' usage states total their SIBLING runs' own
# usage documents (bedrock-inference / frame-inspection publish them)
# and never parse Lambda execution history, so the billing assertions
# in TestUsageTrackingBillsFailedAttempts do not apply. They became
# DISCOVERABLE with the 2026-10-06 tree restructure (before it they
# were flat states/usage_tracking.py and the guard below could not see
# them) — listed here so the exemption is a recorded decision, never
# silence.
USAGE_AGGREGATOR_STACKS = (
    "audio-description-on-aws",
    "conversation-analysis-on-aws",
    "frame-inspection-on-aws",
    "shot-metadata-enrichment-on-aws",
)

_USAGE_MEMORY_MB = 1792                       # 1-core, the shared tier vocabulary
_USAGE_ARCH = "arm64"
_USAGE_PER_GB_SEC = 0.0000133334
_USAGE_PER_REQUEST = 0.0000002
_USAGE_ARN = ("arn:aws:lambda:us-east-1:000000000000:function:"
              "probe-1-core")
_USAGE_EXEC_ARN = ("arn:aws:states:us-east-1:000000000000:execution:"
                   "probe:00000000")


class _UsageFakeLambda:
    """GetFunctionConfiguration for the one ARN the synthetic histories name."""

    def get_function_configuration(self, FunctionName):
        return {"MemorySize": _USAGE_MEMORY_MB,
                "Architectures": [_USAGE_ARCH]}


class _UsageFakeS3:
    def put_object(self, **kwargs):
        return {}


class _UsageFakeSfn:
    def __init__(self, events):
        self.events = events

    def get_execution_history(self, **kwargs):
        return {"events": self.events}


class _UsageAnyModule(types.ModuleType):
    """lib.s3_utils stand-in: every helper resolves to a no-op.

    The eight stacks import different subsets by name (`put_json`, `get_json`,
    `parse_s3uri`, `s3`), and none of them is reached — only the history
    walkers are driven here, never a handler.
    """

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *args, **kwargs: None


def _load_usage_module(stack):
    """Import a stack's usage_tracking module with boto3 and lib/ stubbed.

    These modules build boto3 clients at import time, so the stubs have to be
    in sys.modules BEFORE exec_module and out of it afterwards — a leaked
    `boto3` stub would silently disarm every other test that imports one.
    """
    path = os.path.join(DOCKER_ROOT, stack, "states", "usage_tracking",
                        "lambda_function.py")
    boto3_stub = types.ModuleType("boto3")
    boto3_stub.client = lambda name, **kwargs: (
        _UsageFakeLambda() if name == "lambda" else _UsageFakeS3())
    lib_stub = types.ModuleType("lib")
    s3_stub = _UsageAnyModule("lib.s3_utils")
    s3_stub.s3 = _UsageFakeS3()
    lib_stub.s3_utils = s3_stub
    # face-recognition's usage state imports its counter-key tuples from
    # lib.input_schema (single source of truth, 2026-09-07). That module
    # is client-free by contract, so load the REAL file — a stub with
    # invented tuples would let the mirror drift this import exists to
    # prevent.
    stubbed = {"boto3": boto3_stub, "lib": lib_stub,
               "lib.s3_utils": s3_stub}
    schema_path = os.path.join(DOCKER_ROOT, stack, "lib", "input_schema.py")
    needs_schema = "lib.input_schema" in open(path, encoding="utf-8").read()
    if needs_schema and os.path.exists(schema_path):
        schema_spec = importlib.util.spec_from_file_location(
            "lib.input_schema", schema_path)
        schema_mod = importlib.util.module_from_spec(schema_spec)
        schema_spec.loader.exec_module(schema_mod)
        lib_stub.input_schema = schema_mod
        stubbed["lib.input_schema"] = schema_mod

    # The tree-restructured stacks (2026-10-06) import their own lib
    # modules package-style, and lib modules import EACH OTHER — no
    # import scan bottoms out. For a stack whose lib/ is a real package
    # (lib/__init__.py exists), load the REAL package: repo root on
    # sys.path, boto3 stubbed, and the `lib` namespace quarantined per
    # stack (two stacks' lib packages share a name; a leaked module
    # would hand stack B stack A's lib). Pre-restructure stacks keep
    # the stub path.
    repo_root = os.path.join(DOCKER_ROOT, stack)
    if os.path.exists(os.path.join(repo_root, "lib", "__init__.py")):
        saved_mods = {k: sys.modules[k] for k in list(sys.modules)
                      if k == "boto3" or k == "lib"
                      or k.startswith("lib.")}
        for k in saved_mods:
            if k != "boto3":
                del sys.modules[k]
        # a REAL s3_utils does `from boto3.s3.transfer import
        # TransferConfig` — the stub grows that submodule chain
        b3_s3 = types.ModuleType("boto3.s3")
        b3_transfer = types.ModuleType("boto3.s3.transfer")
        b3_transfer.TransferConfig = lambda *a, **k: None
        b3_s3.transfer = b3_transfer
        boto3_stub.s3 = b3_s3
        saved_mods.setdefault("boto3.s3", sys.modules.get("boto3.s3"))
        saved_mods.setdefault("boto3.s3.transfer",
                              sys.modules.get("boto3.s3.transfer"))
        sys.modules["boto3"] = boto3_stub
        sys.modules["boto3.s3"] = b3_s3
        sys.modules["boto3.s3.transfer"] = b3_transfer
        sys.path.insert(0, repo_root)
        try:
            modname = "usage_tracking_" + re.sub(r"\W", "_", stack)
            spec = importlib.util.spec_from_file_location(modname, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        finally:
            sys.path.remove(repo_root)
            for k in [k for k in sys.modules
                      if k in ("boto3", "boto3.s3", "boto3.s3.transfer",
                               "lib") or k.startswith("lib.")]:
                del sys.modules[k]
            sys.modules.update(
                {k: v for k, v in saved_mods.items() if v is not None})

    names = tuple(stubbed)
    saved = {name: sys.modules.get(name) for name in names}
    sys.modules.update(stubbed)
    try:
        modname = "usage_tracking_" + re.sub(r"\W", "_", stack)
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


_USAGE_T0 = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)


def _usage_at(seconds):
    return _USAGE_T0 + datetime.timedelta(seconds=seconds)


def _usage_event(event_id, previous_id, etype, seconds, **details):
    event = {"id": event_id, "previousEventId": previous_id, "type": etype,
             "timestamp": _usage_at(seconds)}
    event.update(details)
    return event


def _usage_scheduled(event_id, previous_id, seconds, operation):
    """A `TaskScheduled` in the SDK-integration dialect.

    Every real history examined for the fix (tts, face-recognition,
    media-processing) uses `arn:aws:states:::lambda:invoke` exclusively, and
    that dialect carries the function ARN in `parameters.FunctionName` — not in
    `resource`, which holds the integration name. A synthetic history in the
    legacy `LambdaFunctionScheduled` dialect would exercise a branch no
    deployed state machine takes.
    """
    return _usage_event(
        event_id, previous_id, "TaskScheduled", seconds,
        taskScheduledEventDetails={
            "resource": "arn:aws:states:::lambda:invoke",
            "resourceType": "lambda",
            "parameters": json.dumps({"FunctionName": _USAGE_ARN,
                                      "Payload": {"operation": operation}}),
        })


def _usage_history_retry():
    """One state, attempt 1 FAILS after 100 s, attempt 2 SUCCEEDS after 50 s.

    Both attempts chain back to the SAME `TaskStateEntered` (id 2) — verified
    against real history: an ASL `Retry` does not re-enter the state. AWS bills
    both, so `billed_sec` must be 150.
    """
    return [
        _usage_event(1, 0, "ExecutionStarted", 0),
        _usage_event(2, 1, "TaskStateEntered", 0,
                     stateEnteredEventDetails={"name": "Work"}),
        _usage_scheduled(3, 2, 0, "StateWork"),
        _usage_event(4, 3, "TaskStarted", 1),
        _usage_event(5, 4, "TaskFailed", 100),
        _usage_scheduled(6, 5, 110, "StateWork"),
        _usage_event(7, 6, "TaskStarted", 111),
        _usage_event(8, 7, "TaskSucceeded", 160),
        _usage_event(9, 8, "TaskStateExited", 161,
                     stateExitedEventDetails={"name": "Work"}),
    ]


def _usage_history_concurrent():
    """Two CONCURRENT occurrences of one state, 50 s each, fully overlapping.

    Two separate `TaskStateEntered` events — which is what a Map iterator pair
    looks like, and the case `wall_sec` must NOT sum.
    """
    return [
        _usage_event(1, 0, "ExecutionStarted", 0),
        _usage_event(2, 1, "TaskStateEntered", 0,
                     stateEnteredEventDetails={"name": "Work"}),
        _usage_scheduled(3, 2, 0, "StateWork"),
        _usage_event(4, 1, "TaskStateEntered", 0,
                     stateEnteredEventDetails={"name": "Work"}),
        _usage_scheduled(5, 4, 0, "StateWork"),
        _usage_event(6, 3, "TaskSucceeded", 50),
        _usage_event(7, 5, "TaskSucceeded", 50),
        _usage_event(8, 6, "TaskStateExited", 50,
                     stateExitedEventDetails={"name": "Work"}),
        _usage_event(9, 7, "TaskStateExited", 50,
                     stateExitedEventDetails={"name": "Work"}),
    ]


def _usage_history_in_flight():
    """One attempt with NO terminal event — this state's own record.

    The only case where charging 0 s is right, and the case `incomplete` is
    now reserved for. Before the fix it also fired for every retry, so the
    flag could not distinguish "still running" from "billed but unreported".
    """
    return [
        _usage_event(1, 0, "ExecutionStarted", 0),
        _usage_event(2, 1, "TaskStateEntered", 0,
                     stateEnteredEventDetails={"name": "Work"}),
        _usage_scheduled(3, 2, 0, "StateWork"),
        _usage_event(4, 3, "TaskStarted", 1),
    ]


def _usage_scheduled_token(event_id, previous_id, seconds, operation):
    """A `TaskScheduled` for lambda:invoke.waitForTaskToken — same SDK dialect
    as _usage_scheduled, with the token suffix in `resource` (which is what
    face-recognition's _kind_of classifies on)."""
    return _usage_event(
        event_id, previous_id, "TaskScheduled", seconds,
        taskScheduledEventDetails={
            "resource": "arn:aws:states:::lambda:invoke.waitForTaskToken",
            "resourceType": "lambda",
            "parameters": json.dumps({"FunctionName": _USAGE_ARN,
                                      "Payload": {"operation": operation}}),
        })


def _usage_history_held_token():
    """A held waitForTaskToken state: the Lambda RETURNS after 2 s
    (TaskSubmitted) and the token completes 360 s later, when the sibling
    run it submitted finishes. Only the 2 s is Lambda compute — the wait is
    Step Functions holding a token, and the sibling's own published usage
    doc already bills its time (user ruling 2026-10-12; before the fix the
    whole 362 s was charged as parent compute ON TOP of the sibling's
    embedded bill). Event shape verified against a real speech-enhancement
    conversion-path history: Scheduled→Started→Submitted(+2 s)→
    Succeeded(+6 min)."""
    return [
        _usage_event(1, 0, "ExecutionStarted", 0),
        _usage_event(2, 1, "TaskStateEntered", 0,
                     stateEnteredEventDetails={"name": "Work"}),
        _usage_scheduled_token(3, 2, 0, "StateWork"),
        _usage_event(4, 3, "TaskStarted", 0),
        _usage_event(5, 4, "TaskSubmitted", 2),
        _usage_event(6, 5, "TaskSucceeded", 362),
        _usage_event(7, 6, "TaskStateExited", 362,
                     stateExitedEventDetails={"name": "Work"}),
    ]


class TestUsageTrackingBillsFailedAttempts(BundleTestCase):
    """Eight hand-copied history walkers must agree on what AWS bills.

    The bug: seven of the eight read only `succeeded_ts`, so a `TaskFailed` —
    a Lambda killed at its own timeout, the most expensive invocation a
    workflow can have — contributed 0 s. Measured 7.4x low on a real tts run
    ($0.031 reported against a $0.2318 bill), and it under-reported *worst*
    exactly when the workflow behaved worst. face-recognition had the inverse
    defect (state-span pricing, 2.17x high). See README.md §C10.

    Driven behaviourally rather than by grepping for the old gate, because
    each copy spells the walk differently (`_parse_execution` returning a
    list of summaries in seven, `_execution_cost` returning a dict in
    face-recognition) and only the arithmetic is common to all eight.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.usage = {stack: _load_usage_module(stack)
                     for stack in USAGE_TRACKING_STACKS}

    def summarize(self, stack, events):
        """One state's row, normalised across the THREE module shapes."""
        mod = self.usage[stack]
        if hasattr(mod, "_execution_cost"):
            # Two dialects of the same walk. Seven stacks and face-recognition
            # bind `_sfn` at import as a module-level CLIENT; bedrock-inference
            # binds `_SFN = None` and reaches it through a lazy `_sfn()`
            # accessor, because that stack's suite runs offline and a client
            # built at import time makes it pass or fail by the developer's
            # shell. Patch whichever this module actually has — assigning the
            # fake over the accessor is how this harness first failed on it,
            # with `'_UsageFakeSfn' object is not callable`.
            fake = _UsageFakeSfn(events)
            if callable(getattr(mod, "_sfn", None)):
                mod._SFN = fake
            else:
                mod._sfn = fake
            states, _wall = mod._execution_cost(_USAGE_EXEC_ARN)
            self.assertIn("Work", states, stack)
            row = dict(states["Work"])
        else:
            rows = [r for r in mod._parse_execution(events, {})
                    if r["state_name"] == "Work"]
            self.assertEqual(len(rows), 1, f"{stack}: expected one Work row")
            row = dict(rows[0])
            row["billed_sec"] = row.pop("total_billed_sec")
        return row

    def test_the_usage_tracking_stacks_are_exactly_these(self):
        """Guards the list above: a stack that gains a usage_tracking state
        without being added here is silently exempt from every assertion in
        this class — which is how the original bug reached eight copies."""
        found = sorted(
            name for name in os.listdir(DOCKER_ROOT)
            if os.path.isfile(os.path.join(
                DOCKER_ROOT, name, "states", "usage_tracking",
                "lambda_function.py")))
        self.assertEqual(
            found, sorted(USAGE_TRACKING_STACKS
                          + USAGE_AGGREGATOR_STACKS),
            "a usage_tracking state appeared or disappeared — update "
            "USAGE_TRACKING_STACKS (history billers: re-verify the "
            "cost arithmetic) or USAGE_AGGREGATOR_STACKS (sibling-doc "
            "aggregators: record the exemption)")

    def test_a_failed_attempt_is_billed(self):
        """THE regression. 150 s, not 50 — the failed attempt's 100 s is real
        money and used to vanish."""
        for stack in USAGE_TRACKING_STACKS:
            with self.subTest(stack=stack):
                row = self.summarize(stack, _usage_history_retry())
                self.assertEqual(row["invocations"], 2)
                self.assertAlmostEqual(row["billed_sec"], 150.0, places=3)

    def test_a_retried_attempt_is_counted_not_flagged_incomplete(self):
        """`retried` is the reader's explanation for billed > wall. Before the
        fix the only signal was `incomplete`, which said "this number is
        wrong" and offered no way to get the right one; it must now mean
        "no terminal event at all" and nothing else."""
        for stack in USAGE_TRACKING_STACKS:
            with self.subTest(stack=stack):
                row = self.summarize(stack, _usage_history_retry())
                self.assertEqual(row["retried"], 1)
                self.assertFalse(row["incomplete"])

    def test_sequential_attempts_are_summed_into_wall_time(self):
        """Retry attempts of ONE occurrence do not overlap, so wall time must
        cover both (150 s span-summed, or the 161 s enclosing state span) —
        never the longest single attempt, which is 100 s here."""
        for stack in USAGE_TRACKING_STACKS:
            with self.subTest(stack=stack):
                row = self.summarize(stack, _usage_history_retry())
                self.assertGreaterEqual(row["wall_sec"], 150.0)

    def test_concurrent_occurrences_are_not_summed_into_wall_time(self):
        """The other half of the same invariant, and the reason the fix could
        not simply sum every attempt: two Map iterators DO overlap, so the
        state's wall time is 50 s while its billed time is 100 s."""
        for stack in USAGE_TRACKING_STACKS:
            with self.subTest(stack=stack):
                row = self.summarize(stack, _usage_history_concurrent())
                self.assertEqual(row["invocations"], 2)
                self.assertAlmostEqual(row["billed_sec"], 100.0, places=3)
                self.assertAlmostEqual(row["wall_sec"], 50.0, places=3)

    def test_an_attempt_with_no_terminal_event_is_the_only_zero(self):
        """Charging 0 s is right here and nowhere else: usage_tracking's own
        Succeeded event has not been emitted while it is computing."""
        for stack in USAGE_TRACKING_STACKS:
            with self.subTest(stack=stack):
                row = self.summarize(stack, _usage_history_in_flight())
                self.assertEqual(row["invocations"], 1)
                self.assertEqual(row["retried"], 0)
                self.assertTrue(row["incomplete"])
                self.assertAlmostEqual(row["billed_sec"], 0.0, places=3)

    def test_a_held_task_token_wait_is_not_lambda_runtime(self):
        """A waitForTaskToken state's Lambda returns in seconds; the token
        completes when the SIBLING run does. Billing Scheduled→Succeeded
        charged the whole sibling run as parent compute on top of the
        sibling's own embedded bill. The span stacks bill to TaskSubmitted;
        the kind-model stacks (face-recognition, bedrock-inference) classify
        the state _KIND_TOKEN and bill 0 — both satisfy "the wait is not
        Lambda runtime". Wall time must still cover the real wait: pipeline
        wall is derived from it, and the pipeline really did wait."""
        for stack in USAGE_TRACKING_STACKS:
            with self.subTest(stack=stack):
                mod = self.usage[stack]
                row = self.summarize(stack, _usage_history_held_token())
                if hasattr(mod, "_execution_cost"):
                    self.assertAlmostEqual(row["billed_sec"], 0.0, places=3)
                else:
                    self.assertAlmostEqual(row["billed_sec"], 2.0, places=3)
                    self.assertGreaterEqual(row["wall_sec"], 360.0)

    def test_the_cost_covers_both_attempts_compute_and_requests(self):
        """Pins the arithmetic, not just the duration: a walker could bill the
        failed attempt's seconds and still drop its request charge (two of the
        eight had exactly that split before the fix)."""
        expected = (150.0 * (_USAGE_MEMORY_MB / 1024.0) * _USAGE_PER_GB_SEC
                    + 2 * _USAGE_PER_REQUEST)
        for stack in USAGE_TRACKING_STACKS:
            with self.subTest(stack=stack):
                row = self.summarize(stack, _usage_history_retry())
                self.assertEqual(row["memory_mb"], _USAGE_MEMORY_MB)
                # delta absorbs per-attempt vs per-state rounding:
                # face-recognition rounds each attempt to 6 places and sums.
                self.assertAlmostEqual(row["estimated_cost"], expected,
                                       delta=1e-6)


class TestOutputContract(BundleTestCase):
    """gateway_tool/output_contract.json — the field-level output contract each
    child serves via describe_tool and joins into outputs[] entries as `desc`
    (README.md §C4/§C6).

    Why this exists: "labels and output basenames are stable API" was
    [UNENFORCED across stacks] until 2026-08-13, and the field-level CONTENTS
    of the output files were documented nowhere machine-readable. A real
    consumer integrated against face-recognition's ephemeral/ intermediates
    (the only shapes it could find) and its face overlay silently degraded to
    plain video when they changed. The contract file is the boundary that
    makes internal-vs-API distinguishable; this class is what makes the
    boundary enforced rather than aspirational.

    The contract is COPIED per child like workflow_identity.py, but unlike it
    the copies are deliberately DIFFERENT per child — do not byte-pin them.
    What is pinned instead: the key set equals what the gateway emits, the
    file the gateway serves is the file on disk, and the file ships in the
    zip.
    """

    # child dir -> (GatewayTarget name, expected contract keys, style)
    #
    # style governs what the keys mean and whether a desc join exists:
    #   "labels"       — keys ARE the outputs[] labels; gateway joins desc by
    #                    label via _contract_desc.
    #   "format-kinds" — media-processing only: outputs[] labels are runtime
    #                    group names, so keys are the format kinds (+ fixed
    #                    probe/usage) and the join goes through the group's
    #                    reconciled format, never the file extension (a jpeg
    #                    group's output object is frame_manifest.json).
    #   "inline"       — no labeled file outputs; the contract documents
    #                    inline result shapes and there is no desc join.
    _CONTRACTS = {
        "cast-resolution-on-aws": (
            "cast-resolution",
            {"persons", "speaker-map", "face-map", "credits", "usage"},
            "labels"),
        "artwork-selection-on-aws": (
            "artwork-selection",
            {"artwork", "candidates", "frames"},
            "labels"),
        "shot-segmentation-on-aws": (
            "shot-segmentation",
            {"shot-list", "frame-embeddings", "shot-centroids",
             "recurrence", "usage", "debug-grids"},
            "labels"),
        "media-processing-on-aws": (
            "media-processing",
            # ad_mix added 2026-08-31 (the audio-description render:
            # bed ducked under the AD workflow's narration takes;
            # media-processing alpha.0831a); edit_manifest +
            # capture_manifest added 2026-10-09 (the quick_edit /
            # capture_frames tools — fixed-label evidence outputs)
            {"mp4", "wav", "m4a", "jpeg", "png", "metadata",
             "probe", "usage", "ad_mix",
             "edit_manifest", "capture_manifest"},
            "format-kinds"),
        "speech-diarization-on-aws": (
            "speech-diarization",
            {"transcript", "caption", "speaker-manifest",
             "global-speakers", "usage", "speaker-audio"},
            "labels"),
        "audiostem-separation-on-aws": (
            "audio-stem",
            # stem-activity + stem-waveforms added 2026-08-26
            # (program-structure-analysis inputs; audiostem alpha.20260826a);
            # speech-safe added 2026-10-19 (gated mix substitution —
            # audiostem README Design record §5)
            {"speech", "music", "sfx", "speech-safe", "metrics",
             "stem-activity", "stem-waveforms", "usage"},
            "labels"),
        "speech-enhancement-on-aws": (
            "speech-enhancement",
            {"enhanced", "metrics", "usage"},
            "labels"),
        "face-recognition-on-aws": (
            "face-recognition",
            {"face-metadata", "face-groups", "face-clusters",
             "usage", "people-images", "debug-grids"},
            "labels"),
        "tts-on-aws": (
            "text-to-speech",
            {"clips", "batch_qc_summary", "voice", "usage"},
            "inline"),
        "hybrid-search-on-aws": (
            "hybrid-search",
            {"search-results", "index-status"},
            "inline"),
        "bedrock-inference-on-aws": (
            "bedrock-inference",
            {"response", "summary", "usage"},
            "labels"),
        "prompt-registry-on-aws": (
            "prompt-registry",
            {"prompt-artifact", "put-result", "list-entry",
             "lifecycle-result"},
            "inline"),
        "conversation-analysis-on-aws": (
            "conversation-analysis",
            {"analysis", "chunk-manifest", "usage"},
            "labels"),
        "shot-metadata-enrichment-on-aws": (
            "shot-metadata-enrichment",
            {"shots", "scenes", "window-manifest", "usage"},
            "labels"),
        "frame-inspection-on-aws": (
            "frame-inspection",
            {"credits-read", "title-read", "claim-verify", "manifest",
             "usage"},
            "labels"),
        "program-structure-analysis-on-aws": (
            "program-structure-analysis",
            {"structure", "rundown", "manifest"},
            "labels"),
        "audio-description-on-aws": (
            "audio-description",
            {"srt", "vtt", "dapt", "review", "cue-ledger", "windows",
             "gaps", "render-manifest", "manifest", "usage"},
            "labels"),
    }

    _VALID_KINDS = {"file", "prefix", "inline"}

    # hybrid-search is the one contract child absent from
    # TestGatewayTokenContract.GATEWAYS (it takes no idempotency token), so its
    # import env lives here. The four os.environ[...] reads are what raise
    # without these.
    _EXTRA_ENV = {
        # prompt-registry is the second contract child absent from GATEWAYS:
        # no submit tool, no token anywhere (warn-and-ignore on every tool),
        # so the token-contract class must not import it. Its gateway reads
        # every env var with a default, so any non-empty dict works here.
        "prompt-registry-on-aws": {
            "TOOL_VERSION": "test.0",
        },
        "hybrid-search-on-aws": {
            "INDEX_STATE_MACHINE_ARN":
                "arn:aws:states:us-east-1:123456789012:stateMachine:hsi",
            "SEARCH_STATE_MACHINE_ARN":
                "arn:aws:states:us-east-1:123456789012:stateMachine:hss",
            "COLLECTION_TABLE": "collections",
            "INDEX_STATE_TABLE": "index-state",
        },
    }

    @staticmethod
    def _contract_path(child_dir):
        return os.path.join(DOCKER_ROOT, child_dir,
                            "gateway_tool", "output_contract.json")

    def _contract(self, child_dir):
        with open(self._contract_path(child_dir)) as f:
            return json.load(f)

    def _load_gateway(self, child_dir):
        """Import a child's gateway with the same discipline as
        TestGatewayTokenContract._load: this child's gateway_tool/ on sys.path,
        workflow_identity evicted around the import so each child gets its OWN
        copy, boto3 mocked because every gateway builds clients at import."""
        from unittest import mock
        base = self._EXTRA_ENV.get(
            child_dir) or TestGatewayTokenContract.GATEWAYS[child_dir][1]
        env = dict(base, PROXY_BUCKET="test-proxy", TARGET_NAME=child_dir,
                   TOOL_VERSION="test.0", AWS_DEFAULT_REGION="us-east-1")
        # path + module isolation is gateway_import_isolation's, once
        with gateway_import_isolation(child_dir) as gw_dir:
            spec = importlib.util.spec_from_file_location(
                f"bundle_contract_gw_{child_dir.replace('-', '_')}",
                os.path.join(gw_dir, "lambda_function.py"))
            gw = importlib.util.module_from_spec(spec)
            with mock.patch.dict(os.environ, env), \
                    mock.patch("boto3.client"), mock.patch("boto3.resource"):
                spec.loader.exec_module(gw)
        return gw

    def test_the_children_that_ship_a_contract_are_exactly_these(self):
        """Guards the set in both directions: a child that gains an outputs[]
        surface without a contract is undocumented API, and a contract for a
        removed child is a stale promise."""
        having = {d for d in CHILDREN.values()
                  if os.path.exists(self._contract_path(d))}
        self.assertEqual(
            having, set(self._CONTRACTS),
            "a child's output_contract.json appeared or disappeared — update "
            "_CONTRACTS and author/retire the contract deliberately")

    def test_contract_keys_are_exactly_the_declared_set(self):
        """Contract keys == emitted labels (or format kinds / inline shapes).
        Catches both an unadvertised new output and a stale contract entry.
        A new label is a deliberate API addition: update the gateway, the
        contract, AND this set together."""
        for child_dir in sorted(self._CONTRACTS):
            target, expected, _ = self._CONTRACTS[child_dir]
            with self.subTest(child=child_dir):
                doc = self._contract(child_dir)
                self.assertEqual(
                    set(doc["outputs"]), expected,
                    f"{child_dir}'s contract keys drifted from the declared "
                    "set — if the gateway's outputs changed, this is a "
                    "breaking API change: bump .version and update all three "
                    "places")
                self.assertEqual(
                    doc.get("target"), target,
                    f"{child_dir}'s contract target must equal its "
                    "GatewayTarget Name in media-toolkit-gateway/cfnstack.yaml")

    def test_output_labels_are_kebab_case(self):
        """Labels are HYPHEN-separated (kebab-case) — the naming convention
        (settled 2026-10-13): kebab-case names the public selector vocabulary
        (labels, target names, prompt categories, collection ids); snake_case
        names machine identifiers (parameters, JSON fields, basenames). The
        two underscore labels AD shipped with (cue_ledger/render_manifest)
        read like the parameter beside them (render_manifest_s3uri) and an
        agent pattern-matching from siblings guessed wrong — measured in the
        alpha.20261013a E2E. Exempt BY DISCRIMINATOR, never by key:
        'format-kinds' (media-processing — contract keys are format
        identifiers that double as input-enum values) and 'inline' (response
        FIELD shapes, snake by the field rule)."""
        grammar = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
        for child_dir in sorted(self._CONTRACTS):
            _, _, kind = self._CONTRACTS[child_dir]
            if kind != "labels":
                continue
            with self.subTest(child=child_dir):
                for label in self._contract(child_dir)["outputs"]:
                    self.assertRegex(
                        label, grammar,
                        f"{child_dir}: output label {label!r} is not "
                        "kebab-case — labels are hyphen-separated; "
                        "snake_case is for parameters/fields/basenames")

    def test_contract_envelope_invariants(self):
        """The output-contract envelope rules (README.md §C4). The one that
        bites: description must be a SINGLE LINE, because the gateway injects
        it verbatim into every outputs[] entry as `desc` — a multi-line
        description silently bloats every status response."""
        for child_dir in sorted(self._CONTRACTS):
            with self.subTest(child=child_dir):
                doc = self._contract(child_dir)
                self.assertEqual(doc.get("contract_version"), 1)
                self.assertTrue(str(doc.get("stability", "")).strip(),
                                f"{child_dir}: stability paragraph missing — "
                                "it is the contract-boundary statement")
                self.assertTrue(doc["outputs"],
                                f"{child_dir}: empty outputs")
                for key, entry in doc["outputs"].items():
                    desc = entry.get("description", "")
                    self.assertTrue(isinstance(desc, str) and desc.strip(),
                                    f"{child_dir}/{key}: missing description")
                    self.assertNotIn(
                        "\n", desc,
                        f"{child_dir}/{key}: description must be one line — "
                        "it is injected verbatim as desc; move detail to "
                        "notes")
                    kind = entry.get("kind")
                    self.assertIn(kind, self._VALID_KINDS,
                                  f"{child_dir}/{key}: kind {kind!r}")
                    if kind in ("file", "prefix"):
                        self.assertTrue(
                            str(entry.get("basename", "")).strip(),
                            f"{child_dir}/{key}: {kind} entries need a "
                            "basename (or pattern)")

    def test_gateway_serves_the_file_on_disk_and_joins_desc_from_it(self):
        """Runtime, not textual: the module-level _OUTPUT_CONTRACT must be the
        parsed file (the same object describe_tool returns), and for label
        children _contract_desc must answer from it — equality here is what
        guarantees the desc in outputs[] and the contract in describe_tool
        can never disagree, because they are one read of one file."""
        for child_dir in sorted(self._CONTRACTS):
            _, expected, style = self._CONTRACTS[child_dir]
            with self.subTest(child=child_dir):
                gw = self._load_gateway(child_dir)
                on_disk = self._contract(child_dir)
                self.assertEqual(
                    getattr(gw, "_OUTPUT_CONTRACT", None), on_disk,
                    f"{child_dir}: gateway's loaded contract is not the file "
                    "on disk")
                if style in ("labels", "format-kinds"):
                    for key in sorted(expected):
                        self.assertEqual(
                            gw._contract_desc(key),
                            on_disk["outputs"][key]["description"],
                            f"{child_dir}: desc join broken for {key!r}")
                    self.assertIsNone(
                        gw._contract_desc("no-such-label"),
                        f"{child_dir}: desc must be omitted, never "
                        "fabricated, for an unknown label")

    def test_the_contract_ships_in_the_gateway_zip(self):
        """Same trap as workflow_identity.py: a file that is not a zip member
        is not distributed, and `zip` exits 0 when only SOME members are
        missing. Parses the members=() array, not the script text — the
        header comments name the file, so a text search passes even when the
        member is deleted."""
        dir_to_logical = {d: name for name, d in CHILDREN.items()}
        for child_dir in sorted(self._CONTRACTS):
            with self.subTest(child=child_dir):
                build = self.child_build[dir_to_logical[child_dir]]
                match = re.search(r"members=\(([^)]*)\)", build)
                self.assertIsNotNone(
                    match,
                    f"{child_dir}'s build script has no gateway-zip "
                    "members=() array")
                self.assertIn(
                    "output_contract.json", match.group(1).split(),
                    f"{child_dir}'s gateway zip members lack "
                    "output_contract.json — describe_tool would serve no "
                    "contract and every outputs[] entry would lose its desc, "
                    "silently (the loader degrades by design)")
                # legacy inline guard OR the parameterized
                # verify_members guard (see TestWorkflowIdentityParity)
                self.assertTrue(
                    'echo "error: gateway_tool/$f not found"' in build
                    or ('echo "error: $dir/$f not found"' in build
                        and "verify_members gateway_tool" in build),
                    f"{child_dir}'s build script lists members but does not "
                    "verify each exists before zipping")


class TestRenderAndHistoryParity(BundleTestCase):
    """prompt-registry's `prompt_render.py` and `workflow_history.py` are
    COPIED into every llm-prompt-driven consumer, never imported — the same
    distribution mechanism as workflow_identity.py, with a sharper stake:
    prompt_render's output feeds bedrock-inference's workflow_id (the hash
    of RENDERED BYTES), so a drifted copy silently re-identifies every
    request of every consumer. Activated 2026-08-18 with the first real
    copies (conversation-analysis, per that repo's design record and the
    prompt-registry freeze rules).

    Discovery is by file existence under lib/ or gateway_tool/, and the set
    is pinned both ways — a consumer that gains or loses a copy fails here,
    not silently."""

    _MASTER_CHILD = "prompt-registry-on-aws"
    _MODULES = ("prompt_render.py", "workflow_history.py")
    # child dir -> subdirectory holding its copies
    _CONSUMERS = {
        "conversation-analysis-on-aws": "lib",
        "shot-metadata-enrichment-on-aws": "lib",
        "frame-inspection-on-aws": "lib",
        "audio-description-on-aws": "lib",
    }

    def _master(self, module):
        path = os.path.join(DOCKER_ROOT, self._MASTER_CHILD,
                            "gateway_tool", module)
        self.assertTrue(os.path.exists(path),
                        f"the MASTER copy vanished: {path}")
        return _read(path)

    def test_the_consumers_are_exactly_these(self):
        having = {}
        for child in CHILDREN.values():
            if child == self._MASTER_CHILD:
                continue
            for sub in ("lib", "gateway_tool"):
                if all(os.path.exists(os.path.join(
                        DOCKER_ROOT, child, sub, m)) for m in self._MODULES):
                    having[child] = sub
        self.assertEqual(
            having, self._CONSUMERS,
            "a child gained or lost the render/history copies — update "
            "_CONSUMERS and confirm byte-identity")

    def test_every_copy_is_byte_identical_to_its_master(self):
        for module in self._MODULES:
            master = self._master(module)
            for child, sub in sorted(self._CONSUMERS.items()):
                with self.subTest(module=module, child=child):
                    copy = _read(os.path.join(DOCKER_ROOT, child, sub,
                                              module))
                    self.assertEqual(
                        master, copy,
                        f"{child}'s {module} drifted from prompt-registry's "
                        "master — copy it, never edit it; a render change "
                        "re-identifies every consumer request and requires "
                        "every consumer's .version to bump (freeze rules in "
                        "prompt-registry-on-aws/README.md)")


class TestParsePartialOutputParity(BundleTestCase):
    """bedrock-inference's `parse_partial_output.py` — the model-output
    salvage ladder (README.md §PI6.2) — is COPIED into every
    llm-prompt consumer that parses JSONL output, never imported. The
    class §PI6.2 deferred "to the second copy" (conversation-analysis,
    2026-08-26; the first was shot-metadata-enrichment the same day).
    The stake: a drifted copy changes which model lines are recovered vs
    dropped, so two consumers would disagree about the same response
    bytes — and the module is stdlib-only BY DECISION (json-repair was
    measured fabricating a trimmed value on a mid-word cut), so a copy
    that grows an import has been edited, not copied.

    Same discovery discipline as TestRenderAndHistoryParity: file
    existence, set pinned both ways."""

    _MASTER_CHILD = "bedrock-inference-on-aws"
    _MODULE = "parse_partial_output.py"
    _CONSUMERS = {
        "conversation-analysis-on-aws": "lib",
        "shot-metadata-enrichment-on-aws": "lib",
        "frame-inspection-on-aws": "lib",
        "audio-description-on-aws": "lib",
    }

    def test_the_consumers_are_exactly_these(self):
        having = {}
        for child in CHILDREN.values():
            if child == self._MASTER_CHILD:
                continue
            for sub in ("lib", "gateway_tool"):
                if os.path.exists(os.path.join(DOCKER_ROOT, child, sub,
                                               self._MODULE)):
                    having[child] = sub
        self.assertEqual(
            having, self._CONSUMERS,
            "a child gained or lost the parse_partial_output copy — "
            "update _CONSUMERS and confirm byte-identity")

    def test_every_copy_is_byte_identical_to_its_master(self):
        master_path = os.path.join(DOCKER_ROOT, self._MASTER_CHILD,
                                   "lib", self._MODULE)
        self.assertTrue(os.path.exists(master_path),
                        f"the MASTER copy vanished: {master_path}")
        master = _read(master_path)
        for child, sub in sorted(self._CONSUMERS.items()):
            with self.subTest(child=child):
                copy = _read(os.path.join(DOCKER_ROOT, child, sub,
                                          self._MODULE))
                self.assertEqual(
                    master, copy,
                    f"{child}'s {self._MODULE} drifted from "
                    "bedrock-inference's master — copy it, never edit it; "
                    "a salvage change alters which lines every consumer "
                    "recovers and requires each consumer's .version to bump")


if __name__ == "__main__":
    unittest.main()
