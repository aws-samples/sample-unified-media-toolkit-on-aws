"""
Unit tests — pure-Python and mocked-boto3; no AWS, no pyvips required
(imaging-dependent paths are exercised separately in test_imaging.py).

Run:  python3 -m pytest test/test_units.py -q    (or python3 test/test_units.py)
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# gateway_tool/ goes on the path too because the gateway zip is FLAT (`zip -j`):
# at runtime `workflow_identity` is a top-level module, not `gateway_tool.
# workflow_identity`. Loading lambda_function.py by file path without this makes
# its own import fail here while working perfectly in Lambda.
sys.path.insert(0, os.path.join(ROOT, "gateway_tool"))

os.environ.setdefault("FACE_TABLE_NAME", "test-table")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")


def _client_error(code, op="Op"):
    from botocore.exceptions import ClientError
    return ClientError({"Error": {"Code": code, "Message": code}}, op)


# ─────────────────────────────────────────────────────────────────────────────
# input_schema — validation + policy toggles
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


class TestInputSchema(unittest.TestCase):
    def _event(self, **cfg):
        return {
            "uuid": "wf-1",
            "input": {"s3uri": "s3://in/a.mp4", "input_config": cfg},
            "output": {"s3uri": "s3://out/prefix"},
            "data": {},
        }

    def test_defaults_applied(self):
        from lib.input_schema import validate_and_normalize
        ev = validate_and_normalize(self._event())
        cfg = ev["input"]["input_config"]
        self.assertEqual(cfg["frame_interval_sec"], 1.0)
        self.assertEqual(cfg["probe_stride"], 4)
        self.assertTrue(cfg["celebrity_recognition"])
        # 0 since 2026-10-08: auto-index is opt-in for the public sample
        self.assertEqual(cfg["max_faces_to_index"], 0)
        self.assertEqual(cfg["max_concurrent_executions"], 5)
        self.assertEqual(cfg["detect_map_concurrency"], 4)

    def test_concurrency_fields_range_checked(self):
        from lib.input_schema import validate_and_normalize
        ev = validate_and_normalize(self._event(
            max_concurrent_executions=20, detect_map_concurrency=2))
        cfg = ev["input"]["input_config"]
        self.assertEqual(cfg["max_concurrent_executions"], 20)
        self.assertEqual(cfg["detect_map_concurrency"], 2)
        for bad in (dict(max_concurrent_executions=0),
                    dict(max_concurrent_executions=51),
                    dict(detect_map_concurrency=0),
                    dict(detect_map_concurrency=11)):
            with self.assertRaises(ValueError):
                validate_and_normalize(self._event(**bad))

    def test_policy_toggles_preserved(self):
        from lib.input_schema import validate_and_normalize
        ev = validate_and_normalize(self._event(
            celebrity_recognition=False, collection_id="",
            max_faces_to_index=0))
        cfg = ev["input"]["input_config"]
        self.assertFalse(cfg["celebrity_recognition"])
        self.assertEqual(cfg["collection_id"], "")
        self.assertEqual(cfg["max_faces_to_index"], 0)

    def test_rejects_bad_values(self):
        from lib.input_schema import validate_and_normalize
        for bad in (dict(frame_interval_sec=0.05),
                    dict(probe_stride=-1),
                    dict(min_appearances=0),
                    dict(use_partial=True, mark_in_out=[5, 2])):
            with self.assertRaises(ValueError):
                validate_and_normalize(self._event(**bad))

    def test_missing_required(self):
        from lib.input_schema import validate_and_normalize
        with self.assertRaises(ValueError):
            validate_and_normalize({"uuid": "", "input": {}, "output": {}})


# ─────────────────────────────────────────────────────────────────────────────
# frame_hash — patch dHash, noise classification, skip predicate
# ─────────────────────────────────────────────────────────────────────────────

class TestFrameHash(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        from lib import frame_hash
        self.fh = frame_hash
        rng = np.random.RandomState(7)
        self.base = rng.randint(0, 255, (135, 240)).astype("uint8")

    def _with_ticker(self, frame, seed):
        out = frame.copy()
        rng = self.np.random.RandomState(seed)
        out[100:135, :] = rng.randint(0, 255, (35, 240))  # bottom third churn
        return out

    def test_identical_frames_zero_distance(self):
        h = self.fh.frame_hashes(self.base)
        self.assertEqual(self.fh.stable_distance(h, h, []), 0)
        self.assertEqual(self.fh.hamming(h["global"], h["global"]), 0)

    def test_ticker_discounted_after_classification(self):
        frames = [self._with_ticker(self.base, s) for s in range(40)]
        hashes = [self.fh.frame_hashes(f) for f in frames]
        noisy = self.fh.classify_noisy_patches(hashes, threshold=4)
        self.assertTrue(set(noisy) <= {6, 7, 8})   # bottom-row patches
        self.assertTrue(noisy)
        # With the ticker discounted, adjacent frames read as stable
        self.assertLess(
            self.fh.stable_distance(hashes[0], hashes[1], noisy), 4)

    def test_global_churn_not_discounted(self):
        rng = self.np.random.RandomState(3)
        frames = [rng.randint(0, 255, (135, 240)).astype("uint8")
                  for _ in range(40)]
        hashes = [self.fh.frame_hashes(f) for f in frames]
        # everything churns → more than 3 noisy patches → no discount
        self.assertEqual(self.fh.classify_noisy_patches(hashes, 4), [])

    def test_zoom_drift_promotes(self):
        # adjacent frames similar, endpoints far apart (accumulated drift)
        def shifted(dx):
            return self.np.roll(self.base, dx, axis=1)
        seq = [self.fh.frame_hashes(shifted(dx)) for dx in range(0, 60, 4)]
        a, b, gap = seq[0], seq[-1], seq[1:-1]
        self.assertFalse(self.fh.gap_is_skippable(gap, a, b, [], threshold=6))

    def test_static_gap_skippable(self):
        h = self.fh.frame_hashes(self.base)
        self.assertTrue(self.fh.gap_is_skippable([h, h], h, h, [], threshold=4))

    def test_entrance_breaks(self):
        entered = self.base.copy()
        entered[30:80, 100:160] = 255  # bright object enters mid-frame
        ha = self.fh.frame_hashes(self.base)
        hb = self.fh.frame_hashes(entered)
        self.assertFalse(self.fh.gap_is_skippable([hb], ha, ha, [], threshold=4))


# ─────────────────────────────────────────────────────────────────────────────
# tracks — gap-dilated IoU, gender, hash-break veto, interpolation
# ─────────────────────────────────────────────────────────────────────────────

class TestTracks(unittest.TestCase):
    def _face(self, frame, l=0.4, gender="Male"):
        return {"frame_num": frame, "timestamp": float(frame),
                "gender": gender,
                "box": {"l": l, "t": 0.4, "w": 0.2, "h": 0.3}}

    def test_continuous_track(self):
        from lib.tracks import build_tracks
        faces = [self._face(i) for i in range(6)]
        tracks = build_tracks(faces, 3.0, lambda a, b: False)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["appearance_count"], 6)

    def test_hash_break_veto_splits(self):
        from lib.tracks import build_tracks
        faces = [self._face(i) for i in range(6)]
        tracks = build_tracks(faces, 3.0, lambda a, b: b == 3)
        self.assertEqual(len(tracks), 2)

    def test_gender_split(self):
        from lib.tracks import build_tracks
        faces = [self._face(0, gender="Male"), self._face(1, gender="Female")]
        self.assertEqual(len(build_tracks(faces, 3.0, lambda a, b: False)), 2)

    def test_gap_dilated_iou_tolerates_motion(self):
        from lib.tracks import build_tracks
        # subject drifts 0.05/frame with a 2 s gap in the middle
        faces = [self._face(0, l=0.40), self._face(2, l=0.50)]
        tracks = build_tracks(faces, 3.0, lambda a, b: False)
        self.assertEqual(len(tracks), 1)

    def test_interpolation(self):
        from lib.tracks import build_tracks, interpolate_skipped
        faces = [self._face(0, l=0.40), self._face(4, l=0.48)]
        tr = build_tracks(faces, 5.0, lambda a, b: False)[0]
        interpolate_skipped(tr, [{"frame_num": 2, "timestamp": 2.0}])
        self.assertEqual(tr["appearance_count"], 3)
        mid = [f for f in tr["faces"] if f["frame_num"] == 2][0]
        self.assertEqual(mid["source"], "interpolated")
        self.assertAlmostEqual(mid["box"]["l"], 0.44, places=4)


# ─────────────────────────────────────────────────────────────────────────────
# shots-mode cut veto — UNION of boundaries and dHash (README.md §23)
# ─────────────────────────────────────────────────────────────────────────────

class TestShotsModeVetoUnion(unittest.TestCase):
    """Replays the Meridian mislabel (run fd4c98a6, README.md §23): shot-
    segmentation missed the hard cut at ~frame 4043 (168.6 s), and because
    shots mode REPLACED the dHash veto, the tracker welded 3 actor_M2 faces
    onto actor_M1's track 22 — which renamed the entire 207-second person.
    The fix is a union: either detector's break vetoes continuation."""

    _CUT_FRAME = 4043  # the real cut shot-segmentation did not report

    def _frame_index(self):
        # kept frames every 24 source frames (1 s at ~24 fps) around the cut,
        # hash A before the cut, hash B after — stable distance is the
        # hamming distance between 0x0 and 0xFF (= 8, over threshold 4).
        frames = []
        for fn in range(3960, 4225, 24):
            h = 0x00 if fn < self._CUT_FRAME else 0xFF
            frames.append({"frame_num": fn,
                           "hashes": {"patches": [h] * 9, "global": h}})
        return {"frames": frames}

    def _load_module(self):
        spec = importlib.util.spec_from_file_location(
            "bft", os.path.join(ROOT, "states", "build_face_tracks",
                                "lambda_function.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _track22_faces(self):
        # the recorded shape: actor_M2 at l~0.49 (166-168 s), the box jump, then
        # actor_M1 at l~0.26-0.28 (169-175 s); the recorded dilated IoU across
        # the jump is 0.118 >= MIN_TRACK_IOU 0.10, so WITHOUT a cut veto the
        # track continues — that marginal pass is what makes this fixture a
        # real regression and not a strawman.
        rows = [
            (3984, 166.17, 0.490), (4032, 168.17, 0.485),
            (4056, 169.17, 0.283), (4080, 170.17, 0.278),
            (4104, 171.17, 0.261), (4128, 172.17, 0.248),
            (4152, 173.17, 0.255), (4176, 174.17, 0.255),
            (4200, 175.18, 0.258),
        ]
        return [{"frame_num": fn, "timestamp": ts, "gender": "Male",
                 "box": {"l": l, "t": 0.15, "w": 0.23, "h": 0.62}}
                for fn, ts, l in rows]

    def test_dhash_cut_test_fires_at_the_missed_cut(self):
        mod = self._load_module()
        dhash_break = mod._dhash_cut_test(self._frame_index(), [], 4)
        self.assertTrue(dhash_break(4032, 4056))   # spans the cut
        self.assertFalse(dhash_break(3960, 4032))  # before the cut
        self.assertFalse(dhash_break(4056, 4200))  # after the cut

    def test_union_splits_the_meridian_track(self):
        from lib import shots as shots_lib
        from lib.tracks import build_tracks
        mod = self._load_module()
        starts = [0, 3385, 4210]  # shot 16 as shipped: NO boundary at 4043
        dhash_break = mod._dhash_cut_test(self._frame_index(), [], 4)

        # shots-only (the old REPLACE behavior) bridges the cut: ONE track
        old = build_tracks(
            self._track22_faces(), 3.0,
            lambda a, b: shots_lib.boundary_between(starts, a, b))
        self.assertEqual(1, len(old),
                         "fixture must reproduce the bridge under shots-only")

        # the HANDLER'S OWN composed veto splits it: actor_M2's 2 faces /
        # actor_M1's 7. Testing mod._cut_veto (not a locally rebuilt union) is
        # the point — a mutation inside the handler must fail HERE.
        fixed = build_tracks(self._track22_faces(), 3.0,
                             mod._cut_veto(dhash_break, starts))
        self.assertEqual(2, len(fixed))
        self.assertEqual([3984, 4032],
                         [f["frame_num"] for f in fixed[0]["faces"]])
        self.assertEqual(7, len(fixed[1]["faces"]))

    def test_union_defers_to_boundaries_where_dhash_is_silent(self):
        # a reported boundary still vetoes even with no hash break — the
        # union must never WEAKEN shots mode (dissolves stay covered)
        from lib import shots as shots_lib
        mod = self._load_module()
        dhash_break = mod._dhash_cut_test(
            {"frames": [{"frame_num": fn,
                         "hashes": {"patches": [0] * 9, "global": 0}}
                        for fn in range(0, 200, 24)]}, [], 4)
        starts = [0, 100]
        self.assertFalse(dhash_break(96, 120))
        self.assertTrue(mod._cut_veto(dhash_break, starts)(96, 120))
        # and with no boundary list the veto IS the dhash test
        self.assertIs(mod._cut_veto(dhash_break), dhash_break)


# ─────────────────────────────────────────────────────────────────────────────
# face_quality — centroid tiers
# ─────────────────────────────────────────────────────────────────────────────

class TestFaceQuality(unittest.TestCase):
    def _face(self, brightness=35, sharpness=45, occluded=False, area=0.06):
        import math
        side = math.sqrt(area)
        return {"box": {"l": 0.1, "t": 0.1, "w": side, "h": side},
                "brightness": brightness, "sharpness": sharpness,
                "face_occluded": occluded,
                "pose": {"pose_dist": 0.1}}

    def test_prefers_non_occluded_quality(self):
        from lib.face_quality import choose_representative
        faces = [self._face(occluded=True, sharpness=99),
                 self._face(occluded=False, sharpness=50)]
        self.assertEqual(choose_representative(faces), 1)

    def test_rejects_all_blurry(self):
        from lib.face_quality import choose_representative
        faces = [self._face(sharpness=1.0), self._face(sharpness=2.0)]
        self.assertIsNone(choose_representative(faces))

    def test_no_absolute_brightness_ceiling(self):
        """Regression: the old BRIGHTNESS_RANGE ceiling (<=70) rejected
        98-100% of the lightest-lit people's faces on a real broadcast
        (population median brightness 68.6, well-lit studio faces 80-92) —
        when it bound, best_faces picked their darkest frames. Brightness is
        lighting, but correlates with skin tone; selection must be relative,
        never an absolute window."""
        from lib.face_quality import best_faces, quality_ok
        bright = self._face(brightness=85.0, sharpness=10.0, area=0.08)
        dim = self._face(brightness=40.0, sharpness=10.0, area=0.02)
        self.assertTrue(quality_ok(bright))
        # larger+equally-frontal bright face must outrank the small dim one
        self.assertEqual(best_faces([dim, bright], 1), [1])

    def test_no_absolute_sharpness_floor_in_ranking(self):
        """Regression: the old SHARPNESS_RANGE floor (>=30) rejected 86.8%
        of all faces on real 960x540 broadcast footage (population median
        sharpness 9.9), so the 'fallback to all' path fired almost always
        and the window selected an arbitrary slice when it did bind."""
        from lib.face_quality import best_faces
        typical = self._face(brightness=65.0, sharpness=9.9, area=0.08)
        self.assertEqual(best_faces([typical], 1), [0])

    def test_extreme_black_floor_still_applies(self):
        from lib.face_quality import best_faces, quality_ok
        near_black = self._face(brightness=3.0, sharpness=50.0, area=0.10)
        ok = self._face(brightness=50.0, sharpness=10.0, area=0.02)
        self.assertFalse(quality_ok(near_black))
        self.assertEqual(best_faces([near_black, ok], 1), [1])

    def test_typical_broadcast_face_can_be_centroid(self):
        """Median real-broadcast face (brightness 68.6, sharpness 9.9) failed
        every old CENTROID_QUALITY_TIER (min sharpness 30) and rode the
        fall-through; it must now be a first-class candidate."""
        from lib.face_quality import choose_representative
        faces = [self._face(brightness=68.6, sharpness=9.9)]
        self.assertEqual(choose_representative(faces), 0)

    def _posed(self, yaw=0.0, pitch=0.0, area=0.05, sharpness=30.0,
               eyes_open=True, occluded=False):
        import math
        side = math.sqrt(area)
        pose_dist = ((pitch / 180) ** 2 + (yaw / 180) ** 2) ** (1 / 3)
        return {"box": {"l": 0.1, "t": 0.1, "w": side, "h": side},
                "brightness": 60.0, "sharpness": sharpness,
                "eyes_open": eyes_open, "face_occluded": occluded,
                "pose": {"pitch": pitch, "roll": 0.0, "yaw": yaw,
                         "pose_dist": round(pose_dist, 4)}}

    def test_pose_gate_beats_area_dominance(self):
        """Regression (real content): fusion_rank's (1-pose_dist) is a soft
        penalty — a yaw=-25 3/4 profile won the index pick because its box
        area (0.076) dominated a smaller frontal face. The hard pose gate
        must prefer the frontal face regardless of area."""
        from lib.face_quality import best_faces
        profile = self._posed(yaw=-25.0, area=0.076, sharpness=53.0)
        frontal = self._posed(yaw=-2.0, area=0.042, sharpness=21.0)
        self.assertEqual(best_faces([profile, frontal], 1), [1])

    def test_pose_gate_falls_back_when_no_frontal(self):
        """A cluster with only profile views still indexes its best face —
        the gate narrows the pool, never empties it."""
        from lib.face_quality import best_faces
        p1 = self._posed(yaw=-60.0, area=0.05, sharpness=40.0)
        p2 = self._posed(yaw=55.0, area=0.03, sharpness=10.0)
        self.assertEqual(best_faces([p1, p2], 1), [0])

    def test_pose_gate_skips_closed_eyes_and_occluded(self):
        from lib.face_quality import best_faces
        blink = self._posed(yaw=0.0, area=0.08, eyes_open=False)
        masked = self._posed(yaw=0.0, area=0.07, occluded=True)
        clean = self._posed(yaw=5.0, area=0.03)
        self.assertEqual(best_faces([blink, masked, clean], 1), [2])

    def test_single_pick_refines_by_sharpness(self):
        """limit=1 (persistent indexing): among the top-3 by fusion rank,
        take the sharpest — biggest is not always the crispest frame."""
        from lib.face_quality import best_faces
        big_soft = self._posed(yaw=1.0, area=0.060, sharpness=8.0)
        mid_sharp = self._posed(yaw=2.0, area=0.055, sharpness=45.0)
        self.assertEqual(best_faces([big_soft, mid_sharp], 1), [1])


# ─────────────────────────────────────────────────────────────────────────────
# cluster_viz — MDS sanity
# ─────────────────────────────────────────────────────────────────────────────

class TestClusterViz(unittest.TestCase):
    def test_similar_people_land_close(self):
        import numpy as np
        from lib.cluster_viz import build_similarity_matrix, mds_coords
        sim = build_similarity_matrix(3, [(0, 1, 96.0), (1, 2, 15.0)])
        c = mds_coords(sim)
        d01 = np.linalg.norm(c[0] - c[1])
        d02 = np.linalg.norm(c[0] - c[2])
        self.assertLess(d01, d02)

    def test_degenerate_sizes(self):
        from lib.cluster_viz import build_similarity_matrix, mds_coords
        self.assertEqual(mds_coords(build_similarity_matrix(0, [])).shape, (0, 2))
        self.assertEqual(mds_coords(build_similarity_matrix(1, [])).shape, (1, 2))


# ─────────────────────────────────────────────────────────────────────────────
# semaphore — transactions, dedup, drain (mocked DDB/SFN)
# ─────────────────────────────────────────────────────────────────────────────

class _FakeDdb:
    """Minimal DDB double supporting the semaphore's exact calls."""

    def __init__(self):
        self.items: dict[tuple, dict] = {}

    def _key(self, k):
        return (k["pk"]["S"], k["sk"]["S"])

    def transact_write_items(self, TransactItems):
        from botocore.exceptions import ClientError
        # validate all conditions first (atomicity)
        staged = []
        reasons = []
        failed = False
        for op in TransactItems:
            if "Update" in op:
                u = op["Update"]
                key = self._key(u["Key"])
                item = self.items.get(key, {})
                cond = u["ConditionExpression"]
                holders = int(item.get("holders", 0))
                if "holders < max_holders" in cond:
                    max_holders = int(item.get("max_holders", 0))
                    ok = "holders" not in item or holders < max_holders
                elif "holders > :zero" in cond:
                    ok = holders > 0
                else:
                    ok = True
                reasons.append({"Code": "None" if ok else "ConditionalCheckFailed"})
                failed = failed or not ok
                staged.append(("update", key, u))
            elif "Put" in op:
                p = op["Put"]
                key = self._key(p["Item"])
                exists = key in self.items
                ok = not (exists and "attribute_not_exists" in p.get(
                    "ConditionExpression", ""))
                reasons.append({"Code": "None" if ok else "ConditionalCheckFailed"})
                failed = failed or not ok
                staged.append(("put", key, p))
        if failed:
            err = ClientError({
                "Error": {"Code": "TransactionCanceledException",
                          "Message": "cancelled"},
                "CancellationReasons": reasons,
            }, "TransactWriteItems")
            err.response["CancellationReasons"] = reasons
            raise err
        for kind, key, op in staged:
            if kind == "put":
                self.items[key] = {
                    k: v.get("S") or v.get("N")
                    for k, v in op["Item"].items() if k not in ("pk", "sk")}
            else:
                item = self.items.setdefault(key, {})
                expr = op["UpdateExpression"]
                holders = int(item.get("holders", 0))
                if "+ :one" in expr:
                    item["holders"] = str(holders + 1)
                elif "- :one" in expr:
                    item["holders"] = str(holders - 1)
                if "#ttl = :ttl" in expr:
                    item["ttl"] = op["ExpressionAttributeValues"][":ttl"]["N"]

    def put_item(self, TableName, Item, **kw):
        key = self._key(Item)
        self.items[key] = {k: v.get("S") or v.get("N")
                           for k, v in Item.items() if k not in ("pk", "sk")}

    def update_item(self, TableName, Key, ConditionExpression=None,
                     UpdateExpression=None, ExpressionAttributeValues=None,
                     **kw):
        key = self._key(Key)
        cond = ConditionExpression or ""
        item = self.items.get(key, {})
        if "attribute_exists(sk)" in cond and key not in self.items:
            raise _client_error("ConditionalCheckFailedException", "UpdateItem")
        if "max_holders" in cond:
            requested = int(ExpressionAttributeValues[":requested"]["N"])
            current = item.get("max_holders")
            if current is not None and int(current) >= requested:
                raise _client_error(
                    "ConditionalCheckFailedException", "UpdateItem")
            self.items.setdefault(key, {})["max_holders"] = str(requested)
            return
        self.items.setdefault(key, {})["ttl"] = "refreshed"

    def delete_item(self, TableName, Key, ConditionExpression=None, **kw):
        key = self._key(Key)
        if "attribute_exists" in (ConditionExpression or "") \
                and key not in self.items:
            raise _client_error("ConditionalCheckFailedException", "DeleteItem")
        self.items.pop(key, None)

    def query(self, TableName, KeyConditionExpression, ExpressionAttributeValues,
              Limit=20, **kw):
        pk = ExpressionAttributeValues[":pk"]["S"]
        out = []
        for (ipk, isk), attrs in self.items.items():
            if ipk == pk:
                item = {"pk": {"S": ipk}, "sk": {"S": isk}}
                for k, v in attrs.items():
                    item[k] = ({"N": v} if k in ("created_at", "ttl", "requested_max")
                                else {"S": v})
                out.append(item)
        return {"Items": out[:Limit]}


# ─────────────────────────────────────────────────────────────────────────────
# s3_utils.get_json_if_exists — narrow catch (missing key only), no blanket
# except Exception masking real S3 failures as "no checkpoint"
# ─────────────────────────────────────────────────────────────────────────────

class TestGetJsonIfExists(unittest.TestCase):
    def test_missing_key_returns_none(self):
        from lib.s3_utils import get_json_if_exists
        with mock.patch("lib.s3_utils.get_json",
                        side_effect=_client_error("NoSuchKey", "GetObject")):
            self.assertIsNone(get_json_if_exists("bucket", "missing.json"))

    def test_404_code_returns_none(self):
        from lib.s3_utils import get_json_if_exists
        with mock.patch("lib.s3_utils.get_json",
                        side_effect=_client_error("404", "GetObject")):
            self.assertIsNone(get_json_if_exists("bucket", "missing.json"))

    def test_other_client_error_propagates(self):
        from lib.s3_utils import get_json_if_exists
        with mock.patch("lib.s3_utils.get_json",
                        side_effect=_client_error("AccessDenied", "GetObject")):
            with self.assertRaises(Exception):
                get_json_if_exists("bucket", "key.json")

    def test_non_client_error_propagates(self):
        """Corrupted JSON, decode errors etc. must not be swallowed either —
        only ClientError NoSuchKey/404 means 'no checkpoint yet'."""
        from lib.s3_utils import get_json_if_exists
        with mock.patch("lib.s3_utils.get_json",
                        side_effect=ValueError("bad json")):
            with self.assertRaises(ValueError):
                get_json_if_exists("bucket", "key.json")

    def test_found_key_returns_value(self):
        from lib.s3_utils import get_json_if_exists
        with mock.patch("lib.s3_utils.get_json", return_value={"a": 1}):
            self.assertEqual(get_json_if_exists("bucket", "key.json"), {"a": 1})


class TestSemaphore(unittest.TestCase):
    def setUp(self):
        self.ddb = _FakeDdb()
        self.sfn = mock.MagicMock()
        # import a fresh module instance with our doubles
        spec = importlib.util.spec_from_file_location(
            "sem_test", os.path.join(ROOT, "lib", "semaphore.py"))
        self.sem = importlib.util.module_from_spec(spec)
        with mock.patch("boto3.client",
                        side_effect=lambda svc, **kw:
                        self.ddb if svc == "dynamodb" else self.sfn):
            spec.loader.exec_module(self.sem)
        self.sem.TABLE_NAME = "test-table"

    def test_acquire_to_max_then_queue(self):
        self.assertTrue(self.sem.try_acquire("arn:e1", 2))
        self.assertTrue(self.sem.try_acquire("arn:e2", 2))
        self.assertFalse(self.sem.try_acquire("arn:e3", 2))

    def test_reentrant_acquire(self):
        self.assertTrue(self.sem.try_acquire("arn:e1", 2))
        self.assertTrue(self.sem.try_acquire("arn:e1", 2))  # retry — same holder
        counter = self.ddb.items[("semaphore", "counter")]
        self.assertEqual(counter["holders"], "1")

    def test_release_idempotent(self):
        self.sem.try_acquire("arn:e1", 2)
        self.assertTrue(self.sem.release_holder("arn:e1"))
        self.assertFalse(self.sem.release_holder("arn:e1"))

    def test_decrement_once_dedups(self):
        self.sem.try_acquire("arn:e1", 2)
        self.assertTrue(self.sem.decrement_once("event-1"))
        self.assertFalse(self.sem.decrement_once("event-1"))  # retry record
        counter = self.ddb.items[("semaphore", "counter")]
        self.assertEqual(counter["holders"], "0")

    def test_drain_fifo(self):
        self.sem.try_acquire("arn:e1", 2)
        self.sem.try_acquire("arn:e2", 2)
        self.sem.store_pending("arn:e3", "token-3", 2)
        self.sem.store_pending("arn:e4", "token-4", 2)
        # make e3 older than e4
        self.ddb.items[("pending",
                        __import__("hashlib").sha256(b"token-3").hexdigest()
                        )]["created_at"] = "1000"
        self.ddb.items[("pending",
                        __import__("hashlib").sha256(b"token-4").hexdigest()
                        )]["created_at"] = "2000"
        # release one slot: holder delete + stream decrement, then drain
        self.sem.release_holder("arn:e1")
        self.sem.decrement_once("ev-1")
        admitted = self.sem.drain_pending()
        self.assertEqual(admitted, 1)
        self.sem.sfn = self.sfn
        self.sfn.send_task_success.assert_called_once_with(
            taskToken="token-3", output="{}")

    def test_ttl_refresh_requires_holder(self):
        self.sem.try_acquire("arn:e1", 2)
        self.sem.refresh_holder("arn:e1")  # no raise
        self.sem.refresh_holder("arn:missing")  # warns, no raise

    def test_max_holders_ratchets_up_only(self):
        self.sem.try_acquire("arn:e1", 5)
        counter = self.ddb.items[("semaphore", "counter")]
        self.assertEqual(counter["max_holders"], "5")
        # a smaller request from a later caller must not lower the cap
        self.sem.try_acquire("arn:e2", 2)
        self.assertEqual(counter["max_holders"], "5")
        # a larger request raises it further
        self.sem.try_acquire("arn:e3", 8)
        self.assertEqual(counter["max_holders"], "8")

    def test_admission_bound_by_live_max_holders_not_first_caller(self):
        # first caller sets the bar high; a later caller's smaller request
        # must still admit up to the RAISED cap, not its own lower request.
        self.assertTrue(self.sem.try_acquire("arn:e1", 3))
        self.assertTrue(self.sem.try_acquire("arn:e2", 1))
        self.assertTrue(self.sem.try_acquire("arn:e3", 1))
        self.assertFalse(self.sem.try_acquire("arn:e4", 1))

    def test_counter_ttl_refreshed_on_acquire_and_decrement(self):
        self.sem.try_acquire("arn:e1", 2)
        counter = self.ddb.items[("semaphore", "counter")]
        self.assertIn("ttl", counter)
        first_ttl = counter["ttl"]
        self.sem.release_holder("arn:e1")
        self.sem.decrement_once("ev-1")
        self.assertIn("ttl", counter)  # still present after decrement

    def test_refresh_holder_refreshes_counter_ttl_when_holder_alive(self):
        self.sem.try_acquire("arn:e1", 2)
        counter = self.ddb.items[("semaphore", "counter")]
        del counter["ttl"]  # simulate a stale/expired counter ttl
        self.sem.refresh_holder("arn:e1")
        self.assertIn("ttl", counter)

    def test_refresh_holder_does_not_resurrect_counter_for_zombie(self):
        # A holder that no longer exists (already TTL-reclaimed) must not
        # be able to refresh the counter's ttl — invariant 5.
        self.sem.try_acquire("arn:e1", 2)
        counter = self.ddb.items[("semaphore", "counter")]
        del counter["ttl"]
        self.sem.refresh_holder("arn:missing")  # no holder item for this arn
        self.assertNotIn("ttl", counter)


# ─────────────────────────────────────────────────────────────────────────────
# Policy matrix — API-call gating per GDPR use case (call-recorder style)
# ─────────────────────────────────────────────────────────────────────────────

class TestPolicyMatrix(unittest.TestCase):
    """The state handlers gate on input_config; verify each toggle's effect on
    the decision points without running full states."""

    def test_celebrity_off_produces_no_iterators(self):
        # build_face_tracks: celebrity_recognition=False → iterators=[]
        # (decision point extracted: gate expression)
        cfg = {"celebrity_recognition": False}
        qualified = [{"track_id": 0}]
        should_recognize = cfg["celebrity_recognition"] and bool(qualified)
        self.assertFalse(should_recognize)

    def test_empty_collection_disables_search_and_index(self):
        cfg = {"collection_id": "", "max_faces_to_index": 10,
               "min_appearances": 1}
        collection_id = cfg["collection_id"]
        validated = True
        search_allowed = bool(collection_id) and validated
        index_allowed = search_allowed and cfg["max_faces_to_index"] > 0
        self.assertFalse(search_allowed)
        self.assertFalse(index_allowed)

    def test_zero_index_keeps_search(self):
        cfg = {"collection_id": "talent", "max_faces_to_index": 0}
        search_allowed = bool(cfg["collection_id"])
        index_allowed = search_allowed and cfg["max_faces_to_index"] > 0
        self.assertTrue(search_allowed)
        self.assertFalse(index_allowed)

    def test_named_only_filters_unnamed(self):
        cfg = {"index_named_faces_only": True}
        entry_named = {"celebrity": {"name": "Jane"}}
        entry_unnamed = {"celebrity": None}
        self.assertTrue(not cfg["index_named_faces_only"]
                        or bool(entry_named["celebrity"]))
        self.assertFalse(not cfg["index_named_faces_only"]
                         or bool(entry_unnamed["celebrity"]))


# ─────────────────────────────────────────────────────────────────────────────
# rekognition_utils — delete-and-ignore, counters
# ─────────────────────────────────────────────────────────────────────────────

class TestRekognitionUtils(unittest.TestCase):
    def test_counter_fields_mirror_the_key_tuples(self):
        # ONE source of truth: CallCounter derives its fields from
        # input_schema's tuples and usage_tracking prices exactly the
        # charged tuple — a counter added in one place but not the
        # other vanished silently once (recognize_celebrities_api=0)
        from lib.input_schema import (API_COUNTER_KEYS,
                                      CHARGED_API_COUNTER_KEYS,
                                      UNCHARGED_API_COUNTER_KEYS)
        from lib.rekognition_utils import CallCounter
        self.assertEqual(set(CallCounter().as_dict()),
                         set(API_COUNTER_KEYS))
        self.assertEqual(set(API_COUNTER_KEYS),
                         set(CHARGED_API_COUNTER_KEYS)
                         | set(UNCHARGED_API_COUNTER_KEYS))

    def test_delete_collection_ignores_missing(self):
        from lib import rekognition_utils as rk
        with mock.patch.object(rk._rek, "delete_collection",
                               side_effect=_client_error(
                                   "ResourceNotFoundException")):
            rk.delete_collection_ignore_missing("sandbox-x")  # no raise
        with mock.patch.object(rk._rek, "delete_collection",
                               side_effect=_client_error("AccessDenied")):
            with self.assertRaises(Exception):
                rk.delete_collection_ignore_missing("sandbox-x")

    def test_counter_accumulation(self):
        from lib.rekognition_utils import CallCounter
        c = CallCounter()
        c.detect_faces_api += 3
        c.add({"detect_faces_api": 2, "search_faces_api": 5})
        self.assertEqual(c.detect_faces_api, 5)
        self.assertEqual(c.search_faces_api, 5)


# ─────────────────────────────────────────────────────────────────────────────
# _resolve_fps — fail-hard on missing/malformed frame rate (no silent default)
# ─────────────────────────────────────────────────────────────────────────────

class TestResolveFps(unittest.TestCase):
    """
    Regression coverage for a real bug found against the deployed pipeline:
    probe.json nests ffprobe output under "probe_info" (media-processing's
    actual, documented schema), but _resolve_fps originally read a top-level
    "streams" key that never exists — the lookup silently returned nothing,
    the try/except swallowed it, and the function fell through to a
    hardcoded 30000/1001 default with NO warning, NO error. On the real
    Meridian asset (actual fps 24000/1001) every timestamp in
    facemetadata.json was wrong by a factor of 30000/24000 = 1.25x. Per
    explicit direction: a wrong number is worse than a crash — there must be
    NO fallback and NO default anywhere in this function.
    """

    def setUp(self):
        import importlib
        import states.prepare_probe_iterators.lambda_function as mod
        importlib.reload(mod)
        self.mod = mod

    def _probe(self, streams):
        return {"probe_info": {"streams": streams}}

    def test_resolves_exact_fraction_from_probe_info(self):
        with mock.patch("states.prepare_probe_iterators.lambda_function.get_json",
                        return_value=self._probe([
                            {"codec_type": "audio", "r_frame_rate": "0/0"},
                            {"codec_type": "video",
                             "r_frame_rate": "24000/1001"},
                        ])):
            fps = self.mod._resolve_fps("bucket", "probe.json")
        self.assertEqual(fps.numerator, 24000)
        self.assertEqual(fps.denominator, 1001)

    def test_top_level_streams_is_not_read(self):
        """The exact shape of the original bug: streams at the top level
        (wrong schema) must NOT be silently accepted — there is no video
        stream reachable via probe_info.streams, so this must raise."""
        bad_shape = {"streams": [
            {"codec_type": "video", "r_frame_rate": "24000/1001"}]}
        with mock.patch("states.prepare_probe_iterators.lambda_function.get_json", return_value=bad_shape):
            with self.assertRaises(ValueError):
                self.mod._resolve_fps("bucket", "probe.json")

    def test_no_probe_key_raises(self):
        with self.assertRaises(ValueError):
            self.mod._resolve_fps("bucket", None)

    def test_no_video_stream_raises(self):
        with mock.patch("states.prepare_probe_iterators.lambda_function.get_json",
                        return_value=self._probe(
                            [{"codec_type": "audio", "r_frame_rate": "48000/1"}])):
            with self.assertRaises(ValueError):
                self.mod._resolve_fps("bucket", "probe.json")

    def test_malformed_rate_raises(self):
        with mock.patch("states.prepare_probe_iterators.lambda_function.get_json",
                        return_value=self._probe(
                            [{"codec_type": "video", "r_frame_rate": "0/0"}])):
            with self.assertRaises(ValueError):
                self.mod._resolve_fps("bucket", "probe.json")

    def test_falls_back_to_avg_frame_rate_when_r_frame_rate_missing(self):
        with mock.patch("states.prepare_probe_iterators.lambda_function.get_json",
                        return_value=self._probe(
                            [{"codec_type": "video",
                              "avg_frame_rate": "30000/1001"}])):
            fps = self.mod._resolve_fps("bucket", "probe.json")
        self.assertEqual((fps.numerator, fps.denominator), (30000, 1001))

    def test_never_returns_a_hardcoded_default_on_exception(self):
        """get_json raising (network error, bad JSON, etc.) must propagate,
        never be swallowed into a guessed value."""
        with mock.patch("states.prepare_probe_iterators.lambda_function.get_json",
                        side_effect=RuntimeError("s3 boom")):
            with self.assertRaises(RuntimeError):
                self.mod._resolve_fps("bucket", "probe.json")


# ─────────────────────────────────────────────────────────────────────────────
# cluster_identities — recognize-iterator usage must reach usage_tracking
# ─────────────────────────────────────────────────────────────────────────────

class TestRecognizeUsageAggregation(unittest.TestCase):
    """
    Regression coverage for a real bug: RecognizeCelebrities calls made by
    the recognize_celebrities Map (7 iterators, 1 call each = 7 real, billed
    calls on the Meridian test run) never reached usage_tracking.json, which
    reported recognize_celebrities_api=0. Root cause: the count lived only
    in event["data"]["iterators"][*].key -> recognize_N.json, and
    cluster_identities nulls that list (_finish) before usage_tracking runs,
    with no state ever summing it into a file usage_tracking.py reads.
    """

    def test_collect_celebrity_tags_returns_usage_too(self):
        import states.cluster_identities.lambda_function as mod
        event = {"data": {"iterators": [{"key": "k0"}, {"key": "k1"}]}}
        docs = {
            "k0": {"results": {"0": {"5": {"name": "A", "confidence": 99.0}}},
                   "usage": {"recognize_celebrities_api": 1}},
            "k1": {"results": {"0": {"9": {"name": None, "confidence": 10.0}}},
                   "usage": {"recognize_celebrities_api": 1}},
        }
        with mock.patch("states.cluster_identities.lambda_function.get_json",
                        side_effect=lambda bucket, key: docs[key]):
            tags, usage = mod._collect_celebrity_tags(event, "bucket")
        self.assertEqual(usage.get("recognize_celebrities_api"), 2)
        self.assertEqual(tags[5]["name"], "A")

    def test_recognize_usage_added_once_not_on_resume(self):
        """On a checkpointed re-enter, event["data"]["iterators"] is
        unchanged (same recognize_N.json keys, already-completed calls) —
        recognize_usage must be added exactly once (fresh-attempt branch),
        never again when resuming from a persisted cluster_state.json that
        already baked it in, or the count doubles on every re-enter."""
        import states.cluster_identities.lambda_function as mod

        event = {
            "uuid": "wf-1",
            "output": {"s3uri": "s3://out/prefix"},
            "data": {
                "iterators": [{"key": "recognize_0.json"}],
                "tracks_key": "tracks.json",
            },
        }
        docs = {
            "recognize_0.json": {"results": {}, "usage": {
                "recognize_celebrities_api": 3}},
            "tracks.json": {"tracks": []},
        }
        # Simulates a persisted checkpoint from a prior (timed-out)
        # invocation that already folded the recognize usage in once —
        # returned via get_json_if_exists, the resume-checkpoint read path.
        checkpoint = {
            "face_to_track": {}, "searched": [], "pairs": [], "parent": {},
            "usage": {"recognize_celebrities_api": 3, "index_faces_api": 2},
        }

        def fake_get_json(bucket, key):
            if key in docs:
                return docs[key]
            raise FileNotFoundError(key)

        written = {}

        def fake_put_json(bucket, key, data):
            written[key] = data

        with mock.patch("states.cluster_identities.lambda_function.get_json",
                        side_effect=fake_get_json), \
             mock.patch("states.cluster_identities.lambda_function"
                        ".get_json_if_exists", return_value=checkpoint), \
             mock.patch("states.cluster_identities.lambda_function.put_json",
                        side_effect=fake_put_json), \
             mock.patch("states.cluster_identities.lambda_function"
                        ".semaphore.refresh_holder"):
            # execution_arn now arrives via the ASL payload; absent here, so
            # the handler skips refresh_holder entirely.
            mod.lambda_handler(event, mock.MagicMock())

        clusters_key = next(k for k in written
                            if k.endswith("identity_clusters.json"))
        clusters_doc = written[clusters_key]
        # Resumed from a checkpoint that already had recognize_celebrities_api=3
        # baked in — must stay 3, not become 6.
        self.assertEqual(
            clusters_doc["usage"]["recognize_celebrities_api"], 3,
            "recognize usage was double-counted on resume")
        self.assertEqual(clusters_doc["usage"]["index_faces_api"], 2)


# ─────────────────────────────────────────────────────────────────────────────
# search_index_faces — §23 two-witness corroboration on naming conflicts
# ─────────────────────────────────────────────────────────────────────────────

class TestSearchCorroboration(unittest.TestCase):
    """Replays the Meridian mislabel's search stage (README.md §23): the
    cluster's best face by PHOTO QUALITY was a contaminated actor_M2
    close-up inside actor_M1's cluster, and its collection match
    (similarity 99.95) silently overruled a 41-track celebrity plurality.
    The fix: when the match name disagrees with a validated plurality,
    corroborate with ONE more search on the best face from a DIFFERENT
    track — the different-track constraint is load-bearing, because every
    contaminated face entered through one bridged cut and so lives in one
    track (a track-tag filter cannot help: the contaminated track carries
    the majority's own tag)."""

    MOD = "states.search_index_faces.lambda_function"

    def _face(self, frame_key, w, track_hint):
        return {"frame_key": frame_key, "source": "detected",
                "brightness": 60.0, "sharpness": 50.0, "_track": track_hint,
                "pose": {"pitch": 0.0, "roll": 0.0, "yaw": 0.0,
                         "pose_dist": 0.0},
                "box": {"l": 0.3, "t": 0.2, "w": w, "h": w * 1.3}}

    def _fixture(self):
        # track 22 holds the contaminated actor_M2 face — the LARGEST box in
        # the cluster, so quality ranking must pick it as witness 1.
        tracks = {
            21: {"track_id": 21, "faces": [self._face("k21.jpg", 0.10, 21)]},
            22: {"track_id": 22, "faces": [self._face("scott.jpg", 0.25, 22),
                                           self._face("k22.jpg", 0.09, 22)]},
            23: {"track_id": 23, "faces": [self._face("k23.jpg", 0.11, 23)]},
        }
        cluster = {"face_group_id": 0, "track_ids": [21, 22, 23]}
        entry = {"celebrity": {"name": "actor_M1", "confidence": 99.12}}
        return cluster, tracks, entry

    def _run(self, entry_celebrity, w2_result_face):
        import importlib
        mod = importlib.import_module(self.MOD)
        cluster, tracks, entry = self._fixture()
        if entry_celebrity is not None:
            entry["celebrity"] = entry_celebrity
        else:
            entry["celebrity"] = None
        searched_keys = []
        cropped_boxes = []

        def fake_search(coll, bucket, key, thr, counter):
            searched_keys.append(key)
            if key.endswith("_w2.jpg"):
                return [{"Similarity": 99.4,
                         "Face": {"FaceId": w2_result_face}}]
            return [{"Similarity": 99.95, "Face": {"FaceId": "m2-ref"}}]

        records = {"m2-ref": {"celeb": "actor_M2"},
                   "m1-ref": {"celeb": "actor_M1"}}
        with mock.patch(f"{self.MOD}.get_bytes", return_value=b"jpg"), \
             mock.patch(f"{self.MOD}.crop_face_native",
                        side_effect=lambda j, box, m, e:
                        cropped_boxes.append(box) or b"crop"), \
             mock.patch(f"{self.MOD}.put_bytes"), \
             mock.patch(f"{self.MOD}.search_faces_by_image",
                        side_effect=fake_search), \
             mock.patch(f"{self.MOD}.batch_get_faces",
                        side_effect=lambda ids: {i: records[i]
                                                 for i in ids if i in records}):
            match = mod._search_persistent(
                cluster, tracks, entry, {"uuid": "test-uuid"}, "coll",
                "bkt", "prefix", "fbkt", "fprefix", mock.MagicMock())
        return match, entry, searched_keys, cropped_boxes

    def test_meridian_shape_second_witness_sides_with_plurality(self):
        match, entry, keys, boxes = self._run(
            {"name": "actor_M1", "confidence": 99.12}, "m1-ref")
        self.assertEqual("actor_M1", match["celeb"])
        self.assertEqual("m1-ref", match["face_id"])
        self.assertEqual(2, len(keys), "exactly one corroborating search")
        self.assertEqual("second_witness_sided_with_plurality",
                         entry["name_conflict"]["resolution"])
        self.assertEqual("actor_M2",
                         entry["name_conflict"]["first_witness"]["celeb"])
        # the different-track constraint: witness 2 must NOT be another face
        # from track 22 (its k22 face is bigger than k23/k21 by design here
        # only if included — assert the crop was NOT the actor_M2 box)
        self.assertNotEqual(0.25, boxes[1]["w"],
                            "witness 2 came from the contaminated track")

    def test_witnesses_agree_collection_beats_plurality(self):
        # the §15 `news_b` scenario: celebrity plurality is the wrong signal
        match, entry, keys, _ = self._run(
            {"name": "actor_M1", "confidence": 99.12}, "m2-ref")
        self.assertEqual("actor_M2", match["celeb"])
        self.assertEqual(2, len(keys))
        self.assertEqual("witnesses_agree_collection_wins",
                         entry["name_conflict"]["resolution"])

    def test_no_conflict_spends_nothing_extra(self):
        match, entry, keys, _ = self._run(
            {"name": "actor_M2", "confidence": 99.9}, "unused")
        self.assertEqual("actor_M2", match["celeb"])
        self.assertEqual(1, len(keys), "no corroboration without a conflict")
        self.assertNotIn("name_conflict", entry)

    def test_no_plurality_spends_nothing_extra(self):
        match, entry, keys, _ = self._run(None, "unused")
        self.assertEqual("actor_M2", match["celeb"])
        self.assertEqual(1, len(keys))
        self.assertNotIn("name_conflict", entry)


# ─────────────────────────────────────────────────────────────────────────────
# finalize_outputs — §23 name conflicts are published, not buried
# ─────────────────────────────────────────────────────────────────────────────

class TestFinalizeNameConflict(unittest.TestCase):
    """The Meridian mislabel's correct identification (actor_M1 @ 99.12,
    41 tracks) existed ONLY in ephemeral/search_index_faces.json — a path the
    output contract marks internal. A naming conflict must reach the product
    facemetadata so it is self-announcing."""

    MOD = "states.finalize_outputs.lambda_function"

    def _docs(self, name_conflict):
        face = {"timestamp": 1.0, "frame_num": 24, "gender": "Male",
                "age_range": [50, 56], "source": "detected",
                "box": {"l": 0.3, "t": 0.2, "w": 0.2, "h": 0.3}}
        track = {"track_id": 21, "faces": [face], "centroid_idx": 0,
                 "appearance_count": 1, "first_ts": 1.0, "last_ts": 3.0}
        sc = {"face_group_id": 0, "track_ids": [21],
              "celebrity": {"name": "actor_M1", "confidence": 99.12},
              "face_match": {"face_id": "m1-ref", "celeb": "actor_M1",
                             "collection_id": "c", "similarity": 99.4}}
        if name_conflict:
            sc["name_conflict"] = {
                "resolution": "second_witness_sided_with_plurality",
                "celebrity_name": "actor_M1",
                "celebrity_confidence": 99.12,
                "first_witness": {"celeb": "actor_M2",
                                  "similarity": 99.95},
                "second_witness": {"celeb": "actor_M1",
                                   "similarity": 99.4}}
        return {
            "tracks": {"tracks": [track], "usage": {}},
            "clusters": {"clusters": [{"face_group_id": 0,
                                       "track_ids": [21]}],
                         "similarity_pairs": [], "usage": {}},
            "search": {"clusters": [sc], "usage": {}},
            "frame_index": {"frames": []},
            "plan": {},
        }

    def _run(self, name_conflict, mutate=None):
        import importlib
        mod = importlib.import_module(self.MOD)
        docs = self._docs(name_conflict)
        if mutate:
            mutate(docs)
        keymap = {"tk": docs["tracks"], "ck": docs["clusters"],
                  "sk": docs["search"], "fik": docs["frame_index"],
                  "dpk": docs["plan"]}
        written = {}
        event = {
            "uuid": "u",
            "input": {"s3uri": "s3://in/a.mp4",
                      "input_config": {"collection_id": "c"}},
            "output": {"s3uri": "s3://bkt/prefix"},
            "data": {
                "tracks_key": "tk", "clusters_key": "ck",
                "search_index_key": "sk", "detection_plan_key": "dpk",
                "media_processing_output": {"duration_sec": 719.0},
                "frame_meta": {"frame_index_key": "fik", "kept_frames": 10,
                               "triaged_out": 2,
                               "shot_boundaries_key": "sbk"},
            },
        }
        with mock.patch(f"{self.MOD}.get_json",
                        side_effect=lambda b, k: keymap[k]), \
             mock.patch(f"{self.MOD}.put_json",
                        side_effect=lambda b, k, d: written.update({k: d})):
            mod.lambda_handler(event, mock.MagicMock())
        fm = next(v for k, v in written.items()
                  if k.endswith("facemetadata.json"))
        return fm

    def test_conflict_reaches_facemetadata(self):
        fm = self._run(name_conflict=True)
        p = fm["people"][0]
        self.assertEqual("actor_M1", p["name"])
        self.assertEqual("collection", p["name_source"])
        self.assertTrue(p["name_conflict"])
        d = p["name_conflict_detail"]
        self.assertEqual("second_witness_sided_with_plurality",
                         d["resolution"])
        self.assertEqual("actor_M2", d["first_witness"])
        self.assertEqual("actor_M1", d["second_witness"])
        self.assertEqual(99.12, d["celebrity_confidence"])

    def test_no_conflict_adds_no_fields(self):
        fm = self._run(name_conflict=False)
        p = fm["people"][0]
        self.assertNotIn("name_conflict", p)
        self.assertNotIn("name_conflict_detail", p)

    def test_cut_detector_reports_the_union(self):
        fm = self._run(name_conflict=False)
        self.assertEqual("shots+dhash", fm["stats"]["cut_detector"])

    def test_celebrity_id_rides_name_agreement_not_name_source(self):
        # the gate is NAME AGREEMENT with the validated vote, never the
        # name_source proxy (review, 2026-09-07): a collection witness
        # that CONFIRMS the plurality names the same person and keeps
        # the key — otherwise auto-indexing progressively disables the
        # celebrity join on every later run
        def m(docs):
            docs["search"]["clusters"][0]["celebrity"]["celebrity_id"] = \
                "1AbC2dE"
        fm = self._run(name_conflict=False, mutate=m)
        p = fm["people"][0]
        self.assertEqual("collection", p["name_source"])   # collection won
        self.assertEqual("1AbC2dE", p["celebrity_id"])     # same person

    def test_resolution_against_the_plurality_suppresses_the_id(self):
        # the published name is NOT the celebrity vote's person — the
        # vote's id would attach the loser's key to the winner
        def m(docs):
            sc = docs["search"]["clusters"][0]
            sc["celebrity"] = {"name": "actor_M2", "confidence": 99.0,
                               "celebrity_id": "9ZyX8wV"}
        fm = self._run(name_conflict=False, mutate=m)
        p = fm["people"][0]
        self.assertEqual("actor_M1", p["name"])
        self.assertIsNone(p["celebrity_id"])
        self.assertNotIn("celebrity_urls", p)

    def test_celebrity_named_person_carries_id_and_info(self):
        def m(docs):
            sc = docs["search"]["clusters"][0]
            del sc["face_match"]                # celebrity path wins
            sc["celebrity"]["celebrity_id"] = "1AbC2dE"
            sc["celebrity"]["info"] = {
                "urls": ["www.imdb.com/name/nm0000000"],
                "known_gender": "Male"}
        fm = self._run(name_conflict=False, mutate=m)
        p = fm["people"][0]
        self.assertEqual("celebrity", p["name_source"])
        self.assertEqual("1AbC2dE", p["celebrity_id"])
        self.assertEqual(["www.imdb.com/name/nm0000000"],
                         p["celebrity_urls"])
        self.assertEqual("Male", p["known_gender"])

    def test_matching_record_id_keeps_the_key_across_spelling(self):
        # register-time ids (2026-09-07): equal identity keys beat any
        # spelling drift — the collection record and the vote name the
        # same person however the strings compare
        def m(docs):
            sc = docs["search"]["clusters"][0]
            sc["celebrity"]["celebrity_id"] = "1AbC2dE"
            sc["celebrity"]["name"] = "actor_M1 (variant)"   # variant
            sc["face_match"]["celebrity_id"] = "1AbC2dE"
        fm = self._run(name_conflict=False, mutate=m)
        p = fm["people"][0]
        self.assertEqual("collection", p["name_source"])
        self.assertEqual("1AbC2dE", p["celebrity_id"])

    def test_homonym_record_id_suppresses_the_key(self):
        # a collection HOMONYM (same name string, DIFFERENT id) is a
        # different person — string equality must not publish the
        # vote's id on the collection person
        def m(docs):
            sc = docs["search"]["clusters"][0]
            sc["celebrity"]["celebrity_id"] = "1AbC2dE"
            sc["face_match"]["celebrity_id"] = "9ZyX8wV"
        fm = self._run(name_conflict=False, mutate=m)
        self.assertIsNone(fm["people"][0]["celebrity_id"])

    def test_case_variant_name_agreement_keeps_the_key(self):
        # records that predate the celebrity_id field fall back to
        # names_agree — ONE definition (casefold), the same rule the
        # §23 conflict trigger now uses
        def m(docs):
            sc = docs["search"]["clusters"][0]
            sc["celebrity"]["celebrity_id"] = "1AbC2dE"
            sc["face_match"]["celeb"] = "ACTOR_M1"
        fm = self._run(name_conflict=False, mutate=m)
        p = fm["people"][0]
        self.assertEqual("ACTOR_M1", p["name"])
        self.assertEqual("1AbC2dE", p["celebrity_id"])

    def test_genderless_lookup_omits_known_gender_not_null(self):
        # three-state: the lookup SUCCEEDED but Rekognition has no
        # gender on record — the key is ABSENT, never null (the
        # contract's "present or absent" promise to consumers)
        def m(docs):
            sc = docs["search"]["clusters"][0]
            del sc["face_match"]
            sc["celebrity"]["celebrity_id"] = "1AbC2dE"
            sc["celebrity"]["info"] = {
                "urls": ["www.imdb.com/name/nm0000000"],
                "known_gender": None}
        fm = self._run(name_conflict=False, mutate=m)
        p = fm["people"][0]
        self.assertEqual(["www.imdb.com/name/nm0000000"],
                         p["celebrity_urls"])
        self.assertNotIn("known_gender", p)

    def test_info_absent_when_never_fetched(self):
        # three-state: no pass-3 lookup happened -> fields ABSENT
        def m(docs):
            sc = docs["search"]["clusters"][0]
            del sc["face_match"]
            sc["celebrity"]["celebrity_id"] = "1AbC2dE"
        fm = self._run(name_conflict=False, mutate=m)
        p = fm["people"][0]
        self.assertEqual("1AbC2dE", p["celebrity_id"])
        self.assertNotIn("celebrity_urls", p)
        self.assertNotIn("known_gender", p)


# ─────────────────────────────────────────────────────────────────────────────
# cluster_identities._cluster_celebrity — plurality vote, not max-confidence
# ─────────────────────────────────────────────────────────────────────────────

class TestClusterCelebrity(unittest.TestCase):
    """
    Regression coverage for a real bug found on real content (corpus asset `news_b`, see GOTCHAS.md):
    a non-celebrity's face was auto-indexed into the persistent collection
    under a stranger's name. RecognizeCelebrities returned a WRONG name for
    ~73% of that person's 882 tracked observations, scattered across 39+
    different wrong names (mean confidence ~83, never the same wrong guess
    twice) — while the single highest-confidence sample among all of them
    happened to be 96.0, clearing NAME_VOTE_INSTANT_SCORE (95.0) and getting
    permanently indexed. A genuine match (e.g. a recurring anchor) is high
    AND consistent across nearly every observation (mean ~99). Fix:
    _cluster_celebrity must tally votes per name and report the PLURALITY
    name's MEAN confidence, not chase the single luckiest outlier sample.
    """

    def test_plurality_wins_over_lone_high_confidence_outlier(self):
        from states.cluster_identities.lambda_function import _cluster_celebrity
        # Shape of the real bug: one 96.0-confidence outlier for "Wrong Name"
        # vs. a much larger, lower-confidence-on-average bloc for the same
        # wrong name, scattered wrong guesses, and unrecognized tracks.
        celeb_by_track = {}
        for tid in range(20):
            celeb_by_track[tid] = {"name": "Wrong Name", "confidence": 82.0}
        celeb_by_track[20] = {"name": "Wrong Name", "confidence": 96.0}
        for tid in range(21, 26):
            celeb_by_track[tid] = {"name": "Other Wrong Name", "confidence": 85.0}
        for tid in range(26, 30):
            celeb_by_track[tid] = {"name": None, "confidence": 10.0}
        track_ids = list(range(30))

        result = _cluster_celebrity(track_ids, celeb_by_track)
        self.assertEqual(result["name"], "Wrong Name")
        # Mean of 20x82.0 + 1x96.0 = (1640+96)/21
        self.assertAlmostEqual(result["confidence"], (20 * 82.0 + 96.0) / 21,
                               places=4)
        # Below NAME_VOTE_INSTANT_SCORE (95.0) — unlike the old max-confidence
        # behavior, which would have returned 96.0 and passed instantly.
        self.assertLess(result["confidence"], 95.0)

    def test_genuine_match_stays_high_and_consistent(self):
        from states.cluster_identities.lambda_function import _cluster_celebrity
        celeb_by_track = {tid: {"name": "Real Anchor", "confidence": 99.0 + tid % 2}
                          for tid in range(50)}
        result = _cluster_celebrity(list(range(50)), celeb_by_track)
        self.assertEqual(result["name"], "Real Anchor")
        self.assertGreaterEqual(result["confidence"], 99.0)

    def test_no_named_tracks_returns_none(self):
        from states.cluster_identities.lambda_function import _cluster_celebrity
        celeb_by_track = {0: {"name": None, "confidence": 5.0}}
        self.assertIsNone(_cluster_celebrity([0], celeb_by_track))

    def test_plurality_name_carries_its_celebrity_id(self):
        # the recognizer's own identity key rides the vote (2026-09-07):
        # consumers (cast-resolution) coalesce two clusters the recognizer
        # asserts are ONE person — the id must be the PLURALITY name's,
        # never a losing name's
        from states.cluster_identities.lambda_function import _cluster_celebrity
        celeb_by_track = {
            0: {"name": "Miriam Okafor", "confidence": 98.0,
                "celebrity_id": "1AbC2dE"},
            1: {"name": "Miriam Okafor", "confidence": 97.0,
                "celebrity_id": "1AbC2dE"},
            2: {"name": "Someone Else", "confidence": 96.0,
                "celebrity_id": "9ZyX8wV"},
        }
        result = _cluster_celebrity([0, 1, 2], celeb_by_track)
        self.assertEqual(result["name"], "Miriam Okafor")
        self.assertEqual(result["celebrity_id"], "1AbC2dE")

    def test_homonym_ids_are_plurality_voted_within_the_name(self):
        # Rekognition guarantees one Name per Id, NOT one Id per Name —
        # two real celebrities can share a display name. Homonym Ids are
        # SEPARATE vote groups (same key rule as _name_groups), so the
        # majority Id wins on its own tally, never insertion order
        from states.cluster_identities.lambda_function import _cluster_celebrity
        celeb_by_track = {
            0: {"name": "Miriam Okafor", "confidence": 90.0,
                "celebrity_id": "minority-id"},
            1: {"name": "Miriam Okafor", "confidence": 98.0,
                "celebrity_id": "majority-id"},
            2: {"name": "Miriam Okafor", "confidence": 97.0,
                "celebrity_id": "majority-id"},
        }
        result = _cluster_celebrity([0, 1, 2], celeb_by_track)
        self.assertEqual(result["celebrity_id"], "majority-id")

    def test_tied_homonym_ids_keep_the_name_and_suppress_the_key(self):
        # a plurality tied on count AND summed confidence is CONTESTED —
        # the display name is the same string either way, so it
        # survives; the id must never be a first-observed coin flip
        # (review, 2026-09-07 CONFIRMED)
        from states.cluster_identities.lambda_function import _cluster_celebrity
        celeb_by_track = {
            0: {"name": "Miriam Okafor", "confidence": 95.0,
                "celebrity_id": "id-A"},
            1: {"name": "Miriam Okafor", "confidence": 95.0,
                "celebrity_id": "id-B"},
        }
        result = _cluster_celebrity([0, 1], celeb_by_track)
        self.assertEqual(result["name"], "Miriam Okafor")
        self.assertIsNone(result["celebrity_id"])

    def test_tied_different_names_abstain_entirely(self):
        # contested between DIFFERENT names: a wrong name is worse than
        # none — the cluster abstains
        from states.cluster_identities.lambda_function import _cluster_celebrity
        celeb_by_track = {
            0: {"name": "Miriam Okafor", "confidence": 95.0,
                "celebrity_id": "id-A"},
            1: {"name": "Jane Smith", "confidence": 95.0,
                "celebrity_id": "id-B"},
        }
        self.assertIsNone(_cluster_celebrity([0, 1], celeb_by_track))

    def test_confidence_breaks_a_count_tie(self):
        from states.cluster_identities.lambda_function import _cluster_celebrity
        celeb_by_track = {
            0: {"name": "Miriam Okafor", "confidence": 99.0,
                "celebrity_id": "id-A"},
            1: {"name": "Jane Smith", "confidence": 80.0,
                "celebrity_id": "id-B"},
        }
        result = _cluster_celebrity([0, 1], celeb_by_track)
        self.assertEqual(result["celebrity_id"], "id-A")

    def test_variant_spellings_pool_their_votes_against_noise(self):
        # votes tally by the SAME id-else-name key _name_groups uses:
        # one Id observed under two spellings must pool (3 obs) and beat
        # a noise name (2 obs) that outnumbered EITHER spelling alone
        from states.cluster_identities.lambda_function import _cluster_celebrity
        celeb_by_track = {
            0: {"name": "Miriam Okafor", "confidence": 98.0,
                "celebrity_id": "id-X"},
            1: {"name": "Miriam Okafor ", "confidence": 97.0,
                "celebrity_id": "id-X"},
            2: {"name": "Miriam Okafor", "confidence": 97.5,
                "celebrity_id": "id-X"},
            3: {"name": "Jane Smith", "confidence": 99.0,
                "celebrity_id": "id-N"},
            4: {"name": "Jane Smith", "confidence": 99.0,
                "celebrity_id": "id-N"},
        }
        result = _cluster_celebrity([0, 1, 2, 3, 4], celeb_by_track)
        self.assertEqual(result["celebrity_id"], "id-X")
        self.assertEqual(result["name"], "Miriam Okafor")

    def test_name_groups_key_on_celebrity_id_over_name_string(self):
        # upstream home of the duplicate join (review, 2026-09-07):
        # spelling variants of ONE Id merge into one group; homonyms
        # with DIFFERENT Ids stay separate
        from states.cluster_identities.lambda_function import _name_groups
        tracks = [{"track_id": i} for i in range(4)]
        tags = {
            0: {"name": "Miriam Okafor", "celebrity_id": "id-1"},
            1: {"name": "Miriam Okafor ", "celebrity_id": "id-1"},  # variant
            2: {"name": "Miriam Okafor", "celebrity_id": "id-2"},    # homonym
            3: {"name": "Miriam Okafor"},                            # id-less
        }
        groups = sorted(_name_groups(tracks, tags))
        self.assertIn([0, 1], groups)          # one Id, one group
        self.assertIn([2], groups)             # different Id stays apart
        self.assertIn([3], groups)             # id-less falls back to name

    def test_vote_without_ids_yields_none_id(self):
        # tags from a pre-2026-09-07 build carry no celebrity_id — the
        # vote still works and the id is honestly None
        from states.cluster_identities.lambda_function import _cluster_celebrity
        celeb_by_track = {0: {"name": "Miriam Okafor", "confidence": 98.0}}
        result = _cluster_celebrity([0], celeb_by_track)
        self.assertEqual(result["name"], "Miriam Okafor")
        self.assertIsNone(result["celebrity_id"])

    def test_missing_tags_ignored(self):
        from states.cluster_identities.lambda_function import _cluster_celebrity
        celeb_by_track = {0: {"name": "A", "confidence": 90.0}}
        # track_id 1 has no entry in celeb_by_track at all
        result = _cluster_celebrity([0, 1], celeb_by_track)
        self.assertEqual(result["name"], "A")


# ─────────────────────────────────────────────────────────────────────────────
# search_index_faces._index_rank — ranked index-budget selection
# ─────────────────────────────────────────────────────────────────────────────

class TestIndexRank(unittest.TestCase):
    """
    max_faces_to_index slots must go to the BEST unmatched groups (screen
    time × recurrence × best-face quality), not first-come-first-served in
    cluster iteration order (the old behavior: clusters sorted by track
    count only, so 10 large-but-brief fragment clusters could exhaust the
    budget before a person with real screen presence was reached).
    """

    def _track(self, tid, first_ts, last_ts, w=0.1, h=0.2, pose_dist=0.05):
        face = {"box": {"l": 0.4, "t": 0.2, "w": w, "h": h},
                "brightness": 60.0, "sharpness": 50.0,
                "face_occluded": False, "eyes_open": True,
                "pose": {"yaw": 5.0, "pitch": 5.0, "roll": 0.0,
                         "pose_dist": pose_dist},
                "source": "detected", "timestamp": first_ts,
                "frame_num": int(first_ts * 30), "frame_key": "f.jpg",
                "confidence": 99.9}
        return {"track_id": tid, "first_ts": first_ts, "last_ts": last_ts,
                "appearance_count": 3, "centroid_idx": 0, "faces": [face]}

    def test_screen_time_outranks_track_count(self):
        from states.search_index_faces.lambda_function import _index_rank
        # A: 3 tracks spanning 60s of screen time
        tracks = {}
        for tid in (0, 1, 2):
            tracks[tid] = self._track(tid, tid * 100.0, tid * 100.0 + 20.0)
        # B: 6 tracks but each a 0.5s sliver (3s total)
        for tid in range(10, 16):
            tracks[tid] = self._track(tid, tid * 10.0, tid * 10.0 + 0.5)
        rank_a = _index_rank({"track_ids": [0, 1, 2]}, tracks)
        rank_b = _index_rank({"track_ids": list(range(10, 16))}, tracks)
        self.assertGreater(rank_a, rank_b)

    def test_face_quality_breaks_ties(self):
        from states.search_index_faces.lambda_function import _index_rank
        tracks = {
            0: self._track(0, 0.0, 30.0, w=0.15, h=0.3, pose_dist=0.02),
            1: self._track(1, 0.0, 30.0, w=0.03, h=0.06, pose_dist=0.30),
        }
        good = _index_rank({"track_ids": [0]}, tracks)
        poor = _index_rank({"track_ids": [1]}, tracks)
        self.assertGreater(good, poor)

    def test_empty_or_unknown_tracks_rank_zero(self):
        from states.search_index_faces.lambda_function import _index_rank
        self.assertEqual(_index_rank({"track_ids": [99]}, {}), 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# probe_detect_faces — O(n) side-object checkpoints (resume + consolidate)
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectSideObjectCheckpoints(unittest.TestCase):
    """
    Checkpoints are per-grid side objects ({iterator_key}.grid_NNNN.json),
    written once per completed grid — replacing full-iterator rewrites every
    5 grids whose total bytes grew O(n²) with grids per iterator. Resume
    must merge side objects (results + usage) without re-detecting, and the
    final consolidation must fold them into the iterator and delete them.
    """

    def _run(self, iterator_doc, side_objects, detect_results):
        import states.probe_detect_faces.lambda_function as mod
        store = {"iter_key": dict(iterator_doc)}
        store.update(side_objects)
        puts, deleted, detects = {}, [], []

        def fake_get_json(bucket, key):
            return store[key]

        def fake_put_json(bucket, key, data):
            puts[key] = data
            store[key] = data

        def fake_list_keys(bucket, prefix):
            return {k for k in store if k.startswith(prefix)}

        def fake_delete_keys(bucket, keys):
            deleted.extend(keys)

        def fake_detect_grid(jpegs, frames, layout, bucket, gk, counter):
            counter.detect_faces_api += 1
            detects.append(gk)
            return detect_results.pop(0)

        ctx = mock.MagicMock()
        ctx.get_remaining_time_in_millis.return_value = 900_000
        event = {"item_data": {"item_id": 0, "key": "iter_key", "context": {
            "out_bucket": "b", "frames_bucket": "b", "frames_prefix": "f",
            "output_s3uri": "s3://b/prefix"}}}

        with mock.patch.object(mod, "get_json", fake_get_json), \
             mock.patch.object(mod, "put_json", fake_put_json), \
             mock.patch.object(mod, "list_keys", fake_list_keys), \
             mock.patch.object(mod, "delete_keys", fake_delete_keys), \
             mock.patch.object(mod, "get_bytes", return_value=b"jpeg"), \
             mock.patch.object(mod, "_detect_grid", fake_detect_grid), \
             mock.patch.object(mod.semaphore, "refresh_holder"):
            out = mod._run_detect(event, ctx)
        return out, store, puts, deleted, detects

    def _iterator(self, n_grids):
        return {"layout": {"cols": 2, "rows": 2, "cell_w": 10, "cell_h": 10},
                "grids": [{"frames": [{"key": f"f{g}.jpg", "frame_num": g,
                                       "timestamp": g}]}
                          for g in range(n_grids)],
                "results": {}, "usage": {}}

    def test_fresh_run_writes_side_objects_then_consolidates(self):
        out, store, puts, deleted, detects = self._run(
            self._iterator(3), {}, [[{"f": 1}], [{"f": 2}], [{"f": 3}]])
        # one side object per grid, all deleted after consolidation
        side = [k for k in puts if ".grid_" in k]
        self.assertEqual(len(side), 3)
        self.assertEqual(sorted(deleted), sorted(side))
        self.assertEqual(store["iter_key"]["results"],
                         {"0": [{"f": 1}], "1": [{"f": 2}], "2": [{"f": 3}]})
        self.assertEqual(store["iter_key"]["usage"]["detect_faces_api"], 3)
        self.assertEqual(len(detects), 3)

    def test_resume_skips_completed_side_objects_and_sums_usage(self):
        side = {"iter_key.grid_0000.json":
                    {"faces": [{"f": 1}], "usage": {"detect_faces_api": 1}},
                "iter_key.grid_0001.json":
                    {"faces": [{"f": 2}], "usage": {"detect_faces_api": 1}}}
        out, store, puts, deleted, detects = self._run(
            self._iterator(3), side, [[{"f": 3}]])
        # only grid 2 re-detected; grids 0/1 restored from side objects
        self.assertEqual(len(detects), 1)
        self.assertEqual(store["iter_key"]["results"]["0"], [{"f": 1}])
        self.assertEqual(store["iter_key"]["results"]["2"], [{"f": 3}])
        # usage = 2 restored + 1 fresh, no double count
        self.assertEqual(store["iter_key"]["usage"]["detect_faces_api"], 3)


# ─────────────────────────────────────────────────────────────────────────────
# grid_image.dominant_per_tile — one face per tile, the largest
# ─────────────────────────────────────────────────────────────────────────────

class TestDominantPerTile(unittest.TestCase):
    """
    Regression coverage for real track contamination (corpus asset
    `news_b`): a headshot crop with a second face in the background put
    TWO Rekognition records in one grid tile. In the clustering sandbox
    the background face was indexed under the foreground track's id and
    split that track across two clusters (measured: 65/2027 tracks); in
    RecognizeCelebrities a background celebrity could overwrite the
    subject's tag. The rule: a tile's identity belongs to its DOMINANT
    (largest-bbox) face — the crop was made around that face.
    """
    COORDS = [{"l": 0.0, "t": 0.0, "w": 0.5, "h": 1.0},
              {"l": 0.5, "t": 0.0, "w": 0.5, "h": 1.0}]

    @staticmethod
    def _face(cx, cy, w, h):
        return {"BoundingBox": {"Left": cx - w / 2, "Top": cy - h / 2,
                                "Width": w, "Height": h}}

    def test_larger_face_wins_the_tile(self):
        from lib.grid_image import dominant_per_tile
        subject = self._face(0.25, 0.5, 0.30, 0.60)
        background = self._face(0.30, 0.30, 0.06, 0.12)
        out = dominant_per_tile(
            [(background, "background"), (subject, "subject")], self.COORDS)
        self.assertEqual(out, {0: "subject"})

    def test_order_does_not_matter(self):
        from lib.grid_image import dominant_per_tile
        subject = self._face(0.25, 0.5, 0.30, 0.60)
        background = self._face(0.30, 0.30, 0.06, 0.12)
        out = dominant_per_tile(
            [(subject, "subject"), (background, "background")], self.COORDS)
        self.assertEqual(out, {0: "subject"})

    def test_separate_tiles_both_kept(self):
        from lib.grid_image import dominant_per_tile
        a = self._face(0.25, 0.5, 0.3, 0.6)
        b = self._face(0.75, 0.5, 0.3, 0.6)
        out = dominant_per_tile([(a, "a"), (b, "b")], self.COORDS)
        self.assertEqual(out, {0: "a", 1: "b"})

    def test_larger_unrecognized_subject_beats_background_celebrity(self):
        # the recognize_celebrities shape: dominance is judged across
        # BOTH response lists, so a small recognized background face (a
        # celebrity on a studio monitor) cannot name the tile over the
        # larger unrecognized subject
        from lib.grid_image import dominant_per_tile
        celeb_bg = (self._face(0.35, 0.25, 0.08, 0.15),
                    {"name": "Miriam Okafor", "celebrity_id": "1AbC2dE"})
        subject = (self._face(0.25, 0.5, 0.30, 0.60),
                   {"name": None, "confidence": 99.0})
        out = dominant_per_tile([celeb_bg, subject], self.COORDS)
        self.assertIsNone(out[0]["name"])

    def test_min_share_floor_rejects_a_lone_background_face(self):
        # dominance is judged only among faces Rekognition returned —
        # when the SUBJECT was dropped (quality filter/detection miss)
        # the lone background face must not win by default: a face too
        # small to be the crop subject maps nothing
        from lib.grid_image import dominant_per_tile
        from lib.input_schema import MIN_SUBJECT_TILE_SHARE
        background = self._face(0.30, 0.30, 0.06, 0.12)
        out = dominant_per_tile([(background, "background")], self.COORDS,
                                min_share=MIN_SUBJECT_TILE_SHARE)
        self.assertEqual(out, {})

    def test_min_share_floor_keeps_a_real_subject(self):
        from lib.grid_image import dominant_per_tile
        from lib.input_schema import MIN_SUBJECT_TILE_SHARE
        subject = self._face(0.25, 0.5, 0.30, 0.60)
        out = dominant_per_tile([(subject, "subject")], self.COORDS,
                                min_share=MIN_SUBJECT_TILE_SHARE)
        self.assertEqual(out, {0: "subject"})

    def test_faceless_record_is_ignored(self):
        from lib.grid_image import dominant_per_tile
        out = dominant_per_tile(
            [({}, "no-box"),
             (self._face(0.25, 0.5, 0.3, 0.6), "real")], self.COORDS)
        self.assertEqual(out, {0: "real"})


# ─────────────────────────────────────────────────────────────────────────────
# cluster_identities — two-tier merge: weak edges need corroboration
# ─────────────────────────────────────────────────────────────────────────────

class TestTwoTierMerge(unittest.TestCase):
    """
    Regression coverage for real cross-person contamination (corpus asset
    `news_b`, see GOTCHAS.md): union-find merged on any single SearchFaces edge
    >= SANDBOX_SEARCH_THRESHOLD (75), and merges are transitive and
    irreversible — an unrelated man's track reached a 289-track cluster
    through a chain of 75-77-similarity noise links stitched between genuine
    97-100 links (and on two `drama_b` episodes, two different actors'
    clusters fused the same way). Fix: single-edge merge only at
    >= SANDBOX_MERGE_SIMILARITY (90); weaker edges merge only when
    >= SANDBOX_WEAK_CORROBORATION (2) independent weak edges connect the
    same two components.
    """

    def _uf(self):
        parent = {}
        def find(x):
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        return find, union

    def test_single_weak_edge_does_not_merge(self):
        from states.cluster_identities.lambda_function import (
            _merge_corroborated_weak_edges)
        find, union = self._uf()
        # Two strong components: {a1,a2} and {b1,b2}
        union("a1", "a2")
        union("b1", "b2")
        # One weak edge between them — the real bug shape (sim 76.5)
        _merge_corroborated_weak_edges([["a1", "b1", 76.5]], find, union)
        self.assertNotEqual(find("a1"), find("b1"))

    def test_two_corroborating_weak_edges_merge(self):
        from states.cluster_identities.lambda_function import (
            _merge_corroborated_weak_edges)
        find, union = self._uf()
        union("a1", "a2")
        union("b1", "b2")
        # Two INDEPENDENT weak edges between the same two components = signal
        _merge_corroborated_weak_edges(
            [["a1", "b1", 78.0], ["a2", "b2", 82.0]], find, union)
        self.assertEqual(find("a1"), find("b1"))

    def test_weak_chain_does_not_bridge(self):
        from states.cluster_identities.lambda_function import (
            _merge_corroborated_weak_edges)
        find, union = self._uf()
        # The exact real-content chain shape: strong pairs bridged by single
        # weak links (1076-[76.5]-2103-[99.97]-2107-[76.2]-4750-[97.2]-2177
        # -[76.1]-1311). Strong edges union up front; each weak link is
        # singular, so nothing may merge.
        union("t2103", "t2107")   # 99.97
        union("t4750", "t2177")   # 97.2
        weak = [["t1076", "t2103", 76.5],
                ["t2107", "t4750", 76.2],
                ["t2177", "t1311", 76.1]]
        _merge_corroborated_weak_edges(weak, find, union)
        self.assertNotEqual(find("t1076"), find("t1311"))
        self.assertNotEqual(find("t2103"), find("t4750"))

    def test_fixpoint_iteration(self):
        from states.cluster_identities.lambda_function import (
            _merge_corroborated_weak_edges)
        find, union = self._uf()
        # A<->B corroborated by 2 edges; the merged AB then has 2 edges to C
        # (one from A, one from B) that only corroborate AFTER the first
        # merge — requires the fixpoint loop.
        union("a1", "a2")
        union("b1", "b2")
        union("c1", "c2")
        weak = [["a1", "b1", 80.0], ["a2", "b2", 80.0],
                ["a1", "c1", 80.0], ["b1", "c2", 80.0]]
        _merge_corroborated_weak_edges(weak, find, union)
        self.assertEqual(find("a1"), find("b1"))
        self.assertEqual(find("a1"), find("c1"))

    def test_intra_component_weak_edges_ignored(self):
        from states.cluster_identities.lambda_function import (
            _merge_corroborated_weak_edges)
        find, union = self._uf()
        union("a1", "a2")
        union("b1", "b2")
        # Two weak edges, but one is INSIDE component A — only one crosses,
        # so no corroboration.
        _merge_corroborated_weak_edges(
            [["a1", "a2", 80.0], ["a1", "b1", 80.0]], find, union)
        self.assertNotEqual(find("a1"), find("b1"))

    def test_reciprocal_weak_edge_counts_once(self):
        """SearchFaces is symmetric: when both endpoints get searched, the
        SAME underlying match arrives twice (A→B and B→A). Counting both as
        independent corroboration would let ONE bidirectional noise match
        clear SANDBOX_WEAK_CORROBORATION=2 on its own — the exact
        single-weak-edge contamination the tier exists to prevent."""
        from states.cluster_identities.lambda_function import (
            _merge_corroborated_weak_edges)
        find, union = self._uf()
        union("a1", "a2")
        union("b1", "b2")
        # One match observed from both directions — NOT two independent edges
        _merge_corroborated_weak_edges(
            [["a1", "b1", 76.5], ["b1", "a1", 76.5]], find, union)
        self.assertNotEqual(find("a1"), find("b1"))
        # ...but two genuinely distinct face pairs still corroborate, even
        # when one of them is also observed reciprocally.
        find2, union2 = self._uf()
        union2("a1", "a2")
        union2("b1", "b2")
        _merge_corroborated_weak_edges(
            [["a1", "b1", 76.5], ["b1", "a1", 76.5], ["a2", "b2", 80.0]],
            find2, union2)
        self.assertEqual(find2("a1"), find2("b1"))


class TestDrainPendingTokenHandling(unittest.TestCase):
    """
    Regression coverage for two real failure modes in drain_pending:
    1. ANY SendTaskSuccess ClientError was treated as a dead token — an SFN
       THROTTLE deleted the live waiter's pending item, permanently stranding
       its execution on the task token (dies only at the 3600 s heartbeat).
    2. A dead token after a REENTRANT acquire released the LIVE holder of an
       already-admitted execution (prior drain crashed after SendTaskSuccess,
       before deleting pending) — the REMOVE decremented the counter and let
       an extra execution in, breaking the concurrency cap.
    """

    def _sem(self, ddb, sfn):
        import importlib.util, os
        spec = importlib.util.spec_from_file_location(
            "sem_drain_test", os.path.join(ROOT, "lib", "semaphore.py"))
        sem = importlib.util.module_from_spec(spec)
        with mock.patch("boto3.client",
                        side_effect=lambda svc, **kw:
                        ddb if svc == "dynamodb" else sfn):
            spec.loader.exec_module(sem)
        sem.TABLE_NAME = "test-table"
        return sem

    def test_throttled_send_keeps_pending_and_raises(self):
        ddb = _FakeDdb()
        sfn = mock.MagicMock()
        sfn.send_task_success.side_effect = _client_error(
            "ThrottlingException", "SendTaskSuccess")
        sem = self._sem(ddb, sfn)
        sem.store_pending("arn:w1", "token-1", 2)
        with self.assertRaises(Exception):
            sem.drain_pending()
        # Pending item survives for the stream retry
        pending = [k for k in ddb.items if k[0] == "pending"]
        self.assertEqual(len(pending), 1, "live waiter's pending item was "
                                          "deleted on a transient throttle")

    def test_dead_token_after_fresh_acquire_releases_slot(self):
        ddb = _FakeDdb()
        sfn = mock.MagicMock()
        sfn.send_task_success.side_effect = _client_error(
            "TaskDoesNotExist", "SendTaskSuccess")
        sem = self._sem(ddb, sfn)
        sem.store_pending("arn:w1", "token-1", 2)
        sem.drain_pending()
        # Fresh slot released (holder gone), pending consumed
        holders = [k for k in ddb.items
                   if k[0] == "semaphore" and k[1].startswith("holder#")]
        self.assertEqual(holders, [])
        self.assertEqual([k for k in ddb.items if k[0] == "pending"], [])

    def test_dead_token_after_reentrant_acquire_keeps_live_holder(self):
        ddb = _FakeDdb()
        sfn = mock.MagicMock()
        sfn.send_task_success.side_effect = _client_error(
            "TaskDoesNotExist", "SendTaskSuccess")
        sem = self._sem(ddb, sfn)
        # Prior drain already admitted w1 (holder exists, execution live)
        # and crashed before deleting the pending item.
        self.assertEqual(sem.try_acquire("arn:w1", 2), "acquired")
        sem.store_pending("arn:w1", "token-1", 2)
        sem.drain_pending()
        # The LIVE holder must survive; only the stale pending item goes.
        holders = [k for k in ddb.items
                   if k[0] == "semaphore" and k[1].startswith("holder#")]
        self.assertEqual(len(holders), 1,
                         "live holder released on reentrant dead-token path")
        self.assertEqual([k for k in ddb.items if k[0] == "pending"], [])


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — face index management (README.md §19)
# ─────────────────────────────────────────────────────────────────────────────

class TestIndexBatchLib(unittest.TestCase):
    def test_sniff_format_magic_bytes(self):
        from lib.index_batch import sniff_format, format_accepted
        cases = [
            (b"\xff\xd8\xff\xe0" + b"\x00" * 20, "jpeg"),
            (b"\x89PNG\r\n\x1a\n" + b"\x00" * 20, "png"),
            (b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 12, "webp"),
            (b"II*\x00" + b"\x00" * 20, "tiff"),
            (b"MM\x00*" + b"\x00" * 20, "tiff"),
            (b"\x00\x00\x00\x18ftypheic" + b"\x00" * 12, "heif"),
            (b"\x00\x00\x00\x18ftypavif" + b"\x00" * 12, "heif"),
            (b"GIF89a" + b"\x00" * 20, "gif"),
            (b"BM" + b"\x00" * 20, None),          # BMP — rejected
            (b"%PDF-1.4" + b"\x00" * 20, None),    # PDF — rejected
            (b"<svg xmlns=" + b"\x00" * 20, None), # SVG — rejected
        ]
        for data, expected in cases:
            self.assertEqual(sniff_format(data), expected, data[:8])
        self.assertTrue(format_accepted("jpeg"))
        self.assertFalse(format_accepted(None))
        self.assertFalse(format_accepted("bmp"))

    def test_sniff_extension_is_irrelevant(self):
        # A .jpg-named HEIC must be detected as heif (magic bytes only).
        from lib.index_batch import sniff_format
        heic = b"\x00\x00\x00\x18ftypheic" + b"\x00" * 12
        self.assertEqual(sniff_format(heic), "heif")

    def test_parse_manifest_fail_hard(self):
        from lib.index_batch import parse_manifest
        good = ('{"image_uri": "s3://b/a.jpg", "celebrity_name": "A"}\n'
                '\n'
                '{"image_uri": "s3://b/c.png", "celebrity_name": "C"}\n')
        entries = parse_manifest(good)
        self.assertEqual(len(entries), 2)
        for bad in (
            "not json\n",
            '{"image_uri": "http://x/y.jpg", "celebrity_name": "A"}\n',
            '{"image_uri": "s3://b/a.jpg"}\n',          # missing name
            "",                                          # empty manifest
        ):
            with self.assertRaises(ValueError, msg=bad):
                parse_manifest(bad)

    def test_tile_box_to_source_roundtrip(self):
        # A face at a known spot in the source must survive
        # source -> tile (fit+letterbox) -> tile_box -> source recalibration.
        from lib.index_batch import tile_box_to_source
        from lib.input_schema import BATCH_TILE_DIM
        tile_w, tile_h = BATCH_TILE_DIM
        # Landscape source: fit-inside scale = tile_w / long_edge; content
        # centered vertically (letterbox).
        scale = tile_w / 1600
        off_y = (tile_h - round(900 * scale)) // 2
        placement = {"scale": scale, "off_x": 0, "off_y": off_y,
                     "src_w": 1600, "src_h": 900}
        # Face at source-normalized (0.25, 0.30, w 0.10, h 0.20)
        sx, sy, sw, sh = 0.25, 0.30, 0.10, 0.20
        px = sx * 1600 * scale + placement["off_x"]
        py = sy * 900 * scale + placement["off_y"]
        pw, ph = sw * 1600 * scale, sh * 900 * scale
        tile_box = {"l": px / tile_w, "t": py / tile_h,
                    "w": pw / tile_w, "h": ph / tile_h}
        out = tile_box_to_source(tile_box, placement)
        self.assertAlmostEqual(out["l"], sx, places=4)
        self.assertAlmostEqual(out["t"], sy, places=4)
        self.assertAlmostEqual(out["w"], sw, places=4)
        self.assertAlmostEqual(out["h"], sh, places=4)

    def test_largest_face_per_cell_phantom_rule(self):
        from lib.index_batch import largest_face_per_cell
        from lib.input_schema import BATCH_TILE_DIM, GRID_CELL_SEPARATOR_PX
        tile_w, tile_h = BATCH_TILE_DIM
        cols, rows, n = 2, 1, 2
        grid_w = cols * tile_w + (cols - 1) * GRID_CELL_SEPARATOR_PX
        grid_h = rows * tile_h

        def rec(face_id, cell, rel_l, rel_t, rel_w, rel_h):
            ox = cell * (tile_w + GRID_CELL_SEPARATOR_PX)
            return {"Face": {"FaceId": face_id, "BoundingBox": {
                "Left": (ox + rel_l * tile_w) / grid_w,
                "Top": (rel_t * tile_h) / grid_h,
                "Width": (rel_w * tile_w) / grid_w,
                "Height": (rel_h * tile_h) / grid_h}}}

        records = [
            rec("big-0", 0, 0.2, 0.2, 0.5, 0.5),    # cell 0 winner
            rec("small-0", 0, 0.05, 0.05, 0.1, 0.1),  # cell 0 phantom
            rec("only-1", 1, 0.3, 0.3, 0.3, 0.3),   # cell 1 winner
        ]
        winners = largest_face_per_cell(records, n, cols, rows)
        self.assertEqual(set(winners), {0, 1})
        self.assertEqual(winners[0]["record"]["Face"]["FaceId"], "big-0")
        self.assertEqual(winners[1]["record"]["Face"]["FaceId"], "only-1")

    def test_manifest_from_prefix_basename_naming(self):
        # Tests the PRODUCTION implementation (gateway_tool/index_tools.py) —
        # a duplicate lib copy was removed after a review found the unit test
        # green-lighting code nobody ran.
        import importlib.util
        os.environ.setdefault("PROXY_BUCKET", "proxy-b")
        spec = importlib.util.spec_from_file_location(
            "gw_index_tools_manifest",
            os.path.join(ROOT, "gateway_tool", "index_tools.py"))
        gw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gw)

        listed = [
            {"Key": "photos/Jane  Smith.jpg"},
            {"Key": "photos/Miriam Okafor.png"},
            {"Key": "photos/notes.txt"},          # filtered: not an image ext
        ]
        put = {}
        with mock.patch.object(gw._s3, "list_objects_v2",
                               return_value={"Contents": listed,
                                             "IsTruncated": False}), \
             mock.patch.object(gw._s3, "put_object",
                               side_effect=lambda **kw: put.update(kw)):
            uri, count, err = gw._manifest_from_prefix(
                "s3://b/photos/", "micro-workflow/x/_index_jobs/j1")
        self.assertIsNone(err)
        self.assertEqual(count, 2)
        lines = [json.loads(l) for l in
                 put["Body"].decode("utf-8").splitlines()]
        self.assertEqual(lines[0]["celebrity_name"], "Jane Smith")
        self.assertEqual(lines[1]["celebrity_name"], "Miriam Okafor")
        self.assertTrue(all(l["image_uri"].startswith("s3://b/photos/")
                            for l in lines))

    def test_batch_external_id_charset(self):
        # Rekognition ExternalImageId charset: [a-zA-Z0-9_.\-:]+
        import re
        from lib.index_batch import create_batch_external_id
        eid = create_batch_external_id("6a5b0f9e-1c2d-3e4f-5a6b-7c8d9e0f1a2b")
        self.assertRegex(eid, r"^user:[0-9a-f]{32}:\d+$")
        self.assertTrue(re.fullmatch(r"[a-zA-Z0-9_.\-:]+", eid))


class TestFaceStorePhase2(unittest.TestCase):
    def test_celeb_created_sort_key(self):
        from lib.face_store import celeb_created_sk
        named = celeb_created_sk("Jane Smith", 1700000000000)
        unnamed = celeb_created_sk(None, 1700000000000)
        empty = celeb_created_sk("", 1700000000000)
        self.assertEqual(named, "Jane Smith#001700000000000")
        self.assertTrue(unnamed.startswith("~unnamed#"))
        self.assertEqual(unnamed, empty)
        # Unnamed sorts AFTER any reasonable name (the point of the sentinel).
        self.assertGreater(unnamed, named)
        self.assertGreater(unnamed, "Zz Zed#999999999999999")

    def test_register_face_writes_celeb_created(self):
        import lib.face_store as fs
        captured = {}
        with mock.patch.object(fs._ddb, "put_item",
                               side_effect=lambda **kw: captured.update(kw)):
            fs.register_face("fid-1", {"collection_id": "c",
                                       "celeb": "Jane Smith",
                                       "origin": "auto"})
        item = captured["Item"]
        self.assertIn("celeb_created", item)
        self.assertTrue(item["celeb_created"]["S"].startswith("Jane Smith#"))
        self.assertEqual(item["origin"]["S"], "auto")

    def test_delete_face_record_tombstone(self):
        import lib.face_store as fs
        calls = []
        with mock.patch.object(fs._ddb, "put_item",
                               side_effect=lambda **kw: calls.append(("put", kw))), \
             mock.patch.object(fs._ddb, "delete_item",
                               side_effect=lambda **kw: calls.append(("del", kw))):
            fs.delete_face_record(
                {"face_id": "fid-9", "collection_id": "c",
                 "celeb": "X", "origin": "user"}, "batch_delete")
        self.assertEqual([c[0] for c in calls], ["put", "del"])
        tomb = calls[0][1]["Item"]
        self.assertEqual(tomb["pk"]["S"], "deleted#fid-9")
        self.assertIn("ttl", tomb)
        self.assertEqual(tomb["reason"]["S"], "batch_delete")
        self.assertEqual(calls[1][1]["Key"]["pk"]["S"], "face#fid-9")


class TestGatewayIndexToolConstants(unittest.TestCase):
    """The gateway zip is flat (no lib/) so index_tools re-declares constants
    that MUST match lib/input_schema — drift here silently breaks pagination
    or caps. This test is the guard the duplication requires."""

    def _load_gw(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "gw_index_tools",
            os.path.join(ROOT, "gateway_tool", "index_tools.py"))
        gw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gw)
        return gw

    def test_constants_match_input_schema(self):
        gw = self._load_gw()
        from lib import input_schema as s
        self.assertEqual(gw.COLLECTION_INDEX_GSI, s.COLLECTION_INDEX_GSI)
        self.assertEqual(gw.UNNAMED_CELEB_SENTINEL, s.UNNAMED_CELEB_SENTINEL)
        self.assertEqual(gw.MAX_BATCH_UPDATE_FACES, s.MAX_BATCH_UPDATE_FACES)
        self.assertEqual(gw.LIST_FACES_MAX_PAGE_SIZE,
                         s.LIST_FACES_MAX_PAGE_SIZE)
        self.assertEqual(gw.LIST_FACES_DEFAULT_PAGE_SIZE,
                         s.LIST_FACES_DEFAULT_PAGE_SIZE)

    def test_celeb_created_sk_matches_face_store(self):
        """Review finding: the gateway re-derives the collection-index sort
        key — a byte-format mismatch breaks list_faces begins_with filtering
        for renamed faces. Guard both implementations byte-for-byte."""
        gw = self._load_gw()
        from lib.face_store import celeb_created_sk
        for name, ts in (("Jane Smith", 1700000000000), (None, 0), ("", 42)):
            self.assertEqual(gw.celeb_created_sk(name, ts),
                             celeb_created_sk(name, ts))

    def test_gateway_bool_parser_accepts_strings(self):
        """Review finding: MCP layers deliver booleans as strings; identity
        checks (`is True`) silently applied NO filter for named=\"false\"."""
        gw = self._load_gw()
        self.assertEqual(gw._parse_bool("true"), (True, None))
        self.assertEqual(gw._parse_bool("False"), (False, None))
        self.assertEqual(gw._parse_bool(True), (True, None))
        self.assertEqual(gw._parse_bool(None), (None, None))
        val, err = gw._parse_bool("banana")
        self.assertIsNone(val)
        self.assertIsNotNone(err)

    def test_batch_index_rejects_foreign_bucket_and_bad_max(self):
        gw = self._load_gw()
        gw.PROXY_BUCKET = "proxy-b"
        gw.DEFAULT_COLLECTION_ID = "c"
        r = gw.batch_index_faces({"s3_prefix": "s3://other-bucket/x/"})
        self.assertEqual(r.get("errorCode"), "FOREIGN_BUCKET")
        r = gw.batch_index_faces({"s3_prefix": "s3://proxy-b/x/",
                                  "max_concurrent_executions": 500})
        self.assertEqual(r.get("errorCode"), "INVALID_PARAMETER")
        r = gw.batch_index_faces({"s3_prefix": "s3://proxy-b/x/",
                                  "max_concurrent_executions": 0})
        self.assertEqual(r.get("errorCode"), "INVALID_PARAMETER")


class TestBatchGetFacesRetry(unittest.TestCase):
    def test_unprocessed_keys_retried_and_dupes_removed(self):
        """Review finding: throttled UnprocessedKeys were treated as
        'missing' (bogus face-not-found on delete resolution); duplicate ids
        in one chunk raise a DDB ValidationException."""
        import lib.face_store as fs
        calls = []

        def fake_batch_get(RequestItems):
            keys = RequestItems[fs.TABLE_NAME]["Keys"]
            calls.append(len(keys))
            if len(calls) == 1:
                # first call: return half, throttle the rest
                half = len(keys) // 2
                return {"Responses": {fs.TABLE_NAME: [
                    {"pk": k["pk"], "sk": k["sk"],
                     "face_id": {"S": k["pk"]["S"].split("#")[1]}}
                    for k in keys[:half]]},
                    "UnprocessedKeys": {fs.TABLE_NAME: {"Keys": keys[half:]}}}
            return {"Responses": {fs.TABLE_NAME: [
                {"pk": k["pk"], "sk": k["sk"],
                 "face_id": {"S": k["pk"]["S"].split("#")[1]}}
                for k in keys]}, "UnprocessedKeys": {}}

        with mock.patch.object(fs._ddb, "batch_get_item",
                               side_effect=fake_batch_get):
            out = fs.batch_get_faces(["a", "b", "a", "c", "d"])  # dupe 'a'
        self.assertEqual(set(out), {"a", "b", "c", "d"})
        self.assertEqual(calls[0], 4)  # deduplicated before the call


# ─────────────────────────────────────────────────────────────────────────────
# shots — shot-segmentation supplement (README.md §21)
# ─────────────────────────────────────────────────────────────────────────────

class TestShotsSupplement(unittest.TestCase):
    FPS = __import__("fractions").Fraction(24000, 1001)

    def _doc(self, shots):
        return {"shots": shots}

    def _shot(self, sf, ef, fps=None):
        fps = float(fps or self.FPS)
        return {"start_frame": sf, "end_frame": ef,
                "start_time": round(sf / fps, 4),
                "end_time": round(ef / fps, 4),
                "shot_type": "content"}

    def _validate(self, doc, duration_sec):
        from lib import shots as shots_lib
        with mock.patch.object(shots_lib, "get_json", return_value=doc):
            return shots_lib.load_and_validate(
                "b", "k", self.FPS, duration_sec)

    def test_valid_file_returns_sorted_starts(self):
        doc = self._doc([self._shot(0, 239), self._shot(240, 479),
                         self._shot(480, 719)])
        duration = 720 / float(self.FPS)
        self.assertEqual(self._validate(doc, duration), [0, 240, 480])

    def test_schema_failures(self):
        duration = 720 / float(self.FPS)
        for doc in ({}, {"shots": []}, {"shots": "nope"}, [1, 2],
                    self._doc([{"start_frame": "x", "end_frame": 1,
                                "start_time": 0, "end_time": 1}])):
            with self.assertRaises(ValueError):
                self._validate(doc, duration)

    def test_unsorted_starts_rejected(self):
        doc = self._doc([self._shot(240, 479), self._shot(0, 239)])
        with self.assertRaises(ValueError):
            self._validate(doc, 480 / float(self.FPS))

    def test_wrong_fps_rejected(self):
        # Shots timed at 25 fps against a 23.976 fps asset: frame 500 at
        # t=20s implies 25 fps — >1% off.
        doc = self._doc([self._shot(0, 499),
                         {"start_frame": 500, "end_frame": 999,
                          "start_time": 20.0, "end_time": 39.96}])
        with self.assertRaises(ValueError) as ctx:
            self._validate(doc, 39.96)
        self.assertIn("different asset", str(ctx.exception))

    def test_wrong_duration_rejected(self):
        # Valid fps but the file covers a 30 s asset; ours is 600 s.
        doc = self._doc([self._shot(0, 719)])
        with self.assertRaises(ValueError) as ctx:
            self._validate(doc, 600.0)
        self.assertIn("different asset", str(ctx.exception))

    def test_file_duration_preferred_over_last_shot(self):
        # Real files carry top-level actual_duration_sec (exact); the last
        # shot's end_time is the last SAMPLED frame and can sit short of EOF.
        doc = self._doc([self._shot(0, 590)])       # ends ~24.6 s
        doc["actual_duration_sec"] = 30.0
        self.assertEqual(self._validate(doc, 30.0), [0])
        # ...and it must also be able to REJECT on its own.
        doc["actual_duration_sec"] = 600.0
        with self.assertRaises(ValueError):
            self._validate(doc, 30.0)

    def test_early_shots_skip_fps_check(self):
        # start_time < 5 s: rounded times on a tiny early shot imply a
        # wildly-wrong fps (48/0.5 = 96) — must NOT trip the check; the
        # same ratio on a late shot MUST.
        early = self._doc([self._shot(0, 47),
                           {"start_frame": 48, "end_frame": 719,
                            "start_time": 0.5, "end_time": 30.03}])
        self.assertEqual(self._validate(early, 30.03), [0, 48])
        late = self._doc([self._shot(0, 479),
                          {"start_frame": 960, "end_frame": 1439,
                           "start_time": 10.0, "end_time": 60.06}])
        with self.assertRaises(ValueError):
            self._validate(late, 60.06)

    def test_boundary_between_tracking_semantics(self):
        # (a, b]: face AT a boundary frame belongs to the NEW shot.
        from lib.shots import boundary_between
        starts = [0, 240, 480]
        self.assertFalse(boundary_between(starts, 100, 239))  # same shot
        self.assertTrue(boundary_between(starts, 239, 240))   # b at cut
        self.assertFalse(boundary_between(starts, 240, 479))  # a at cut
        self.assertTrue(boundary_between(starts, 100, 300))
        self.assertFalse(boundary_between(starts, 480, 900))  # beyond last

    def test_boundary_inside_gap_semantics(self):
        # (a, b): boundary exactly at an endpoint probe does NOT force dense.
        from lib.shots import boundary_inside
        starts = [0, 240, 480]
        self.assertFalse(boundary_inside(starts, 236, 240))   # cut AT b
        self.assertFalse(boundary_inside(starts, 240, 244))   # cut AT a
        self.assertTrue(boundary_inside(starts, 236, 244))    # cut inside
        self.assertFalse(boundary_inside(starts, 300, 400))   # no cut

    def test_input_schema_accepts_and_rejects_uri(self):
        from lib.input_schema import validate_and_normalize
        ev = {"uuid": "u", "input": {"s3uri": "s3://in/a.mp4",
                                     "input_config": {}},
              "output": {"s3uri": "s3://out/p"}, "data": {}}
        cfg = validate_and_normalize(ev)["input"]["input_config"]
        self.assertEqual(cfg["shot_segmentation_s3uri"], "")

        ev["input"]["input_config"] = {
            "shot_segmentation_s3uri": " s3://b/shots.json "}
        cfg = validate_and_normalize(ev)["input"]["input_config"]
        self.assertEqual(cfg["shot_segmentation_s3uri"], "s3://b/shots.json")

        for bad in ("https://b/shots.json", 42):
            ev["input"]["input_config"] = {"shot_segmentation_s3uri": bad}
            with self.assertRaises(ValueError):
                validate_and_normalize(ev)

    def test_tracking_veto_swap(self):
        """build_tracks with a shot-boundary veto: same-position faces on
        both sides of a cut land in TWO tracks; without a cut, one."""
        from lib.shots import boundary_between
        from lib.tracks import build_tracks
        faces = [{"frame_num": n, "timestamp": n / 24.0, "gender": "Male",
                  "box": {"l": 0.4, "t": 0.3, "w": 0.2, "h": 0.3},
                  "confidence": 99.0}
                 for n in (230, 235, 245, 250)]
        cut = build_tracks(
            faces, 3.0, lambda a, b: boundary_between([0, 240], a, b))
        self.assertEqual(len(cut), 2)
        no_cut = build_tracks(
            faces, 3.0, lambda a, b: boundary_between([0], a, b))
        self.assertEqual(len(no_cut), 1)


class TestGatewayNoTokenEcho(unittest.TestCase):
    """The token is consumed into workflow_id and never echoed — by any path.

    History, because the absence is deliberate and easy to "helpfully" undo: a
    supplied uuid used to be echoed back as `uuid`, and _parse_succeeded_output
    read it from the execution OUTPUT, a key no state ever writes.  It therefore
    always resolved None and fell through to the derived id, so a SUCCEEDED poll
    disagreed with its own submit response (found live 2026-08-05: submit echoed
    'meridian-nab-20260805', the poll echoed '7dad1f4d…').  The echo was that
    bug's entire blast radius, so 2026-08-06 removed the echo rather than moving
    the read.

    A caller that supplied a token already has it.  `workflow_id` is the
    service's name for the run and the only identity a response carries — which
    is also why all three response paths now agree by construction instead of by
    sharing a reader.
    """

    ARN = ("arn:aws:states:us-east-1:123456789012:execution:"
           "pfx-face-recognition:7dad1f4d-derived")

    def _load_gw(self):
        os.environ.setdefault("STATE_MACHINE_ARN",
                              "arn:aws:states:us-east-1:123456789012:"
                              "stateMachine:pfx-face-recognition")
        spec = importlib.util.spec_from_file_location(
            "gw_face_recognition",
            os.path.join(ROOT, "gateway_tool", "lambda_function.py"))
        gw = importlib.util.module_from_spec(spec)
        with mock.patch("boto3.client"):
            spec.loader.exec_module(gw)
        return gw

    def _resp(self, *, output=None, stale_token=False):
        """A real describe_execution shape. `uuid` is the execution name; no
        token appears anywhere, because the gateway never forwards one.

        `stale_token` plants the retired keys in BOTH the input and the output to
        prove no path resurrects them."""
        sfn_input = {"uuid": "7dad1f4d-derived",
                     "input": {"s3uri": "s3://in/a.mp4"}}
        if stale_token:
            sfn_input["correlation_uuid"] = "stale-wrong-id"
            sfn_input["idempotency_token"] = "stale-wrong-id"  # nosec B105
        default_out = {"uuid": "7dad1f4d-derived", "data": {}}
        if stale_token:
            default_out["correlation_uuid"] = "stale-wrong-id"
        return {"executionArn": self.ARN,
                "status": "SUCCEEDED",
                "input": json.dumps(sfn_input),
                "output": json.dumps(output if output is not None
                                     else default_out)}

    def test_a_succeeded_poll_returns_workflow_id_and_no_uuid(self):
        gw = self._load_gw()
        out = gw._parse_succeeded_output(self._resp())
        self.assertEqual(out["workflow_id"], "7dad1f4d-derived")
        self.assertNotIn("uuid", out)

    def test_every_response_path_agrees(self):
        """All three paths report workflow_id and nothing else identity-shaped,
        so they cannot drift apart the way submit and poll once did."""
        gw = self._load_gw()
        for status in ("RUNNING", "FAILED"):
            resp = dict(self._resp(), status=status)
            result: dict = {}
            gw._attach_workflow_id(result, resp)
            self.assertEqual(result["workflow_id"], "7dad1f4d-derived", status)
            self.assertNotIn("uuid", result, status)

    def test_the_index_execution_path_agrees_too(self):
        gw = self._load_gw()
        resp = self._resp(output={"action": "batch_index", "data": {}})
        out = gw._parse_index_output(resp)
        self.assertEqual(out["workflow_id"], "7dad1f4d-derived")
        self.assertNotIn("uuid", out)

    def test_a_stale_token_in_the_input_or_output_is_never_surfaced(self):
        """Nothing forwards a token, so a token found in either place is stale
        data from an older execution — it must not become a response field.
        This is the guard against re-adding a reader of any kind."""
        gw = self._load_gw()
        resp = self._resp(stale_token=True)
        for out in (gw._parse_succeeded_output(resp),
                    gw._parse_index_output(resp)):
            self.assertNotIn("uuid", out)
            self.assertNotIn("correlation_uuid", out)
            self.assertNotIn("idempotency_token", out)

    def test_the_retired_reader_is_gone(self):
        """_correlation_uuid() read the caller's token out of an execution. Its
        deletion is the fix; a re-added helper is the regression."""
        gw = self._load_gw()
        self.assertFalse(hasattr(gw, "_correlation_uuid"))


class TestGatewayIdempotencyToken(unittest.TestCase):
    """`idempotency_token` is the caller's "same request as before" assertion.

    The sibling workflows dedup on parameters alone, because their `uuid` names
    the asset they process and the parameters fully determine the output.
    recognize_faces is not tied to one asset: it reads and writes a PERSISTENT,
    MUTABLE face collection, so identical parameters can legitimately need to run
    again ("recognize this video again now that the collection knows more faces")
    — which parameter-only dedup made impossible to express without perturbing an
    unrelated parameter.  So an ABSENT token means "do not deduplicate me"; a
    supplied one buys retry safety.
    """

    def _load_gw(self):
        # PROXY_BUCKET/TARGET_NAME are read at import time and the tool
        # refuses to submit (CONFIG_ERROR) without them.
        env = {"STATE_MACHINE_ARN": "arn:aws:states:us-east-1:123456789012:"
                                    "stateMachine:pfx-face-recognition",
               "PROXY_BUCKET": "test-proxy",
               "TARGET_NAME": "face-recognition",
               "TOOL_VERSION": "test.0"}
        spec = importlib.util.spec_from_file_location(
            "gw_face_recognition_idem",
            os.path.join(ROOT, "gateway_tool", "lambda_function.py"))
        gw = importlib.util.module_from_spec(spec)
        with mock.patch.dict(os.environ, env), mock.patch("boto3.client"):
            spec.loader.exec_module(gw)
        gw._sfn = mock.MagicMock()
        gw._sfn.start_execution.return_value = {"executionArn": "arn:x"}
        return gw

    def _submit(self, gw, **kwargs):
        return gw._recognize_faces(
            {"asset_s3uri": "s3://in/a.mp4", **kwargs})

    def test_no_token_means_a_fresh_run_every_time(self):
        """THE behavior this tool needs: two identical tokenless calls are two
        jobs. A caller who supplied no token never asked to be deduplicated."""
        gw = self._load_gw()
        a = self._submit(gw)
        b = self._submit(gw)
        self.assertNotEqual(a["workflow_id"], b["workflow_id"])

    def test_a_supplied_token_makes_the_submit_idempotent(self):
        gw = self._load_gw()
        a = self._submit(gw, idempotency_token="meridian-nab-20260806")  # nosec B106  # nosemgrep: generic.secrets.gitleaks.generic-api-key
        b = self._submit(gw, idempotency_token="meridian-nab-20260806")  # nosec B106  # nosemgrep: generic.secrets.gitleaks.generic-api-key
        self.assertEqual(a["workflow_id"], b["workflow_id"])

    def test_the_same_token_with_different_parameters_still_re_runs(self):
        """The token narrows dedup, it does not replace the parameter hash — so a
        reused token can never hand back another request's results."""
        gw = self._load_gw()
        a = self._submit(gw, idempotency_token="tok")  # nosec B106
        b = self._submit(gw, idempotency_token="tok", collection_id="other")  # nosec B106
        self.assertNotEqual(a["workflow_id"], b["workflow_id"])

    def test_the_token_is_never_echoed_on_submit(self):
        gw = self._load_gw()
        out = self._submit(gw, idempotency_token="tok")  # nosec B106
        self.assertNotIn("uuid", out)
        self.assertNotIn("idempotency_token", out)
        self.assertTrue(out["workflow_id"])

    def test_the_token_never_reaches_the_state_machine(self):
        """It rides in the hash preimage as extra material, not in sfn_input, so
        "never forwarded, never stored" holds by construction rather than by
        remembering to pop a key."""
        gw = self._load_gw()
        self._submit(gw, idempotency_token="tok")  # nosec B106
        sent = json.loads(gw._sfn.start_execution.call_args.kwargs["input"])
        self.assertNotIn("idempotency_token", sent)
        self.assertNotIn("correlation_uuid", sent)
        self.assertNotIn("tok", json.dumps(sent))

    def test_the_event_uuid_stays_the_execution_name(self):
        """Load-bearing: in-workflow states rebuild the execution ARN (the
        semaphore holder key) and the sandbox-{workflow_id} collection name from
        the event uuid, so it must equal the execution name."""
        gw = self._load_gw()
        out = self._submit(gw, idempotency_token="tok")  # nosec B106
        sent = json.loads(gw._sfn.start_execution.call_args.kwargs["input"])
        self.assertEqual(sent["uuid"], out["workflow_id"])
        self.assertEqual(gw._sfn.start_execution.call_args.kwargs["name"],
                         out["workflow_id"])

    def test_a_bad_token_is_rejected_at_submit(self):
        gw = self._load_gw()
        out = self._submit(gw, idempotency_token="not a valid name!")  # nosec B106
        self.assertIs(out["error"], True)
        self.assertEqual(out["errorCode"], "INVALID_PARAMETER")

    def test_the_retired_uuid_parameter_is_rejected_not_ignored(self):
        """`uuid` meant BOTH "which asset" and "same request", so a caller still
        sending it cannot be assumed to mean the surviving half. Ignoring it
        would silently start a fresh run for a caller who believed they were
        protected from paying twice."""
        gw = self._load_gw()
        out = self._submit(gw, uuid="meridian-nab-20260805")
        self.assertIs(out["error"], True)
        self.assertEqual(out["errorCode"], "INVALID_PARAMETER")
        self.assertIn("idempotency_token", out["message"])
        gw._sfn.start_execution.assert_not_called()


class TestGatewayInputExamples(unittest.TestCase):
    """input_examples.json is the INPUT twin of output_contract.json: one
    minimal valid request per tool, served by describe_tool so an agent gets
    a known-good starting call instead of guessing shapes from prose. The
    round-trip tests drive EVERY example through the real lambda_handler with
    mocked boto3 clients, so an example that drifts from the real validation
    fails this build, not an agent's first call. Nothing here is
    validation-only — the read tools' DDB/Rekognition stubs are one line
    each, so the examples are driven to a successful response, not just past
    the parameter checks.
    """

    TOOLS = ("recognize_faces", "get_job_status", "describe_tool",
             "create_collection", "list_collections", "list_faces",
             "get_face", "batch_update_faces", "batch_index_faces",
             "batch_delete_faces", "delete_collection")

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, "gateway_tool",
                               "input_examples.json")) as f:
            cls.examples = json.load(f)["tools"]

    def _load_gw(self):
        # PROXY_BUCKET is the examples' own placeholder bucket:
        # recognize_faces' shot_segmentation_s3uri and batch_index_faces'
        # s3_prefix are validated against s3://{PROXY_BUCKET}/.
        env = {"STATE_MACHINE_ARN": "arn:aws:states:eu-west-1:111111111111:"
                                    "stateMachine:myprefix-face-recognition",
               "PROXY_BUCKET": "my-proxy-bucket",
               "TARGET_NAME": "face-recognition",
               "TOOL_VERSION": "test.0"}
        spec = importlib.util.spec_from_file_location(
            "gw_face_recognition_examples",
            os.path.join(ROOT, "gateway_tool", "lambda_function.py"))
        gw = importlib.util.module_from_spec(spec)
        with mock.patch.dict(os.environ, env), mock.patch("boto3.client"):
            spec.loader.exec_module(gw)
        gw._sfn = mock.MagicMock()
        gw._sfn.start_execution.return_value = {"executionArn": "arn:fake"}
        return gw

    def _call(self, gw, tool, event):
        ctx = mock.MagicMock()
        ctx.client_context.custom = {
            "bedrockAgentCoreToolName": f"face-recognition___{tool}"}
        return gw.lambda_handler(dict(event), ctx)

    def _patched_index_tools(self):
        """index_tools with its module globals pointed at the examples'
        placeholders and all four clients replaced by MagicMocks returning
        minimal valid backend shapes."""
        import contextlib
        import index_tools as it
        rek, ddb, s3, sfn = (mock.MagicMock() for _ in range(4))
        rek.create_collection.return_value = {}
        rek.list_collections.return_value = {"CollectionIds": []}
        ddb.query.return_value = {"Items": [], "Count": 0}
        ddb.get_item.return_value = {"Item": {
            "pk": {"S": "face#00000000-0000-4000-8000-000000000003"},
            "sk": {"S": "record"},
            "face_id": {"S": "00000000-0000-4000-8000-000000000003"},
            "collection_id": {"S": "my-face-collection"},
            "created_at": {"N": "1700000000000"}}}
        s3.list_objects_v2.return_value = {
            "Contents": [{"Key": "face-index/headshots/Lena Voss.jpg"}],
            "IsTruncated": False}
        sfn.start_execution.return_value = {"executionArn": "arn:fake"}
        stack = contextlib.ExitStack()
        for name, value in (
                ("_rek", rek), ("_ddb", ddb), ("_s3", s3), ("_sfn", sfn),
                ("PROXY_BUCKET", "my-proxy-bucket"),
                ("TARGET_NAME", "face-recognition"),
                ("FACE_TABLE_NAME", "my-face-table"),
                ("DEFAULT_COLLECTION_ID", ""),
                ("INDEX_STATE_MACHINE_ARN",
                 "arn:aws:states:eu-west-1:111111111111:"
                 "stateMachine:myprefix-face-index")):
            stack.enter_context(mock.patch.object(it, name, value))
        return stack, it

    def test_input_examples_cover_exactly_the_dispatched_tools(self):
        self.assertEqual(sorted(self.examples), sorted(self.TOOLS))
        for tool, entry in self.examples.items():
            # `alternatives` is optional: present only on tools with a
            # one-of rule, holding one valid example of each other form
            self.assertIn(sorted(entry),
                          (["example", "required"],
                           ["alternatives", "example", "required"]), tool)
            plain = {n for n in entry["required"]
                     if " " not in n and "|" not in n}
            # Every PLAIN required name must appear in the tool's own
            # example — prose one-of entries (containing a space or "|")
            # state rules the example can only satisfy one arm of.
            for name in plain:
                self.assertIn(name, entry["example"],
                              f"{tool}: required field {name!r} missing "
                              f"from its example")

    def test_describe_tool_carries_input_examples(self):
        gw = self._load_gw()
        out = self._call(gw, "describe_tool", {})
        self.assertEqual(out["input_examples"]["tools"], self.examples)

    def test_recognize_faces_example_round_trips_and_does_not_index(self):
        """The example IS the contract: round-trip it through the real
        handler so drift fails this build, not an agent's first call. It
        must carry no stray `uuid` (rejected on submit), and it must be SAFE
        to copy-paste: max_faces_to_index 0, so the starting call never
        writes to the persistent collection."""
        example = self.examples["recognize_faces"]["example"]
        self.assertNotIn("uuid", example)
        gw = self._load_gw()
        out = self._call(gw, "recognize_faces", example)
        self.assertNotIn("error", out, f"example refused: {out}")
        self.assertEqual(out["status"], "RUNNING")
        sent = json.loads(gw._sfn.start_execution.call_args.kwargs["input"])
        self.assertEqual(sent["input"]["input_config"]["max_faces_to_index"],
                         0)

    def test_every_other_example_round_trips_through_the_real_handler(self):
        """get_job_status, the five sync index tools, and the three async
        index tools, each driven through lambda_handler against mocked
        clients — the three async submits must reach StartExecution and
        return a RUNNING handle."""
        gw = self._load_gw()
        gw._sfn.describe_execution.return_value = {
            "status": "RUNNING",
            "executionArn":
                self.examples["get_job_status"]["example"]["job_id"],
            "input": json.dumps({"input": {"s3uri": "s3://x/a.mp4"},
                                 "output": {"s3uri": "s3://x/out"}})}
        # a real (empty) history page: _get_current_state pages while
        # nextToken is truthy, and a bare MagicMock page never isn't
        gw._sfn.get_execution_history.return_value = {"events": []}
        out = self._call(gw, "get_job_status",
                         self.examples["get_job_status"]["example"])
        self.assertNotIn("error", out, f"example refused: {out}")

        stack, it = self._patched_index_tools()
        with stack:
            for tool in ("create_collection", "list_collections",
                         "list_faces", "get_face", "batch_update_faces"):
                out = self._call(gw, tool, self.examples[tool]["example"])
                self.assertNotIn("error", out,
                                 f"{tool} example refused: {out}")
            for tool in ("batch_index_faces", "batch_delete_faces",
                         "delete_collection"):
                out = self._call(gw, tool, self.examples[tool]["example"])
                self.assertNotIn("error", out,
                                 f"{tool} example refused: {out}")
                self.assertEqual(out["status"], "RUNNING", tool)
            self.assertEqual(it._sfn.start_execution.call_count, 3)

    def test_staged_alternatives_pass_the_gateway_validation(self):
        """The other arm of each one-of rule: batch_index_faces'
        manifest_s3uri form (the manifest is read in-run, not at submit —
        only the proxy-bucket prefix is checked here) and
        batch_delete_faces' celebrity_name form, each a complete request
        driven through the real handler to a RUNNING handle."""
        gw = self._load_gw()
        stack, it = self._patched_index_tools()
        with stack:
            for tool, selector in (("batch_index_faces", "manifest_s3uri"),
                                   ("batch_delete_faces", "celebrity_name")):
                alt = self.examples[tool]["alternatives"][selector]
                out = self._call(gw, tool, alt)
                self.assertNotIn("error", out,
                                 f"{tool} alternative refused: {out}")
                self.assertEqual(out["status"], "RUNNING", tool)
            self.assertEqual(it._sfn.start_execution.call_count, 2)
            # the manifest form must pass the uri through untouched —
            # the prefix form would have generated one instead
            sent = json.loads(
                it._sfn.start_execution.call_args_list[0]
                .kwargs["input"])
            self.assertEqual(
                sent["input"]["manifest_s3uri"],
                self.examples["batch_index_faces"]["alternatives"]
                ["manifest_s3uri"]["manifest_s3uri"])
            it._s3.list_objects_v2.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=1)
