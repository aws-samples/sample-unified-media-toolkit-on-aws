"""
OFFLINE unit tests for the archive-tier guard in lib/s3_utils.py.
No S3 calls — archive_tier is pure, it reads a HeadObject dict.

Run: python3 test/test_s3_utils_units.py
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.s3_utils import archive_tier  # noqa: E402



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


class TestArchiveTier(unittest.TestCase):
    def test_readable(self):
        for head in (
            {},
            {"StorageClass": "STANDARD"},
            {"StorageClass": "STANDARD_IA"},
            {"StorageClass": "INTELLIGENT_TIERING"},          # frequent/infrequent tier
        ):
            self.assertIsNone(archive_tier(head), head)

    def test_glacier_ir_is_readable(self):
        # GLACIER_IR reads like STANDARD — blocking it would reject valid sources.
        self.assertIsNone(archive_tier({"StorageClass": "GLACIER_IR"}))

    def test_intelligent_tiering_archive_tiers(self):
        # The live failure: StorageClass stays INTELLIGENT_TIERING, so the tier
        # is only visible through ArchiveStatus.
        for status in ("ARCHIVE_ACCESS", "DEEP_ARCHIVE_ACCESS"):
            self.assertEqual(
                archive_tier({"StorageClass": "INTELLIGENT_TIERING",
                              "ArchiveStatus": status}), status)

    def test_glacier_storage_classes(self):
        for cls in ("GLACIER", "DEEP_ARCHIVE"):
            self.assertEqual(archive_tier({"StorageClass": cls}), cls)

    def test_restore_in_progress_still_blocked(self):
        self.assertEqual(
            archive_tier({"StorageClass": "INTELLIGENT_TIERING",
                          "ArchiveStatus": "ARCHIVE_ACCESS",
                          "Restore": 'ongoing-request="true"'}),
            "ARCHIVE_ACCESS")

    def test_completed_restore_clears_the_block(self):
        self.assertIsNone(archive_tier({
            "StorageClass": "GLACIER",
            "Restore": 'ongoing-request="false", '
                       'expiry-date="Fri, 12 Sep 2026 00:00:00 GMT"',
        }))


if __name__ == "__main__":
    unittest.main(verbosity=2)
