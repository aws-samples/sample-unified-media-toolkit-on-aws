"""Regression: the defusedxml swap (2026-10-08) broke module IMPORT —
defusedxml.ElementTree exports parse functions, not the Element class,
and the annotation at module level evaluated at import time. The
pipeline tests are *_test.py (outside the discover pattern), so nothing
gated the import. This file is in-pattern and does."""
import sys
import unittest


class TestMxfUtilsImports(unittest.TestCase):
    def test_module_imports_and_parses(self):
        sys.path.insert(0, ".")
        try:
            from lib import mxf_utils
            root = mxf_utils.ET.fromstring("<BMX><a>1</a></BMX>")
            self.assertEqual("BMX", root.tag)
        finally:
            sys.path.remove(".")

    def test_dtd_is_refused(self):
        sys.path.insert(0, ".")
        try:
            from lib import mxf_utils
            with self.assertRaises(Exception):
                mxf_utils.ET.fromstring(
                    '<?xml version="1.0"?><!DOCTYPE a [<!ENTITY b "x">]><a>&b;</a>')
        finally:
            sys.path.remove(".")


if __name__ == "__main__":
    unittest.main()
