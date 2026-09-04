#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE = Path(__file__).with_name("store_robotruck_occ_gridfs.py")
SPEC = importlib.util.spec_from_file_location("occ_content_store_tested", MODULE)
STORE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(STORE)


class ContentAddressedOccStoreTest(unittest.TestCase):
    def test_store_is_content_addressed_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "occ_ijk.i32.bin"
            source.write_bytes(b"same OCC payload")
            first = STORE.store(source, root / "assets")
            second = STORE.store(source, root / "assets")
            self.assertEqual(first, second)
            self.assertEqual(first["storage"], "content_addressed_file")
            self.assertEqual(first["sha256"], STORE.sha256(source))
            self.assertEqual(Path(first["uri"]).read_bytes(), source.read_bytes())
            self.assertNotIn("gridfs_id", first)


if __name__ == "__main__":
    unittest.main()
