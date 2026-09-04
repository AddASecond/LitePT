#!/usr/bin/env python3
"""OCC unit tests."""
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from paths import ensure_import_path

ensure_import_path()
import gss_mongo as GSS
import store as STORE


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


class GssOccMongoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = {
            "md5": "a" * 32,
            "timestamp": 123,
            "clip_id": "clip-1",
            "source": {
                "db": "perception_experiment",
                "frame_collection": "raw_data_frames_fp_matrix",
                "clip_collection": "raw_data_clips_fp_matrix",
                "raw_id": "raw-id",
                "frame_md5": "a" * 32,
            },
            "grid": {"voxel": 0.2, "origin": [-30, -200, -5], "shape": [300, 3000, 125]},
            "stats": {"n_occ": 2},
            "assets": {
                "occupancy": {
                    "ijk": {
                        "storage": "content_addressed_file",
                        "uri": "/data/rawdata-4/occupancy/aa/bb/blob.bin",
                        "sha256": "b" * 64,
                        "dtype": "int32",
                        "shape": [2, 3],
                    }
                }
            },
        }

    def test_collection_name_preserves_dataset_suffix(self) -> None:
        self.assertEqual(
            GSS.groundtruth_collection_name("raw_data_frames_fp_matrix"),
            "occ_data_groundtruths_fp_matrix",
        )
        self.assertEqual(
            GSS.groundtruth_collection_name("raw_data_frames"),
            "occ_data_groundtruths",
        )

    def test_frame_keeps_raw_trace_and_content_reference(self) -> None:
        result = GSS.build_gss_frame(self.frame)
        self.assertEqual(result["raw_data"]["frame_collection"], "raw_data_frames_fp_matrix")
        self.assertEqual(result["raw_data"]["document_id"], "raw-id")
        self.assertEqual(result["occupancy"]["voxel_count"], 2)
        self.assertEqual(
            result["occupancy"]["assets"]["ijk"]["storage"],
            "content_addressed_file",
        )

    def test_document_matches_gss_clips_frames_layout(self) -> None:
        clip = GSS.build_gss_clip({"clip_id": "clip-1", "bag_name": "x.bag"}, [self.frame])
        document = GSS.build_gss_document(
            tag="test",
            version="v1",
            run_id="run-1",
            clips=[clip],
            producer={"name": "LitePT"},
            timestamp=datetime.now(timezone.utc),
        )
        self.assertEqual(document["schema_version"], "gss_occ_groundtruth/v1")
        self.assertEqual(document["clip_count"], 1)
        self.assertEqual(document["frame_count"], 1)
        self.assertEqual(document["clips"][0]["frames"][0]["md5"], "a" * 32)

    def test_rejects_frame_without_occupancy_assets(self) -> None:
        with self.assertRaisesRegex(ValueError, "assets.occupancy"):
            GSS.build_gss_frame({"assets": {}})


if __name__ == "__main__":
    unittest.main()
