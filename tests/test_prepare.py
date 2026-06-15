from __future__ import annotations

import unittest
from pathlib import Path

from occlusion_pipeline.image_metadata import (
    read_jpeg_metadata,
    read_png_alpha_metadata,
)
from occlusion_pipeline.prepare import DEFAULT_SAMPLE_IDS, validate_prepared_dataset
from occlusion_pipeline.stage2 import check_stage2_environment
from occlusion_pipeline.stage2 import validate_cucumber_masks


SAMPLE_ROOT = Path(__file__).resolve().parents[1] / "sample_data"


class ImageMetadataTest(unittest.TestCase):
    def test_selected_images_match_portrait_annotation_coordinates(self) -> None:
        if not SAMPLE_ROOT.exists():
            self.skipTest("Local sample data is not available")
        for source_id in DEFAULT_SAMPLE_IDS:
            metadata = read_jpeg_metadata(
                SAMPLE_ROOT / "raw_cucumber_images" / f"{source_id}.jpg"
            )
            self.assertEqual((3000, 4000), metadata.display_size)
            self.assertEqual(6, metadata.exif_orientation)

    def test_leaf_samples_have_usable_alpha(self) -> None:
        leaves = sorted((SAMPLE_ROOT / "cropped_leaves_samples").glob("*.png"))
        if not leaves:
            self.skipTest("Local leaf samples are not available")
        for leaf in leaves:
            metadata = read_png_alpha_metadata(leaf)
            self.assertTrue(metadata.has_alpha)
            self.assertGreater(metadata.nonzero_fraction, 0)
            self.assertLess(metadata.nonzero_fraction, 1)


class PreparedDatasetTest(unittest.TestCase):
    def test_generated_quickstart_contract(self) -> None:
        dataset_root = SAMPLE_ROOT / "quickstart"
        if not dataset_root.exists():
            self.skipTest("Generated quickstart data is not available")
        report = validate_prepared_dataset(dataset_root)
        self.assertTrue(report["valid"], report["issues"])
        self.assertEqual(3, report["image_count"])
        self.assertEqual(3, report["cucumber_bbox_count"])
        self.assertGreaterEqual(report["leaf_cutout_count"], 2)

    def test_bbox_previews_are_self_contained(self) -> None:
        preview = SAMPLE_ROOT / "quickstart" / "previews" / "bbox_scene_001.svg"
        if not preview.exists():
            self.skipTest("Generated quickstart data is not available")
        self.assertIn("data:image/jpeg;base64,", preview.read_text(encoding="utf-8"))


class Stage2EnvironmentTest(unittest.TestCase):
    def test_environment_report_has_required_modules(self) -> None:
        report = check_stage2_environment()
        self.assertEqual({"numpy", "PIL", "torch", "sam2"}, set(report["modules"]))

    def test_generated_cucumber_masks(self) -> None:
        manifest = (
            Path(__file__).resolve().parents[1]
            / "outputs"
            / "quickstart"
            / "stage2"
            / "annotations"
            / "cucumber_masks.json"
        )
        environment = check_stage2_environment()
        if not manifest.exists() or not environment["modules"]["PIL"]:
            self.skipTest("Generated masks or Pillow are not available")
        report = validate_cucumber_masks(SAMPLE_ROOT / "quickstart")
        self.assertTrue(report["valid"], report["issues"])
        self.assertEqual(3, report["mask_count"])


if __name__ == "__main__":
    unittest.main()
