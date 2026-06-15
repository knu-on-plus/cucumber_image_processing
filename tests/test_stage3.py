from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from occlusion_pipeline.stage3 import (
    CONDITIONS,
    _layout_specs_for_mode,
    _mark_stage3_complete,
    validate_synthesis,
)


OUTPUT_ROOT = (
    Path(__file__).resolve().parents[1] / "outputs" / "quickstart" / "synthesis"
)


class Stage3OutputTest(unittest.TestCase):
    def setUp(self) -> None:
        if not OUTPUT_ROOT.exists():
            self.skipTest("Generated Stage 3 outputs are not available")

    def test_representative_output_contract(self) -> None:
        report = validate_synthesis(OUTPUT_ROOT)
        self.assertTrue(report["valid"], report["issues"])
        self.assertEqual(12, report["layout_count"])
        self.assertEqual(36, report["image_count"])
        self.assertEqual(90, report["annotation_count"])

    def test_all_conditions_and_gamma_values_are_present(self) -> None:
        instances = json.loads(
            (OUTPUT_ROOT / "annotations" / "instances.json").read_text()
        )
        conditions = {image["condition"] for image in instances["images"]}
        gammas = Counter(image["gamma"] for image in instances["images"])
        self.assertEqual(set(CONDITIONS), conditions)
        self.assertEqual({0.3: 12, 1.0: 12, 3.0: 12}, dict(gammas))

    def test_cucumber_annotations_follow_aisformer_contract(self) -> None:
        instances = json.loads(
            (OUTPUT_ROOT / "annotations" / "instances.json").read_text()
        )
        cucumbers = [
            annotation
            for annotation in instances["annotations"]
            if annotation["category_id"] == 1
        ]
        self.assertEqual(36, len(cucumbers))
        for annotation in cucumbers:
            self.assertIn("segmentation", annotation)
            self.assertIn("bbox", annotation)
            self.assertIn("inmodal_seg", annotation)
            self.assertIn("inmodal_bbox", annotation)
            self.assertEqual(
                annotation["amodal_area"],
                annotation["inmodal_area"] + annotation["occluded_area"],
            )


class Stage3ManifestTest(unittest.TestCase):
    def test_external_output_root_is_recorded_as_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as dataset_directory:
            with tempfile.TemporaryDirectory() as output_directory:
                dataset_root = Path(dataset_directory)
                output_root = Path(output_directory)
                manifest_path = dataset_root / "manifest.json"
                manifest_path.write_text("{}\n", encoding="utf-8")
                _mark_stage3_complete(dataset_root, output_root, 30, 90)
                manifest = json.loads(
                    (output_root / "manifest.json").read_text(encoding="utf-8")
                )
        self.assertEqual(str(output_root), manifest["stage3"]["output_root"])


class Stage3SamplingPlanTest(unittest.TestCase):
    def test_weighted_sample_plan_matches_five_four_one_per_condition(self) -> None:
        source_images = [{"id": index + 1} for index in range(3)]
        specs = _layout_specs_for_mode(
            source_images=source_images,
            mode="weighted-sample",
            conditions=CONDITIONS[1:],
            ratios=(0.5, 0.75, 0.9),
            baseline_ratio=0.5,
            ratio_weights=(5, 4, 1),
            sample_count=None,
            samples_per_condition=10,
        )
        self.assertEqual(30, len(specs))
        by_condition = {
            condition: Counter(
                spec["requested_ratio"]
                for spec in specs
                if spec["condition"] == condition
            )
            for condition in CONDITIONS[1:]
        }
        for counts in by_condition.values():
            self.assertEqual({0.5: 5, 0.75: 4, 0.9: 1}, dict(counts))


if __name__ == "__main__":
    unittest.main()
