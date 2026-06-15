from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from occlusion_pipeline.configuration import load_synthesis_settings
from occlusion_pipeline.stage3 import CONDITIONS


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class SynthesisConfigurationTest(unittest.TestCase):
    def test_quickstart_preset_exposes_all_paper_conditions(self) -> None:
        settings = load_synthesis_settings(REPOSITORY_ROOT / "configs/quickstart.json")
        self.assertEqual(CONDITIONS, settings.conditions)
        self.assertEqual("representative", settings.mode)
        self.assertEqual((0.5, 0.75, 0.9), settings.ratios)
        self.assertEqual((0.3, 1.0, 3.0), settings.gammas)
        self.assertEqual(
            0.38,
            settings.condition_parameters[CONDITIONS[3]][
                "attachment_overlap_ratio"
            ],
        )

    def test_weighted_preset_uses_exact_paper_ratio_weights(self) -> None:
        settings = load_synthesis_settings(
            REPOSITORY_ROOT / "configs/paper-weighted.json"
        )
        self.assertEqual("weighted-sample", settings.mode)
        self.assertEqual(CONDITIONS[1:], settings.conditions)
        self.assertEqual((5, 4, 1), settings.ratio_weights)
        self.assertEqual(10, settings.samples_per_condition)

    def test_condition_subset_uses_default_tuning_parameters(self) -> None:
        config = {
            "dataset_root": "sample_data/quickstart",
            "synthesis": {
                "conditions": [CONDITIONS[1]],
                "occlusion_ratios": [0.9],
                "gamma_values": [1.0],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "subset.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            settings = load_synthesis_settings(path)
        self.assertEqual((CONDITIONS[1],), settings.conditions)
        self.assertEqual(120, settings.condition_parameters[CONDITIONS[1]]["search_steps"])

    def test_invalid_condition_is_rejected(self) -> None:
        config = {
            "synthesis": {
                "conditions": ["unknown"],
                "occlusion_ratios": [0.5],
                "gamma_values": [1.0],
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown synthesis conditions"):
                load_synthesis_settings(path)

    def test_weighted_mode_rejects_baseline_condition(self) -> None:
        config = {
            "synthesis": {
                "mode": "weighted-sample",
                "conditions": [CONDITIONS[0], CONDITIONS[1]],
                "occlusion_ratios": [0.5, 0.75, 0.9],
                "ratio_sampling_weights": [5, 4, 1],
                "samples_per_condition": 10,
                "gamma_values": [1.0],
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-weighted.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "baseline_single_center"):
                load_synthesis_settings(path)


if __name__ == "__main__":
    unittest.main()
