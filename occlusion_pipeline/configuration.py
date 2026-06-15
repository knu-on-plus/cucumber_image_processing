"""Validated configuration loading for the synthesis pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .stage3 import CONDITIONS, DEFAULT_CONDITION_PARAMETERS


@dataclass(frozen=True)
class SynthesisSettings:
    name: str
    dataset_root: Path
    mask_root: Path
    output_root: Path
    conditions: tuple[str, ...]
    ratios: tuple[float, ...]
    gammas: tuple[float, ...]
    baseline_ratio: float
    ratio_weights: tuple[int, ...]
    sample_count: int | None
    samples_per_condition: int | None
    target_size: tuple[int, int]
    tolerance: float
    seed: int
    mode: str
    condition_parameters: dict[str, dict[str, Any]]


def load_synthesis_settings(config_path: Path) -> SynthesisSettings:
    """Load and validate a synthesis preset."""
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Synthesis config is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    synthesis = config.get("synthesis") or config.get("stages", {}).get("synthesis")
    if not isinstance(synthesis, dict):
        raise ValueError("Config must contain a synthesis object.")

    dataset_root = Path(config.get("dataset_root", "sample_data/quickstart"))
    mask_root = Path(config.get("mask_root", "outputs/quickstart/stage2"))
    output_root = Path(
        config.get("output_root", "outputs/quickstart/synthesis")
    )
    conditions = tuple(synthesis.get("conditions", CONDITIONS))
    ratios = tuple(float(value) for value in synthesis.get("occlusion_ratios", ()))
    gammas = tuple(float(value) for value in synthesis.get("gamma_values", ()))
    ratio_weights = tuple(
        int(value) for value in synthesis.get("ratio_sampling_weights", (5, 4, 1))
    )
    target_size_values = synthesis.get("target_size", (768, 1024))
    if len(target_size_values) != 2:
        raise ValueError("target_size must contain [width, height].")

    settings = SynthesisSettings(
        name=str(config.get("name", config_path.stem)),
        dataset_root=dataset_root,
        mask_root=mask_root,
        output_root=output_root,
        conditions=conditions,
        ratios=ratios,
        gammas=gammas,
        baseline_ratio=float(synthesis.get("baseline_occlusion_ratio", 0.5)),
        ratio_weights=ratio_weights,
        sample_count=_optional_positive_int(synthesis.get("sample_count")),
        samples_per_condition=_optional_positive_int(
            synthesis.get("samples_per_condition")
        ),
        target_size=(int(target_size_values[0]), int(target_size_values[1])),
        tolerance=float(synthesis.get("occlusion_tolerance", 0.05)),
        seed=int(synthesis.get("seed", 0)),
        mode=str(
            synthesis.get("mode", synthesis.get("quickstart_mode", "representative"))
        ),
        condition_parameters=_merge_condition_parameters(
            synthesis.get("condition_parameters", {})
        ),
    )
    _validate_settings(settings)
    return settings


def _merge_condition_parameters(
    overrides: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not isinstance(overrides, dict):
        raise ValueError("condition_parameters must be an object.")
    merged: dict[str, dict[str, Any]] = {}
    for condition, defaults in DEFAULT_CONDITION_PARAMETERS.items():
        values = overrides.get(condition, {})
        if not isinstance(values, dict):
            raise ValueError(f"Parameters for {condition} must be an object.")
        merged[condition] = {**defaults, **values}
    unknown = set(overrides) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"Unknown condition parameters: {sorted(unknown)}")
    return merged


def _validate_settings(settings: SynthesisSettings) -> None:
    unknown = set(settings.conditions) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"Unknown synthesis conditions: {sorted(unknown)}")
    if not settings.conditions or len(set(settings.conditions)) != len(
        settings.conditions
    ):
        raise ValueError("conditions must be non-empty and unique.")
    if settings.mode not in {"representative", "full-grid", "weighted-sample"}:
        raise ValueError("mode must be representative, full-grid, or weighted-sample.")
    if not settings.ratios or any(not 0 < ratio < 1 for ratio in settings.ratios):
        raise ValueError("occlusion_ratios must contain values between 0 and 1.")
    if not settings.gammas or any(gamma <= 0 for gamma in settings.gammas):
        raise ValueError("gamma_values must contain positive values.")
    if not 0 < settings.baseline_ratio < 1:
        raise ValueError("baseline_occlusion_ratio must be between 0 and 1.")
    if any(dimension <= 0 for dimension in settings.target_size):
        raise ValueError("target_size dimensions must be positive.")
    if not 0 <= settings.tolerance < 1:
        raise ValueError("occlusion_tolerance must be between 0 and 1.")
    if settings.mode == "weighted-sample":
        if len(settings.ratio_weights) != len(settings.ratios):
            raise ValueError("ratio_sampling_weights must align with occlusion_ratios.")
        if any(weight <= 0 for weight in settings.ratio_weights):
            raise ValueError("ratio_sampling_weights must contain positive integers.")
        if CONDITIONS[0] in settings.conditions:
            raise ValueError("weighted-sample does not accept baseline_single_center.")
        if settings.sample_count is not None and settings.samples_per_condition is not None:
            raise ValueError("Use either sample_count or samples_per_condition, not both.")
        if settings.sample_count is None and settings.samples_per_condition is None:
            raise ValueError(
                "weighted-sample requires sample_count or samples_per_condition."
            )
        weight_total = sum(settings.ratio_weights)
        if settings.samples_per_condition is not None:
            if settings.samples_per_condition % weight_total != 0:
                raise ValueError(
                    "samples_per_condition must be a multiple of "
                    f"sum(ratio_sampling_weights)={weight_total}."
                )
        elif settings.sample_count is not None:
            divisor = len(settings.conditions) * weight_total
            if settings.sample_count % divisor != 0:
                raise ValueError(
                    "sample_count must be a multiple of "
                    f"len(conditions) * sum(ratio_sampling_weights)={divisor}."
                )

    for condition in settings.conditions:
        values = settings.condition_parameters[condition]
        if int(values["search_steps"]) < 2:
            raise ValueError(f"{condition}.search_steps must be at least 2.")
        if condition in {CONDITIONS[0], CONDITIONS[1], CONDITIONS[3]}:
            _validate_pair(values["leaf_height_range"], f"{condition}.leaf_height_range")
        if condition in {CONDITIONS[1], CONDITIONS[3]}:
            positions = values["positions"]
            fractions = values["position_fractions"]
            if not positions or len(positions) != len(fractions):
                raise ValueError(
                    f"{condition} positions and position_fractions must align."
                )
        if condition == CONDITIONS[2]:
            _validate_pair(
                values["outer_fraction_range"],
                f"{condition}.outer_fraction_range",
            )
            if float(values["leaf_height_ratio"]) <= 0:
                raise ValueError(f"{condition}.leaf_height_ratio must be positive.")
            overlap = float(values["max_leaf_overlap_ratio"])
            if not 0 <= overlap < 1:
                raise ValueError(
                    f"{condition}.max_leaf_overlap_ratio must be between 0 and 1."
                )
        if condition == CONDITIONS[3]:
            overlap = float(values["attachment_overlap_ratio"])
            if not 0 <= overlap < 1:
                raise ValueError(
                    f"{condition}.attachment_overlap_ratio must be between 0 and 1."
                )


def _validate_pair(value: Any, name: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain two values.")
    if float(value[0]) >= float(value[1]):
        raise ValueError(f"{name} must be ordered from minimum to maximum.")


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    integer = int(value)
    if integer <= 0:
        raise ValueError("Sample counts must be positive integers.")
    return integer
