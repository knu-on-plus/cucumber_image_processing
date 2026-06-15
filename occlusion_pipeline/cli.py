from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .configuration import load_synthesis_settings
from .prepare import DEFAULT_SAMPLE_IDS, prepare_sample, validate_prepared_dataset
from .stage2 import (
    check_stage2_environment,
    segment_cucumbers,
    validate_cucumber_masks,
)
from .stage3 import synthesize_quickstart, validate_synthesis


def parse_size(value: str) -> tuple[int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Size must be WIDTHxHEIGHT")
    return int(parts[0]), int(parts[1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Condition-based synthetic occlusion dataset pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-sample",
        help="Normalize source images, detection labels, and leaf cutouts",
    )
    prepare.add_argument("--source-root", type=Path, default=Path("sample_data"))
    prepare.add_argument(
        "--output-root", type=Path, default=Path("sample_data/quickstart")
    )
    prepare.add_argument(
        "--sample-id",
        action="append",
        dest="sample_ids",
        help="Source ID to include. Repeat for multiple IDs; defaults to the three examples.",
    )
    prepare.add_argument(
        "--leaf-limit",
        type=int,
        default=0,
        help="Maximum number of leaf cutouts to copy; 0 includes all.",
    )
    prepare.add_argument("--overwrite", action="store_true")

    validate = subparsers.add_parser(
        "validate-inputs", help="Validate normalized Stage 1 outputs"
    )
    validate.add_argument(
        "--dataset-root", type=Path, default=Path("sample_data/quickstart")
    )

    subparsers.add_parser(
        "check-stage2", help="Check whether the SAM2 runtime is available"
    )

    segment = subparsers.add_parser(
        "segment-cucumbers", help="Generate full cucumber masks from bbox prompts with SAM2"
    )
    segment.add_argument(
        "--dataset-root", type=Path, default=Path("sample_data/quickstart")
    )
    segment.add_argument(
        "--output-root", type=Path, default=Path("outputs/quickstart/stage2")
    )
    segment.add_argument("--checkpoint", type=Path, required=True)
    segment.add_argument(
        "--model-config",
        default="sam2_hiera_l.yaml",
        help="SAM2 Hydra model config name",
    )
    segment.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto"
    )
    segment.add_argument("--score-threshold", type=float, default=0.7)
    segment.add_argument("--use-center-point", action="store_true")
    segment.add_argument("--overwrite", action="store_true")

    validate_masks = subparsers.add_parser(
        "validate-masks", help="Validate Stage 2 cucumber mask outputs"
    )
    validate_masks.add_argument(
        "--dataset-root", type=Path, default=Path("sample_data/quickstart")
    )
    validate_masks.add_argument(
        "--output-root", type=Path, default=Path("outputs/quickstart/stage2")
    )
    validate_masks.add_argument("--score-threshold", type=float, default=0.7)

    synthesize = subparsers.add_parser(
        "synthesize-quickstart",
        help="Generate synthetic images and AISFormer-compatible annotations",
    )
    synthesize.add_argument(
        "--dataset-root", type=Path, default=Path("sample_data/quickstart")
    )
    synthesize.add_argument(
        "--mask-root", type=Path, default=Path("outputs/quickstart/stage2")
    )
    synthesize.add_argument(
        "--output-root", type=Path, default=Path("outputs/quickstart/synthesis")
    )
    synthesize.add_argument("--target-size", type=parse_size, default=(768, 1024))
    synthesize.add_argument("--tolerance", type=float, default=0.05)
    synthesize.add_argument("--seed", type=int, default=0)
    synthesize.add_argument(
        "--mode", choices=("representative", "full-grid"), default="representative"
    )
    synthesize.add_argument("--overwrite", action="store_true")

    synthesize_config = subparsers.add_parser(
        "synthesize",
        help="Generate a synthetic dataset from a validated JSON preset",
    )
    synthesize_config.add_argument(
        "--config", type=Path, default=Path("configs/quickstart.json")
    )
    synthesize_config.add_argument(
        "--output-root",
        type=Path,
        help="Override the output_root declared by the preset",
    )
    synthesize_config.add_argument("--overwrite", action="store_true")

    validate_synth = subparsers.add_parser(
        "validate-synthesis", help="Validate Stage 3 synthetic dataset outputs"
    )
    validate_synth.add_argument(
        "--output-root", type=Path, default=Path("outputs/quickstart/synthesis")
    )
    validate_synth.add_argument("--tolerance", type=float, default=0.05)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare-sample":
            report = prepare_sample(
                args.source_root,
                args.output_root,
                args.sample_ids or DEFAULT_SAMPLE_IDS,
                args.leaf_limit,
                args.overwrite,
            )
        elif args.command == "validate-inputs":
            report = validate_prepared_dataset(args.dataset_root)
        elif args.command == "check-stage2":
            report = check_stage2_environment()
        elif args.command == "segment-cucumbers":
            report = segment_cucumbers(
                args.dataset_root,
                args.output_root,
                args.checkpoint,
                args.model_config,
                args.device,
                args.score_threshold,
                args.use_center_point,
                args.overwrite,
            )
        elif args.command == "validate-masks":
            report = validate_cucumber_masks(
                args.dataset_root, args.output_root, args.score_threshold
            )
        elif args.command == "synthesize-quickstart":
            report = synthesize_quickstart(
                args.dataset_root,
                args.output_root,
                args.mask_root,
                args.target_size,
                args.tolerance,
                args.seed,
                args.mode,
                args.overwrite,
            )
        elif args.command == "synthesize":
            settings = load_synthesis_settings(args.config)
            report = synthesize_quickstart(
                dataset_root=settings.dataset_root,
                mask_root=settings.mask_root,
                output_root=args.output_root or settings.output_root,
                target_size=settings.target_size,
                tolerance=settings.tolerance,
                seed=settings.seed,
                mode=settings.mode,
                overwrite=args.overwrite,
                conditions=settings.conditions,
                ratios=settings.ratios,
                gammas=settings.gammas,
                baseline_ratio=settings.baseline_ratio,
                ratio_weights=settings.ratio_weights,
                sample_count=settings.sample_count,
                samples_per_condition=settings.samples_per_condition,
                condition_parameters=settings.condition_parameters,
            )
        else:
            report = validate_synthesis(args.output_root, args.tolerance)
    except (
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        ValueError,
        KeyError,
    ) as error:
        print(f"ERROR: {error}")
        return 2

    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 1
