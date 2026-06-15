"""Stage 3: condition-based synthesis and AISFormer-compatible export."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any


CONDITIONS = (
    "baseline_single_center",
    "condition_1_single_top_or_bottom",
    "condition_2_two_separate_top_and_bottom",
    "condition_3_two_attached_top_or_bottom",
)
RATIOS = (0.5, 0.75, 0.9)
GAMMAS = (0.3, 1.0, 3.0)
DEFAULT_CONDITION_PARAMETERS: dict[str, dict[str, Any]] = {
    CONDITIONS[0]: {
        "leaf_height_range": [0.12, 2.1],
        "search_steps": 120,
    },
    CONDITIONS[1]: {
        "positions": ["top", "bottom"],
        "position_fractions": [0.18, 0.82],
        "leaf_height_range": [0.12, 2.1],
        "search_steps": 120,
    },
    CONDITIONS[2]: {
        "leaf_height_ratio": 0.72,
        "outer_fraction_range": [-0.15, 0.42],
        "search_steps": 180,
        "max_leaf_overlap_ratio": 0.01,
    },
    CONDITIONS[3]: {
        "positions": ["top", "bottom"],
        "position_fractions": [0.18, 0.82],
        "leaf_height_range": [0.12, 4.0],
        "search_steps": 180,
        "attachment_overlap_ratio": 0.38,
    },
}


def synthesize_quickstart(
    dataset_root: Path,
    output_root: Path,
    mask_root: Path | None = None,
    target_size: tuple[int, int] = (768, 1024),
    tolerance: float = 0.05,
    seed: int = 0,
    mode: str = "representative",
    overwrite: bool = False,
    conditions: tuple[str, ...] = CONDITIONS,
    ratios: tuple[float, ...] = RATIOS,
    gammas: tuple[float, ...] = GAMMAS,
    baseline_ratio: float = 0.5,
    ratio_weights: tuple[int, ...] = (5, 4, 1),
    sample_count: int | None = None,
    samples_per_condition: int | None = None,
    condition_parameters: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    dataset_root = dataset_root.resolve()
    output_root = output_root.resolve()
    mask_root = (mask_root or dataset_root).resolve()
    condition_parameters = condition_parameters or DEFAULT_CONDITION_PARAMETERS
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output is not empty: {output_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    instances = _read_json(dataset_root / "annotations" / "instances.json")
    cucumber_masks = _read_json(mask_root / "annotations" / "cucumber_masks.json")[
        "masks"
    ]
    leaf_assets = _read_json(dataset_root / "leaf_assets.json")["leaf_assets"]
    if len(leaf_assets) < 2:
        raise ValueError("Stage 3 requires at least two leaf cutouts.")

    images_dir = output_root / "images"
    masks_root = output_root / "masks"
    previews_dir = output_root / "previews" / "layouts"
    annotations_dir = output_root / "annotations"
    for directory in (
        images_dir,
        annotations_dir,
        previews_dir,
        masks_root / "amodal",
        masks_root / "inmodal",
        masks_root / "occlusion",
        masks_root / "occluder",
        masks_root / "leaves",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    mask_by_image = {entry["image_id"]: entry for entry in cucumber_masks}
    leaves = [
        np.asarray(Image.open(dataset_root / asset["file_name"]).convert("RGBA"))
        for asset in leaf_assets
    ]
    leaf_offset = seed % len(leaves)
    layouts: list[dict[str, Any]] = []
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    image_id = annotation_id = 1
    target_width, target_height = target_size
    scene_cache: dict[int, tuple[Any, Any]] = {}
    layout_specs = _layout_specs_for_mode(
        instances["images"],
        mode,
        conditions,
        ratios,
        baseline_ratio,
        ratio_weights,
        sample_count,
        samples_per_condition,
    )

    for spec in layout_specs:
        scene_index = spec["scene_index"]
        source_image = spec["source_image"]
        condition_index = spec["condition_index"]
        condition = spec["condition"]
        requested_ratio = spec["requested_ratio"]
        if scene_index in scene_cache:
            image_array, amodal_mask = scene_cache[scene_index]
        else:
            image_array, amodal_mask = _load_scene_arrays(
                dataset_root,
                source_image,
                mask_by_image,
                mask_root,
                target_size,
                cv2,
                np,
                Image,
                ImageOps,
            )
            scene_cache[scene_index] = (image_array, amodal_mask)

        layout_id = (
            f"{spec['sample_prefix']}"
            f"scene_{source_image['id']:03d}"
            f"__{_short_condition(condition)}"
            f"__r{int(requested_ratio * 100):02d}"
        )
        leaf_index = (
            spec["leaf_seed"] * len(conditions) + condition_index + leaf_offset
        ) % len(leaves)
        selected = [
            leaves[leaf_index],
            leaves[(leaf_index + 1) % len(leaves)],
        ]
        placement = _optimize_condition(
            condition,
            amodal_mask,
            selected,
            requested_ratio,
            spec["leaf_seed"],
            cv2,
            np,
            baseline_ratio,
            condition_parameters,
        )
        achieved_ratio = placement["achieved_ratio"]
        ratio_error = abs(achieved_ratio - placement["target_ratio"])
        if ratio_error > tolerance:
            raise ValueError(
                f"{layout_id} missed target ratio: target="
                f"{placement['target_ratio']:.3f}, achieved={achieved_ratio:.3f}, "
                f"error={ratio_error:.3f}, tolerance={tolerance:.3f}"
            )

        synthetic, leaf_masks = _composite_leaves(
            image_array,
            placement["leaves"],
            placement["centers"],
            np,
            placement.get("component_masks"),
        )
        occluder_mask = np.logical_or.reduce(leaf_masks)
        occlusion_mask = amodal_mask & occluder_mask
        inmodal_mask = amodal_mask & ~occluder_mask
        mask_paths = _save_layout_masks(
            output_root,
            layout_id,
            amodal_mask,
            inmodal_mask,
            occlusion_mask,
            occluder_mask,
            leaf_masks,
            Image,
            np,
        )
        _write_layout_preview(
            previews_dir / f"{layout_id}.jpg",
            synthetic,
            amodal_mask,
            inmodal_mask,
            occluder_mask,
            occlusion_mask,
            Image,
            np,
        )

        layout = {
            "id": layout_id,
            "sample_number": spec["sample_number"],
            "source_image_id": source_image["id"],
            "source_file_name": source_image["file_name"],
            "condition": condition,
            "target_occlusion_ratio": round(placement["target_ratio"], 6),
            "achieved_occlusion_ratio": round(achieved_ratio, 6),
            "ratio_error": round(ratio_error, 6),
            "leaf_count": len(leaf_masks),
            "leaf_asset_ids": [
                leaf_assets[leaf_index]["id"],
                leaf_assets[(leaf_index + 1) % len(leaves)]["id"],
            ][: len(leaf_masks)],
            "mask_files": mask_paths,
            "placement": placement["metadata"],
        }
        layouts.append(layout)

        for gamma in gammas:
            output_name = f"{layout_id}__g{_gamma_name(gamma)}.jpg"
            gamma_image = _apply_gamma(synthetic, gamma, np)
            Image.fromarray(gamma_image).save(
                images_dir / output_name, quality=92, subsampling=0
            )
            coco_images.append(
                {
                    "id": image_id,
                    "file_name": f"images/{output_name}",
                    "width": target_width,
                    "height": target_height,
                    "source_image_id": source_image["id"],
                    "layout_id": layout_id,
                    "condition": condition,
                    "gamma": gamma,
                    "gamma_label": _gamma_label(gamma),
                    "target_occlusion_ratio": round(placement["target_ratio"], 6),
                    "achieved_occlusion_ratio": round(achieved_ratio, 6),
                    "mask_files": mask_paths,
                }
            )
            cucumber_annotation = _cucumber_annotation(
                annotation_id,
                image_id,
                amodal_mask,
                inmodal_mask,
                occlusion_mask,
                occluder_mask,
                cv2,
                np,
            )
            coco_annotations.append(cucumber_annotation)
            annotation_id += 1
            for leaf_mask in leaf_masks:
                coco_annotations.append(
                    _leaf_annotation(annotation_id, image_id, leaf_mask, cv2, np)
                )
                annotation_id += 1
            image_id += 1

    if not layouts:
        raise ValueError("No synthesis layouts were generated.")

    dataset = {
        "info": {
            "description": "Condition-based synthetic amodal cucumber dataset",
            "schema_version": "1.0",
            "seed": seed,
            "mode": mode,
            "target_size": [target_width, target_height],
            "occlusion_tolerance": tolerance,
            "conditions": list(conditions),
            "occlusion_ratios": list(ratios),
            "baseline_occlusion_ratio": baseline_ratio,
            "gamma_values": list(gammas),
            "ratio_sampling_weights": list(ratio_weights),
            "sample_count": sample_count,
            "samples_per_condition": samples_per_condition,
            "condition_parameters": condition_parameters,
        },
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [
            {"id": 1, "name": "cucumber", "supercategory": "vegetable"},
            {"id": 2, "name": "leaf", "supercategory": "vegetable"},
        ],
        "aisformer_contract": {
            "amodal_segmentation_field": "segmentation",
            "amodal_bbox_field": "bbox",
            "inmodal_segmentation_field": "inmodal_seg",
            "inmodal_bbox_field": "inmodal_bbox",
        },
    }
    _write_json(annotations_dir / "instances.json", dataset)
    _write_json(annotations_dir / "layouts.json", {"layouts": layouts})
    _write_synthesis_summary(output_root / "previews" / "summary.md", layouts)
    _mark_stage3_complete(dataset_root, output_root, len(layouts), len(coco_images))
    return validate_synthesis(output_root, tolerance)


def validate_synthesis(output_root: Path, tolerance: float = 0.05) -> dict[str, Any]:
    output_root = output_root.resolve()
    instances_path = output_root / "annotations" / "instances.json"
    layouts_path = output_root / "annotations" / "layouts.json"
    if not instances_path.is_file() or not layouts_path.is_file():
        return {
            "valid": False,
            "layout_count": 0,
            "image_count": 0,
            "issues": ["Missing Stage 3 annotation outputs."],
            "warnings": [],
        }
    instances = _read_json(instances_path)
    layouts = _read_json(layouts_path)["layouts"]
    issues: list[str] = []
    warnings: list[str] = []
    image_ids = {image["id"] for image in instances["images"]}
    for image in instances["images"]:
        if not (output_root / image["file_name"]).is_file():
            issues.append(f"Missing synthetic image: {image['file_name']}")
        for mask_file in image["mask_files"].values():
            paths = mask_file if isinstance(mask_file, list) else [mask_file]
            for path in paths:
                if not (output_root / path).is_file():
                    issues.append(f"Missing mask: {path}")
    for annotation in instances["annotations"]:
        if annotation["image_id"] not in image_ids:
            issues.append(f"Unknown image_id in annotation {annotation['id']}")
        if annotation["category_id"] == 1:
            required = {"segmentation", "bbox", "inmodal_seg", "inmodal_bbox"}
            missing = required - annotation.keys()
            if missing:
                issues.append(
                    f"Missing AISFormer fields in annotation {annotation['id']}: "
                    f"{sorted(missing)}"
                )
            if (
                annotation["inmodal_area"] + annotation["occluded_area"]
                != annotation["amodal_area"]
            ):
                issues.append(f"Mask area identity failed: annotation {annotation['id']}")
    for layout in layouts:
        if layout["ratio_error"] > tolerance:
            issues.append(f"Occlusion ratio outside tolerance: {layout['id']}")
        if layout["achieved_occlusion_ratio"] >= 0.98:
            warnings.append(f"Almost fully occluded cucumber: {layout['id']}")
        if layout["condition"] in {CONDITIONS[2], CONDITIONS[3]} and layout[
            "leaf_count"
        ] != 2:
            issues.append(f"Expected two leaf instances: {layout['id']}")
    return {
        "valid": not issues,
        "layout_count": len(layouts),
        "image_count": len(instances["images"]),
        "annotation_count": len(instances["annotations"]),
        "issues": issues,
        "warnings": warnings,
    }


def _ratios_for_mode(
    mode: str,
    scene_index: int,
    conditions: tuple[str, ...],
    ratios: tuple[float, ...],
    baseline_ratio: float,
) -> dict[str, tuple[float, ...]]:
    if mode == "full-grid":
        return {
            condition: (baseline_ratio,) if condition == CONDITIONS[0] else ratios
            for condition in conditions
        }
    if mode != "representative":
        raise ValueError(f"Unknown synthesis mode: {mode}")
    representative = ratios[scene_index % len(ratios)]
    return {
        condition: (baseline_ratio,)
        if condition == CONDITIONS[0]
        else (representative,)
        for condition in conditions
    }


def _layout_specs_for_mode(
    source_images: list[dict[str, Any]],
    mode: str,
    conditions: tuple[str, ...],
    ratios: tuple[float, ...],
    baseline_ratio: float,
    ratio_weights: tuple[int, ...],
    sample_count: int | None,
    samples_per_condition: int | None,
) -> list[dict[str, Any]]:
    if mode in {"representative", "full-grid"}:
        specs: list[dict[str, Any]] = []
        for scene_index, source_image in enumerate(source_images):
            condition_ratios = _ratios_for_mode(
                mode, scene_index, conditions, ratios, baseline_ratio
            )
            for condition_index, condition in enumerate(conditions):
                for requested_ratio in condition_ratios[condition]:
                    specs.append(
                        {
                            "sample_number": None,
                            "sample_prefix": "",
                            "scene_index": scene_index,
                            "source_image": source_image,
                            "condition_index": condition_index,
                            "condition": condition,
                            "requested_ratio": requested_ratio,
                            "leaf_seed": scene_index,
                        }
                    )
        return specs
    if mode != "weighted-sample":
        raise ValueError(f"Unknown synthesis mode: {mode}")
    if CONDITIONS[0] in conditions:
        raise ValueError(
            "weighted-sample supports Conditions 1-3 only; "
            "baseline_single_center is fixed-ratio and should be generated "
            "with representative or full-grid mode."
        )
    per_condition = _resolve_samples_per_condition(
        len(conditions), ratio_weights, sample_count, samples_per_condition
    )
    ratio_counts = _weighted_ratio_counts(per_condition, ratio_weights)
    ratio_plan = _interleaved_ratio_plan(ratios, ratio_counts)
    specs = []
    sample_number = 1
    for condition_index, condition in enumerate(conditions):
        for condition_sample_index, requested_ratio in enumerate(ratio_plan):
            scene_index = (condition_index + condition_sample_index) % len(source_images)
            specs.append(
                {
                    "sample_number": sample_number,
                    "sample_prefix": f"sample_{sample_number:05d}__",
                    "scene_index": scene_index,
                    "source_image": source_images[scene_index],
                    "condition_index": condition_index,
                    "condition": condition,
                    "requested_ratio": requested_ratio,
                    "leaf_seed": condition_sample_index,
                }
            )
            sample_number += 1
    return specs


def _resolve_samples_per_condition(
    condition_count: int,
    ratio_weights: tuple[int, ...],
    sample_count: int | None,
    samples_per_condition: int | None,
) -> int:
    if sample_count is not None and samples_per_condition is not None:
        raise ValueError("Use either sample_count or samples_per_condition, not both.")
    weight_total = sum(ratio_weights)
    if samples_per_condition is not None:
        per_condition = samples_per_condition
    elif sample_count is not None:
        if sample_count % condition_count != 0:
            raise ValueError("sample_count must divide evenly across conditions.")
        per_condition = sample_count // condition_count
    else:
        raise ValueError("weighted-sample requires sample_count or samples_per_condition.")
    if per_condition <= 0:
        raise ValueError("samples_per_condition must be positive.")
    if per_condition % weight_total != 0:
        raise ValueError(
            "samples_per_condition must be a multiple of "
            f"sum(ratio_sampling_weights)={weight_total} for an exact ratio."
        )
    return per_condition


def _weighted_ratio_counts(
    samples_per_condition: int, ratio_weights: tuple[int, ...]
) -> tuple[int, ...]:
    weight_total = sum(ratio_weights)
    multiplier = samples_per_condition // weight_total
    return tuple(weight * multiplier for weight in ratio_weights)


def _interleaved_ratio_plan(
    ratios: tuple[float, ...], ratio_counts: tuple[int, ...]
) -> tuple[float, ...]:
    plan: list[float] = []
    remaining = list(ratio_counts)
    while any(count > 0 for count in remaining):
        for index, ratio in enumerate(ratios):
            if remaining[index] > 0:
                plan.append(ratio)
                remaining[index] -= 1
    return tuple(plan)


def _load_scene_arrays(
    dataset_root: Path,
    source_image: dict[str, Any],
    mask_by_image: dict[int, dict[str, Any]],
    mask_root: Path,
    target_size: tuple[int, int],
    cv2: Any,
    np: Any,
    image_module: Any,
    image_ops_module: Any,
) -> tuple[Any, Any]:
    image = image_ops_module.exif_transpose(
        image_module.open(dataset_root / source_image["file_name"])
    ).convert("RGB")
    image_array = cv2.resize(
        np.asarray(image), target_size, interpolation=cv2.INTER_AREA
    )
    mask_entry = mask_by_image.get(source_image["id"])
    if mask_entry is None:
        raise ValueError(f"Missing cucumber mask for image_id={source_image['id']}")
    amodal_mask = (
        cv2.resize(
            np.asarray(
                image_module.open(mask_root / mask_entry["file_name"]).convert("L")
            ),
            target_size,
            interpolation=cv2.INTER_NEAREST,
        )
        > 0
    )
    if not amodal_mask.any():
        raise ValueError(f"Empty cucumber mask for image_id={source_image['id']}")
    return image_array, amodal_mask


def _optimize_condition(
    condition: str,
    cucumber_mask: Any,
    leaves: list[Any],
    requested_ratio: float,
    scene_index: int,
    cv2: Any,
    np: Any,
    baseline_ratio: float,
    condition_parameters: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    target_ratio = baseline_ratio if condition == CONDITIONS[0] else requested_ratio
    parameters = condition_parameters[condition]
    x, y, width, height = _bbox(cucumber_mask, np)
    center_x = x + width / 2
    center_y = y + height / 2
    best: dict[str, Any] | None = None

    def consider(
        candidate_leaves: list[Any],
        centers: list[tuple[float, float]],
        metadata: dict[str, Any],
        component_masks: list[Any] | None = None,
    ) -> None:
        nonlocal best
        masks = [
            _place_mask(leaf[:, :, 3] > 0, center, cucumber_mask.shape, np)
            for leaf, center in zip(candidate_leaves, centers)
        ]
        combined = np.logical_or.reduce(masks)
        achieved = float((combined & cucumber_mask).sum() / cucumber_mask.sum())
        error = abs(achieved - target_ratio)
        candidate = {
            "leaves": candidate_leaves,
            "centers": centers,
            "target_ratio": target_ratio,
            "achieved_ratio": achieved,
            "metadata": metadata,
            "component_masks": component_masks,
            "error": error,
        }
        if best is None or error < best["error"]:
            best = candidate

    if condition in {CONDITIONS[0], CONDITIONS[1]}:
        position = "center"
        center = (center_x, center_y)
        if condition == CONDITIONS[1]:
            position_index = scene_index % len(parameters["positions"])
            position = parameters["positions"][position_index]
            center = (
                center_x,
                y + height * float(parameters["position_fractions"][position_index]),
            )
        height_range = parameters["leaf_height_range"]
        for leaf_height in np.linspace(
            height * float(height_range[0]),
            height * float(height_range[1]),
            int(parameters["search_steps"]),
        ):
            resized = _resize_leaf(leaves[0], leaf_height, cv2)
            consider(
                [resized],
                [center],
                {
                    "strategy": "increased_leaf_size"
                    if condition == CONDITIONS[1]
                    else "fixed_center",
                    "position": position,
                    "leaf_height": int(resized.shape[0]),
                },
            )
    elif condition == CONDITIONS[2]:
        resized_leaves = [
            _resize_leaf(leaves[0], height * float(parameters["leaf_height_ratio"]), cv2),
            _resize_leaf(leaves[1], height * float(parameters["leaf_height_ratio"]), cv2),
        ]
        outer_range = parameters["outer_fraction_range"]
        for outer_fraction in np.linspace(
            float(outer_range[0]),
            float(outer_range[1]),
            int(parameters["search_steps"]),
        ):
            centers = [
                (center_x, y + height * outer_fraction),
                (center_x, y + height * (1 - outer_fraction)),
            ]
            masks = [
                _place_mask(leaf[:, :, 3] > 0, center, cucumber_mask.shape, np)
                for leaf, center in zip(resized_leaves, centers)
            ]
            if (masks[0] & masks[1]).sum() > min(masks[0].sum(), masks[1].sum()) * float(
                parameters["max_leaf_overlap_ratio"]
            ):
                continue
            consider(
                resized_leaves,
                centers,
                {
                    "strategy": "position_based_adjustment",
                    "positions": ["top", "bottom"],
                    "outer_fraction": round(float(outer_fraction), 6),
                },
            )
    elif condition == CONDITIONS[3]:
        position_index = scene_index % len(parameters["positions"])
        position = parameters["positions"][position_index]
        center = (
            center_x,
            y + height * float(parameters["position_fractions"][position_index]),
        )
        height_range = parameters["leaf_height_range"]
        for leaf_height in np.linspace(
            height * float(height_range[0]),
            height * float(height_range[1]),
            int(parameters["search_steps"]),
        ):
            attached, component_masks = _attach_leaves(
                leaves,
                leaf_height,
                cv2,
                np,
                float(parameters["attachment_overlap_ratio"]),
            )
            consider(
                [attached],
                [center],
                {
                    "strategy": "increased_combined_leaf_size",
                    "position": position,
                    "combined_leaf_height": int(attached.shape[0]),
                    "source_leaf_count": 2,
                },
                component_masks,
            )
    else:
        raise ValueError(f"Unknown condition: {condition}")

    if best is None:
        raise ValueError(f"No valid placement found for condition: {condition}")
    best.pop("error")
    return best


def _resize_leaf(leaf: Any, target_height: float, cv2: Any) -> Any:
    height, width = leaf.shape[:2]
    scale = max(1.0 / height, target_height / height)
    target_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(leaf, (target_width, resized_height), interpolation=cv2.INTER_AREA)


def _attach_leaves(
    leaves: list[Any],
    target_height: float,
    cv2: Any,
    np: Any,
    overlap_ratio: float = 0.38,
) -> tuple[Any, list[Any]]:
    first = _resize_leaf(leaves[0], target_height, cv2)
    second = _resize_leaf(leaves[1], target_height, cv2)
    overlap = int(min(first.shape[1], second.shape[1]) * overlap_ratio)
    canvas_height = max(first.shape[0], second.shape[0])
    canvas_width = first.shape[1] + second.shape[1] - overlap
    canvas = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
    first_x, first_y = 0, (canvas_height - first.shape[0]) // 2
    second_x = first.shape[1] - overlap
    second_y = (canvas_height - second.shape[0]) // 2
    _alpha_composite_region(canvas, first, first_x, first_y, np)
    _alpha_composite_region(
        canvas,
        second,
        second_x,
        second_y,
        np,
    )
    first_mask = np.zeros((canvas_height, canvas_width), dtype=bool)
    second_mask = np.zeros((canvas_height, canvas_width), dtype=bool)
    first_mask[
        first_y : first_y + first.shape[0], first_x : first_x + first.shape[1]
    ] = first[:, :, 3] > 0
    second_mask[
        second_y : second_y + second.shape[0], second_x : second_x + second.shape[1]
    ] = second[:, :, 3] > 0
    first_mask &= ~second_mask
    return canvas, [first_mask, second_mask]


def _composite_leaves(
    image: Any,
    leaves: list[Any],
    centers: list[tuple[float, float]],
    np: Any,
    component_masks: list[Any] | None = None,
) -> tuple[Any, list[Any]]:
    result = image.copy()
    masks: list[Any] = []
    for leaf, center in zip(leaves, centers):
        x = int(round(center[0] - leaf.shape[1] / 2))
        y = int(round(center[1] - leaf.shape[0] / 2))
        _alpha_composite_region(result, leaf, x, y, np)
        masks.append(_place_mask(leaf[:, :, 3] > 0, center, image.shape[:2], np))
    if component_masks is not None:
        masks = [
            _place_mask(mask, centers[0], image.shape[:2], np)
            for mask in component_masks
        ]
    return result, masks


def _alpha_composite_region(canvas: Any, overlay: Any, x: int, y: int, np: Any) -> None:
    height, width = canvas.shape[:2]
    overlay_height, overlay_width = overlay.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + overlay_width), min(height, y + overlay_height)
    if x1 >= x2 or y1 >= y2:
        return
    source = overlay[y1 - y : y2 - y, x1 - x : x2 - x]
    alpha = source[:, :, 3:4].astype(np.float32) / 255.0
    destination = canvas[y1:y2, x1:x2]
    if destination.shape[2] == 4:
        source_alpha = source[:, :, 3:4]
        canvas[y1:y2, x1:x2, :3] = (
            source[:, :, :3] * alpha + destination[:, :, :3] * (1 - alpha)
        ).astype(np.uint8)
        canvas[y1:y2, x1:x2, 3:4] = np.maximum(
            destination[:, :, 3:4], source_alpha
        )
    else:
        canvas[y1:y2, x1:x2] = (
            source[:, :, :3] * alpha + destination * (1 - alpha)
        ).astype(np.uint8)


def _place_mask(mask: Any, center: tuple[float, float], shape: tuple[int, int], np: Any) -> Any:
    canvas = np.zeros(shape, dtype=bool)
    height, width = shape
    mask_height, mask_width = mask.shape
    x = int(round(center[0] - mask_width / 2))
    y = int(round(center[1] - mask_height / 2))
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + mask_width), min(height, y + mask_height)
    if x1 < x2 and y1 < y2:
        canvas[y1:y2, x1:x2] = mask[y1 - y : y2 - y, x1 - x : x2 - x]
    return canvas


def _save_layout_masks(
    output_root: Path,
    layout_id: str,
    amodal: Any,
    inmodal: Any,
    occlusion: Any,
    occluder: Any,
    leaves: list[Any],
    image_module: Any,
    np: Any,
) -> dict[str, Any]:
    paths: dict[str, Any] = {}
    for name, mask in (
        ("amodal", amodal),
        ("inmodal", inmodal),
        ("occlusion", occlusion),
        ("occluder", occluder),
    ):
        relative = f"masks/{name}/{layout_id}.png"
        image_module.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
            output_root / relative
        )
        paths[name] = relative
    leaf_paths: list[str] = []
    for index, mask in enumerate(leaves, start=1):
        relative = f"masks/leaves/{layout_id}__leaf_{index:02d}.png"
        image_module.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
            output_root / relative
        )
        leaf_paths.append(relative)
    paths["leaves"] = leaf_paths
    return paths


def _write_layout_preview(
    path: Path,
    synthetic: Any,
    amodal: Any,
    inmodal: Any,
    occluder: Any,
    occlusion: Any,
    image_module: Any,
    np: Any,
) -> None:
    height, width = synthetic.shape[:2]
    panels = [synthetic]
    for mask, color in (
        (amodal, (0, 255, 0)),
        (inmodal, (0, 190, 255)),
        (occluder, (255, 0, 0)),
    ):
        panel = synthetic.copy()
        tint = np.zeros_like(panel)
        tint[:, :] = color
        panel[mask] = (panel[mask] * 0.35 + tint[mask] * 0.65).astype(np.uint8)
        panel[occlusion] = np.array([255, 0, 255], dtype=np.uint8)
        panels.append(panel)
    grid = np.concatenate(
        [np.concatenate(panels[:2], axis=1), np.concatenate(panels[2:], axis=1)],
        axis=0,
    )
    image_module.fromarray(grid).resize((width, height)).save(path, quality=88)


def _cucumber_annotation(
    annotation_id: int,
    image_id: int,
    amodal: Any,
    inmodal: Any,
    occlusion: Any,
    occluder: Any,
    cv2: Any,
    np: Any,
) -> dict[str, Any]:
    amodal_seg = _mask_to_polygons(amodal, cv2, np)
    inmodal_seg = _mask_to_polygons(inmodal, cv2, np)
    occlusion_seg = _mask_to_polygons(occlusion, cv2, np)
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "iscrowd": 0,
        "bbox": _bbox(amodal, np),
        "area": int(amodal.sum()),
        "segmentation": amodal_seg,
        "amodal_bbox": _bbox(amodal, np),
        "amodal_area": int(amodal.sum()),
        "amodal_segm": amodal_seg,
        "amodal_segmentation": amodal_seg,
        "inmodal_bbox": _bbox(inmodal, np),
        "inmodal_area": int(inmodal.sum()),
        "inmodal_seg": inmodal_seg,
        "inmodal_segm": inmodal_seg,
        "visible_bbox": _bbox(inmodal, np),
        "visible_segm": inmodal_seg,
        "occluded_area": int(occlusion.sum()),
        "occluded_segm": occlusion_seg,
        "occluder_segm": _mask_to_polygons(occluder, cv2, np),
    }


def _leaf_annotation(
    annotation_id: int, image_id: int, mask: Any, cv2: Any, np: Any
) -> dict[str, Any]:
    segmentation = _mask_to_polygons(mask, cv2, np)
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 2,
        "iscrowd": 0,
        "bbox": _bbox(mask, np),
        "area": int(mask.sum()),
        "segmentation": segmentation,
    }


def _bbox(mask: Any, np: Any) -> list[int]:
    y, x = np.nonzero(mask)
    if len(x) == 0:
        return [0, 0, 0, 0]
    return [
        int(x.min()),
        int(y.min()),
        int(x.max() - x.min() + 1),
        int(y.max() - y.min() + 1),
    ]


def _mask_to_polygons(mask: Any, cv2: Any, np: Any) -> list[list[float]]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons: list[list[float]] = []
    for contour in contours:
        if cv2.contourArea(contour) < 4:
            continue
        polygon = contour.reshape(-1, 2).astype(float).flatten().tolist()
        if len(polygon) >= 6:
            polygons.append(polygon)
    return polygons


def _apply_gamma(image: Any, gamma: float, np: Any) -> Any:
    if math.isclose(gamma, 1.0):
        return image.copy()
    normalized = image.astype(np.float32) / 255.0
    corrected = np.power(normalized, 1.0 / gamma)
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)


def _short_condition(condition: str) -> str:
    return {
        CONDITIONS[0]: "baseline",
        CONDITIONS[1]: "condition1",
        CONDITIONS[2]: "condition2",
        CONDITIONS[3]: "condition3",
    }[condition]


def _gamma_name(gamma: float) -> str:
    return str(gamma).replace(".", "p")


def _gamma_label(gamma: float) -> str:
    if math.isclose(gamma, 0.3):
        return "dark"
    if math.isclose(gamma, 1.0):
        return "original"
    if math.isclose(gamma, 3.0):
        return "bright"
    return f"gamma_{_gamma_name(gamma)}"


def _write_synthesis_summary(path: Path, layouts: list[dict[str, Any]]) -> None:
    rows = [
        "# Stage 3 synthesis preview summary",
        "",
        "Each preview is a 2x2 grid: synthetic image, amodal mask overlay, "
        "inmodal mask overlay, and occluder mask overlay. Magenta marks the "
        "cucumber/leaf overlap.",
        "",
        "| Layout | Condition | Target | Achieved | Error | Leaves |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for layout in layouts:
        rows.append(
            f"| [{layout['id']}](layouts/{layout['id']}.jpg) | "
            f"{layout['condition']} | {layout['target_occlusion_ratio']:.3f} | "
            f"{layout['achieved_occlusion_ratio']:.3f} | "
            f"{layout['ratio_error']:.3f} | {layout['leaf_count']} |"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _mark_stage3_complete(
    dataset_root: Path, output_root: Path, layout_count: int, image_count: int
) -> None:
    try:
        output_reference = str(output_root.relative_to(dataset_root))
    except ValueError:
        _write_json(
            output_root / "manifest.json",
            {
                "schema_version": "1.0",
                "input_dataset": str(dataset_root),
                "stage3": {
                    "status": "complete",
                    "output_root": str(output_root),
                    "layout_count": layout_count,
                    "image_count": image_count,
                },
            },
        )
        return
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = _read_json(manifest_path)
    manifest["stage3"] = {
        "status": "complete",
        "output_root": output_reference,
        "layout_count": layout_count,
        "image_count": image_count,
    }
    manifest["next_step"] = {
        "stage": "validation",
        "name": "Review synthesis layouts and annotations",
        "required_output": "Approved representative synthesis dataset",
    }
    _write_json(manifest_path, manifest)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
