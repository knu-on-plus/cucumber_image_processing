"""Prepare and validate normalized inputs for the public three-stage pipeline."""

from __future__ import annotations

import base64
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .image_metadata import read_jpeg_metadata, read_png_alpha_metadata


DEFAULT_SAMPLE_IDS = [
    "V003_3_3_1_2_4_2_2_1_0_0_20221019_5107_20240422195047",
    "V003_3_3_1_2_4_2_2_1_0_0_20221019_5143_20240422195049",
    "V003_3_3_1_2_4_2_2_1_0_0_20221019_5284_20240422195057",
]


def prepare_sample(
    source_root: Path,
    output_root: Path,
    sample_ids: Iterable[str] = DEFAULT_SAMPLE_IDS,
    leaf_limit: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output is not empty: {output_root}. Pass --overwrite to update it."
            )
        shutil.rmtree(output_root)

    image_source = source_root / "raw_cucumber_images"
    label_source = source_root / "raw_cucumber_labels"
    leaf_source = source_root / "cropped_leaves_samples"
    for required in (image_source, label_source, leaf_source):
        if not required.is_dir():
            raise FileNotFoundError(f"Required input directory is missing: {required}")

    images_dir = output_root / "images"
    annotations_dir = output_root / "annotations"
    leaves_dir = output_root / "leaf_cutouts"
    previews_dir = output_root / "previews"
    for directory in (images_dir, annotations_dir, leaves_dir, previews_dir):
        directory.mkdir(parents=True, exist_ok=True)

    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    image_manifest: list[dict[str, Any]] = []
    annotation_id = 1
    for image_id, source_id in enumerate(sample_ids, start=1):
        source_image = image_source / f"{source_id}.jpg"
        source_label = label_source / f"{source_id}.json"
        if not source_image.is_file() or not source_label.is_file():
            raise FileNotFoundError(f"Image/label pair is missing for sample: {source_id}")

        label = json.loads(source_label.read_text(encoding="utf-8"))
        description = label["description"]
        annotation_width = int(description["width"])
        annotation_height = int(description["height"])
        jpeg = read_jpeg_metadata(source_image)
        if jpeg.display_size != (annotation_width, annotation_height):
            raise ValueError(
                f"Coordinate mismatch for {source_id}: annotation "
                f"{annotation_width}x{annotation_height}, EXIF display {jpeg.display_size}"
            )

        output_name = f"scene_{image_id:03d}.jpg"
        shutil.copy2(source_image, images_dir / output_name)
        coco_images.append(
            {
                "id": image_id,
                "file_name": f"images/{output_name}",
                "width": annotation_width,
                "height": annotation_height,
                "source_id": source_id,
                "stored_width": jpeg.stored_width,
                "stored_height": jpeg.stored_height,
                "exif_orientation": jpeg.exif_orientation,
                "orientation_policy": "apply_exif_before_using_annotations",
            }
        )
        image_manifest.append(
            {
                "scene": f"scene_{image_id:03d}",
                "source_id": source_id,
                "annotation_source": str(source_label.relative_to(source_root)),
                "bbox_source": "provided_annotation",
            }
        )

        for result in label.get("result", []):
            if result.get("type") != "bbox":
                continue
            bbox = [int(result[key]) for key in ("x", "y", "w", "h")]
            _validate_bbox(bbox, annotation_width, annotation_height, source_id)
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "segmentation": [],
                    "iscrowd": 0,
                    "source_annotation_id": result.get("id"),
                    "bbox_source": "provided_annotation",
                }
            )
            annotation_id += 1

    if not coco_annotations:
        raise ValueError("No cucumber bbox annotations were found.")

    leaf_paths = sorted(leaf_source.glob("*.png"))
    if leaf_limit > 0:
        leaf_paths = leaf_paths[:leaf_limit]
    if len(leaf_paths) < 2:
        raise ValueError("At least two leaf cutouts are required for paper conditions 2 and 3.")

    leaf_assets: list[dict[str, Any]] = []
    for leaf_id, source_leaf in enumerate(leaf_paths, start=1):
        metadata = read_png_alpha_metadata(source_leaf)
        if not metadata.has_alpha or metadata.nonzero_fraction == 0:
            raise ValueError(f"Leaf cutout has no usable alpha mask: {source_leaf}")
        shutil.copy2(source_leaf, leaves_dir / source_leaf.name)
        leaf_assets.append(
            {
                "id": f"leaf_{leaf_id:03d}",
                "file_name": f"leaf_cutouts/{source_leaf.name}",
                "width": metadata.width,
                "height": metadata.height,
                "alpha_nonzero_fraction": round(metadata.nonzero_fraction, 6),
                "alpha_partial_pixels": metadata.partial_pixels,
                "mask_source": "provided_rgba_alpha",
            }
        )

    instances = {
        "info": {
            "description": "Quickstart cucumber detection input",
            "schema_version": "1.0",
        },
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [
            {"id": 1, "name": "cucumber"},
            {"id": 2, "name": "leaf"},
        ],
    }
    _write_json(annotations_dir / "instances.json", instances)
    _write_json(output_root / "leaf_assets.json", {"leaf_assets": leaf_assets})

    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Reproducible quickstart for the condition-based synthesis pipeline",
        "images": image_manifest,
        "inputs": {
            "cucumber_detection": "provided_annotation",
            "cucumber_mask": "pending_sam2",
            "leaf_detection": "not_required_for_provided_cutouts",
            "leaf_mask": "provided_rgba_alpha",
        },
        "next_step": {
            "stage": 2,
            "name": "SAM cucumber mask generation",
            "required_output": "One full cucumber mask per scene",
        },
        "paper_conditions": {
            "occlusion_ratios": [0.5, 0.75, 0.9],
            "ratio_sampling_weights": [5, 4, 1],
            "gamma_values": [0.3, 1.0, 3.0],
            "conditions": [
                "baseline_single_center",
                "condition_1_single_top_or_bottom",
                "condition_2_two_separate_top_and_bottom",
                "condition_3_two_attached_top_or_bottom",
            ],
        },
    }
    _write_json(output_root / "manifest.json", manifest)
    _write_bbox_previews(previews_dir, instances)
    _write_input_summary(previews_dir / "input_summary.md", instances, leaf_assets)
    return validate_prepared_dataset(output_root)


def validate_prepared_dataset(dataset_root: Path) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    instances = json.loads(
        (dataset_root / "annotations" / "instances.json").read_text(encoding="utf-8")
    )
    leaf_assets = json.loads(
        (dataset_root / "leaf_assets.json").read_text(encoding="utf-8")
    )["leaf_assets"]

    issues: list[str] = []
    for image in instances["images"]:
        image_path = dataset_root / image["file_name"]
        if not image_path.is_file():
            issues.append(f"Missing image: {image['file_name']}")
            continue
        jpeg = read_jpeg_metadata(image_path)
        if jpeg.display_size != (image["width"], image["height"]):
            issues.append(f"EXIF display size mismatch: {image['file_name']}")

    images_by_id = {image["id"]: image for image in instances["images"]}
    for annotation in instances["annotations"]:
        image = images_by_id.get(annotation["image_id"])
        if image is None:
            issues.append(f"Unknown image_id in annotation {annotation['id']}")
            continue
        try:
            _validate_bbox(
                annotation["bbox"], image["width"], image["height"], image["source_id"]
            )
        except ValueError as error:
            issues.append(str(error))

    for leaf in leaf_assets:
        leaf_path = dataset_root / leaf["file_name"]
        if not leaf_path.is_file():
            issues.append(f"Missing leaf cutout: {leaf['file_name']}")
            continue
        metadata = read_png_alpha_metadata(leaf_path)
        if not metadata.has_alpha or metadata.nonzero_fraction == 0:
            issues.append(f"Invalid alpha mask: {leaf['file_name']}")

    return {
        "valid": not issues,
        "image_count": len(instances["images"]),
        "cucumber_bbox_count": len(instances["annotations"]),
        "leaf_cutout_count": len(leaf_assets),
        "issues": issues,
    }


def _validate_bbox(bbox: list[int], width: int, height: int, source_id: str) -> None:
    if len(bbox) != 4:
        raise ValueError(f"Invalid bbox length for {source_id}: {bbox}")
    x, y, box_width, box_height = bbox
    if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
        raise ValueError(f"Invalid bbox values for {source_id}: {bbox}")
    if x + box_width > width or y + box_height > height:
        raise ValueError(f"Out-of-bounds bbox for {source_id}: {bbox}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_input_summary(
    path: Path, instances: dict[str, Any], leaf_assets: list[dict[str, Any]]
) -> None:
    rows = [
        "# Quickstart input summary",
        "",
        "The cucumber boxes are supplied annotations and replace YOLO output for this sample.",
        "Leaf cutouts already contain alpha masks and replace leaf YOLO + SAM output.",
        "",
        "| Scene | Source ID | Cucumber bbox | EXIF orientation |",
        "|---|---|---|---|",
    ]
    annotations_by_image = {
        annotation["image_id"]: annotation for annotation in instances["annotations"]
    }
    for image in instances["images"]:
        annotation = annotations_by_image[image["id"]]
        rows.append(
            f"| {Path(image['file_name']).stem} | `{image['source_id']}` | "
            f"`{annotation['bbox']}` | {image['exif_orientation']} |"
        )
    rows.extend(
        [
            "",
            "Visual bbox checks:",
            "",
            *[
                f"- [{Path(image['file_name']).stem}](bbox_{Path(image['file_name']).stem}.svg)"
                for image in instances["images"]
            ],
            "",
            f"Usable leaf cutouts: **{len(leaf_assets)}**",
            "",
            "Next step: generate one full cucumber mask per scene with SAM2.",
            "",
        ]
    )
    path.write_text("\n".join(rows), encoding="utf-8")


def _write_bbox_previews(previews_dir: Path, instances: dict[str, Any]) -> None:
    annotations_by_image = {
        annotation["image_id"]: annotation for annotation in instances["annotations"]
    }
    for image in instances["images"]:
        scene = Path(image["file_name"]).stem
        x, y, width, height = annotations_by_image[image["id"]]["bbox"]
        image_path = previews_dir.parent / image["file_name"]
        encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg"
  viewBox="0 0 {image['width']} {image['height']}">
  <image href="data:image/jpeg;base64,{encoded_image}"
    width="{image['width']}" height="{image['height']}"
    preserveAspectRatio="none" style="image-orientation:from-image"/>
  <rect x="{x}" y="{y}" width="{width}" height="{height}"
    fill="none" stroke="#ff2020" stroke-width="20"/>
  <text x="{x}" y="{max(40, y - 25)}" fill="#ff2020"
    font-family="sans-serif" font-size="60" font-weight="bold">cucumber</text>
</svg>
"""
        (previews_dir / f"bbox_{scene}.svg").write_text(svg, encoding="utf-8")
