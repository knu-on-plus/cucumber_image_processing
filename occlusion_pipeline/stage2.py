"""Stage 2: generate and validate cucumber masks with SAM2 bbox prompts."""

from __future__ import annotations

import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any


def check_stage2_environment() -> dict[str, Any]:
    required = ("numpy", "PIL", "torch", "sam2")
    modules = {name: importlib.util.find_spec(name) is not None for name in required}
    report: dict[str, Any] = {
        "valid": all(modules.values()),
        "modules": modules,
        "device": None,
        "issues": [],
    }
    for name, available in modules.items():
        if not available:
            report["issues"].append(f"Missing Python module: {name}")
    if modules["torch"]:
        import torch

        report["torch_version"] = torch.__version__
        report["device"] = _resolve_device(torch, "auto")
        if report["device"] == "cpu":
            report["issues"].append("No CUDA/MPS device detected; SAM2 will be slow on CPU.")
    return report


def segment_cucumbers(
    dataset_root: Path,
    output_root: Path,
    checkpoint: Path,
    model_config: str,
    device: str = "auto",
    score_threshold: float = 0.7,
    use_center_point: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    environment = check_stage2_environment()
    if not environment["valid"]:
        raise RuntimeError("; ".join(environment["issues"]))
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint is missing: {checkpoint}")

    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageOps
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    dataset_root = dataset_root.resolve()
    output_root = output_root.resolve()
    instances = _read_json(dataset_root / "annotations" / "instances.json")
    masks_dir = output_root / "masks" / "cucumber"
    annotations_dir = output_root / "annotations"
    previews_dir = output_root / "previews" / "masks"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = _resolve_device(torch, device)
    sam2 = build_sam2(model_config, str(checkpoint), device=resolved_device)
    predictor = SAM2ImagePredictor(sam2)
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in instances["annotations"]:
        if annotation["category_id"] == 1:
            annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    mask_entries: list[dict[str, Any]] = []
    for image in instances["images"]:
        image_path = dataset_root / image["file_name"]
        pil_image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        if pil_image.size != (image["width"], image["height"]):
            raise ValueError(
                f"Image/annotation size mismatch after EXIF transpose: {image_path}"
            )
        image_array = np.array(pil_image)
        predictor.set_image(image_array)

        for object_index, annotation in enumerate(
            annotations_by_image.get(image["id"], []), start=1
        ):
            x, y, width, height = annotation["bbox"]
            box = np.array([x, y, x + width, y + height], dtype=np.float32)
            prompts: dict[str, Any] = {"box": box, "multimask_output": True}
            if use_center_point:
                prompts["point_coords"] = np.array(
                    [[x + width / 2, y + height / 2]], dtype=np.float32
                )
                prompts["point_labels"] = np.array([1], dtype=np.int32)

            inference_context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if resolved_device == "cuda"
                else nullcontext()
            )
            with torch.inference_mode(), inference_context:
                masks, scores, _ = predictor.predict(**prompts)
            best_index = int(np.argmax(scores))
            score = float(scores[best_index])
            if score < score_threshold:
                raise ValueError(
                    f"SAM2 score below threshold for {image['file_name']}: "
                    f"{score:.4f} < {score_threshold:.4f}"
                )
            mask = np.asarray(masks[best_index], dtype=bool)
            if mask.shape != (image["height"], image["width"]) or not mask.any():
                raise ValueError(f"Invalid SAM2 mask for {image['file_name']}")

            mask_name = f"{Path(image['file_name']).stem}_{object_index:03d}.png"
            mask_path = masks_dir / mask_name
            if mask_path.exists() and not overwrite:
                raise FileExistsError(
                    f"Mask already exists: {mask_path}. Pass --overwrite to replace it."
                )
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)

            active_y, active_x = np.nonzero(mask)
            mask_bbox = [
                int(active_x.min()),
                int(active_y.min()),
                int(active_x.max() - active_x.min() + 1),
                int(active_y.max() - active_y.min() + 1),
            ]
            mask_area = int(mask.sum())
            mask_entries.append(
                {
                    "image_id": image["id"],
                    "annotation_id": annotation["id"],
                    "file_name": str(mask_path.relative_to(output_root)),
                    "sam_score": round(score, 6),
                    "prompt_bbox": annotation["bbox"],
                    "mask_bbox": mask_bbox,
                    "mask_area": mask_area,
                    "image_area_fraction": round(
                        mask_area / (image["width"] * image["height"]), 6
                    ),
                    "prompt_mode": "box_and_center_point"
                    if use_center_point
                    else "box",
                }
            )
            _write_mask_preview(
                pil_image,
                mask,
                annotation["bbox"],
                previews_dir / f"{Path(mask_name).stem}.jpg",
                Image,
                ImageDraw,
            )

    mask_manifest = {
        "schema_version": "1.0",
        "generator": "SAM2ImagePredictor",
        "model_config": model_config,
        "checkpoint_name": checkpoint.name,
        "device": resolved_device,
        "score_threshold": score_threshold,
        "masks": mask_entries,
    }
    _write_json(annotations_dir / "cucumber_masks.json", mask_manifest)
    _mark_stage2_complete(dataset_root, output_root, mask_manifest)
    return validate_cucumber_masks(dataset_root, output_root, score_threshold)


def validate_cucumber_masks(
    dataset_root: Path,
    output_root: Path = Path("outputs/quickstart/stage2"),
    score_threshold: float = 0.7,
) -> dict[str, Any]:
    environment = check_stage2_environment()
    if not environment["modules"]["numpy"] or not environment["modules"]["PIL"]:
        raise RuntimeError("Mask validation requires numpy and Pillow.")

    import numpy as np
    from PIL import Image

    dataset_root = dataset_root.resolve()
    output_root = output_root.resolve()
    instances = _read_json(dataset_root / "annotations" / "instances.json")
    manifest_path = output_root / "annotations" / "cucumber_masks.json"
    if not manifest_path.is_file():
        return {
            "valid": False,
            "mask_count": 0,
            "required_mask_count": len(instances["annotations"]),
            "issues": [f"Missing mask manifest: {manifest_path}"],
            "warnings": [],
        }
    manifest = _read_json(manifest_path)
    images_by_id = {image["id"]: image for image in instances["images"]}
    issues: list[str] = []
    warnings: list[str] = []
    for entry in manifest["masks"]:
        image = images_by_id.get(entry["image_id"])
        mask_path = output_root / entry["file_name"]
        if image is None:
            issues.append(f"Unknown image_id: {entry['image_id']}")
            continue
        if not mask_path.is_file():
            issues.append(f"Missing mask: {entry['file_name']}")
            continue
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        if mask.shape != (image["height"], image["width"]):
            issues.append(f"Mask size mismatch: {entry['file_name']}")
        if not mask.any():
            issues.append(f"Empty mask: {entry['file_name']}")
        if entry["sam_score"] < score_threshold:
            issues.append(f"Low SAM2 score: {entry['file_name']}")
        if _touches_boundary(mask):
            warnings.append(f"Mask touches image boundary: {entry['file_name']}")
        if entry["image_area_fraction"] > 0.3:
            warnings.append(f"Mask covers over 30% of image: {entry['file_name']}")

    required_count = sum(
        annotation["category_id"] == 1 for annotation in instances["annotations"]
    )
    if len(manifest["masks"]) != required_count:
        issues.append(
            f"Expected {required_count} cucumber masks, found {len(manifest['masks'])}"
        )
    return {
        "valid": not issues,
        "mask_count": len(manifest["masks"]),
        "required_mask_count": required_count,
        "issues": issues,
        "warnings": warnings,
    }


def _write_mask_preview(
    image: Any,
    mask: Any,
    bbox: list[int],
    output_path: Path,
    image_module: Any,
    image_draw_module: Any,
) -> None:
    import numpy as np

    base = np.asarray(image).copy()
    color = np.zeros_like(base)
    color[:, :, 1] = 255
    blended = base.copy()
    blended[mask] = (base[mask] * 0.45 + color[mask] * 0.55).astype(np.uint8)
    preview = image_module.fromarray(blended)
    x, y, width, height = bbox
    image_draw_module.Draw(preview).rectangle(
        (x, y, x + width, y + height), outline=(255, 32, 32), width=10
    )
    preview.save(output_path, quality=90)


def _touches_boundary(mask: Any) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _resolve_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _mark_stage2_complete(
    dataset_root: Path, output_root: Path, mask_manifest: dict[str, Any]
) -> None:
    if output_root != dataset_root:
        _write_json(
            output_root / "manifest.json",
            {
                "schema_version": "1.0",
                "input_dataset": str(dataset_root),
                "stage2": {
                    "status": "complete",
                    "output_root": str(output_root),
                    "generator": mask_manifest["generator"],
                    "model_config": mask_manifest["model_config"],
                    "checkpoint_name": mask_manifest["checkpoint_name"],
                    "mask_count": len(mask_manifest["masks"]),
                },
            },
        )
        return
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = _read_json(manifest_path)
    manifest["inputs"]["cucumber_mask"] = "generated_sam2"
    manifest["stage2"] = {
        "status": "complete",
        "generator": mask_manifest["generator"],
        "model_config": mask_manifest["model_config"],
        "checkpoint_name": mask_manifest["checkpoint_name"],
        "mask_count": len(mask_manifest["masks"]),
    }
    manifest["next_step"] = {
        "stage": 3,
        "name": "Condition-based synthesis and AISFormer export",
        "required_output": "Synthetic images, masks, and amodal annotations",
    }
    _write_json(manifest_path, manifest)
