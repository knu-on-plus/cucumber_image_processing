# config.py
from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(os.environ.get("OMG_DATA_ROOT", REPO_ROOT / "data"))
DEFAULT_OUT_ROOT = Path(os.environ.get("OMG_OUT_ROOT", REPO_ROOT / "outputs"))

HYPERPARAMETERS = {
    "dataset_type": "debugging",  # [train, valid, debugging]
    "image_index_start": 0,
    "position": "middle",  # top, middle, bottom, random
    "occlusion_ratio": 0.5,
    "multi_leaves": 0,  # 0: single leaf, 1: dual leaves, 2: overlap dual leaves
    "random_ratio": True,
    "initial_leaf_ratio": (0.20, 0.4),
    "r_settings": [50, 70, 90],
    "r_proportions": [5, 4, 1],
    "sort": True,
}

DEFAULT_SAMPLE_LIMITS = {
    "train": 10000,
    "valid": 4000,
    "debugging": 5,
}

DEFAULT_TARGET_SIZE = (768, 1024)  # (width, height)


def resolve_input_paths(data_root: Path):
    data_root = Path(data_root)
    return {
        "cucumber_images_dir": data_root / "splitted" / "images",
        "cucumber_masks_dir": data_root / "splitted" / "masks",
        "leaf_cropped_dir": data_root / "splitted" / "cropped_leaves",
    }


def resolve_output_paths(out_root: Path):
    out_root = Path(out_root)
    return {
        "save_dir": out_root / "amodal_images",
        "mask_save_dir": out_root / "modal_masks",
        "json_dir": out_root / "amodal_info",
    }


# Backward-compatible string paths
_input_paths = resolve_input_paths(DEFAULT_DATA_ROOT)
_output_paths = resolve_output_paths(DEFAULT_OUT_ROOT)

INPUT_PATHS = {k: str(v) + os.sep for k, v in _input_paths.items()}
OUTPUT_PATHS = {k: str(v) for k, v in _output_paths.items()}
