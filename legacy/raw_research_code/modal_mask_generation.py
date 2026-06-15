import argparse
import logging
import os
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from custom_utils import (
    ensure_directories_exist,
    get_bbox_from_mask,
    get_image_paths_from_folder,
    random_sample_leaf_paths,
    resize_image_and_masks,
    save_image,
    mask_to_polygon,
)
from amodal_utils import (
    calculate_leaf_location,
    leaf_size_initialization,
    resize_leaf_to_target_ratio,
    adjust_leaves_to_occlusion,
    overlap_dual_leaves,
    merge_and_crop_leaf,
    get_amodal_masks,
    generate_annotation,
    save_processed_masks,
    create_occlusion_ratio_list,
)
from coco_json import initialize_coco_json, save_coco_json
from config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUT_ROOT,
    DEFAULT_SAMPLE_LIMITS,
    DEFAULT_TARGET_SIZE,
    HYPERPARAMETERS,
    resolve_input_paths,
    resolve_output_paths,
)


def merge_leaf_to_cucumber(cucumber_image, leaf_image, cucumber_mask, position, occlusion_ratio, initial_leaf_ratio):
    cucumber_bbox = get_bbox_from_mask(cucumber_mask)
    leaf_location = calculate_leaf_location(cucumber_bbox, position)
    leaf_image = leaf_size_initialization(cucumber_mask, leaf_image, initial_leaf_ratio)
    resized_leaf_image = resize_leaf_to_target_ratio(cucumber_mask, leaf_image, leaf_location, occlusion_ratio)
    merged_image, leaf_mask = merge_and_crop_leaf(cucumber_image, resized_leaf_image, leaf_location)
    return merged_image, leaf_mask


def merge_multi_leaves_to_cucumber(cucumber_image, leaf_image1, leaf_image2, cucumber_mask, position, occlusion_ratio, initial_leaf_ratio):
    cucumber_bbox = get_bbox_from_mask(cucumber_mask)
    leaf_location1 = calculate_leaf_location(cucumber_bbox, "top")
    leaf_location2 = calculate_leaf_location(cucumber_bbox, "bottom")

    leaf_image1 = leaf_size_initialization(cucumber_mask, leaf_image1, initial_leaf_ratio)
    leaf_image2 = leaf_size_initialization(cucumber_mask, leaf_image2, initial_leaf_ratio)

    leaf_location1, leaf_location2 = adjust_leaves_to_occlusion(
        cucumber_mask, leaf_image1, leaf_image2, leaf_location1, leaf_location2, occlusion_ratio
    )
    merged_image1, leaf_mask1 = merge_and_crop_leaf(cucumber_image, leaf_image1, leaf_location1)
    merged_image2, leaf_mask2 = merge_and_crop_leaf(merged_image1, leaf_image2, leaf_location2)
    final_leaf_mask = cv2.bitwise_or(leaf_mask1, leaf_mask2)
    return merged_image2, final_leaf_mask


def synthesize_images(
    cucumber_image_path,
    cucumber_mask_path,
    leaf_image_paths,
    position,
    occlusion_ratio,
    initial_leaf_ratio,
    save_dir=None,
    global_image_id=0,
    target_size=(768, 1024),
    multi_leaves=0,
):
    cucumber_image = read_image(cucumber_image_path, cv2.IMREAD_UNCHANGED, "Cucumber image")
    leaf_image = read_image(leaf_image_paths[0], cv2.IMREAD_UNCHANGED, "Leaf image")
    require_alpha(leaf_image, leaf_image_paths[0])
    cucumber_mask = read_image(cucumber_mask_path, cv2.IMREAD_GRAYSCALE, "Cucumber mask")

    if multi_leaves in (1, 2):
        if len(leaf_image_paths) < 2:
            raise ValueError("multi_leaves requires at least two leaf images")
        leaf_image2 = read_image(leaf_image_paths[1], cv2.IMREAD_UNCHANGED, "Leaf image (second)")
        require_alpha(leaf_image2, leaf_image_paths[1])
        if multi_leaves == 1:
            merged_image, leaf_mask = merge_multi_leaves_to_cucumber(
                cucumber_image,
                leaf_image,
                leaf_image2,
                cucumber_mask,
                position,
                occlusion_ratio,
                initial_leaf_ratio,
            )
        else:
            overlapped_leaves = overlap_dual_leaves(cucumber_mask, leaf_image, leaf_image2, initial_leaf_ratio)
            merged_image, leaf_mask = merge_leaf_to_cucumber(
                cucumber_image,
                overlapped_leaves,
                cucumber_mask,
                position,
                occlusion_ratio,
                initial_leaf_ratio,
            )
    else:
        merged_image, leaf_mask = merge_leaf_to_cucumber(
            cucumber_image, leaf_image, cucumber_mask, position, occlusion_ratio, initial_leaf_ratio
        )

    resized_image, resized_masks = resize_image_and_masks(merged_image, [cucumber_mask, leaf_mask], target_size=target_size)
    amodal_mask, leaf_mask = resized_masks

    cucumber_image_name = os.path.basename(str(cucumber_image_path))
    merged_image_name = f"{os.path.splitext(cucumber_image_name)[0]}_merged_{global_image_id:06d}_{occlusion_ratio}.png"
    resized_image_path = save_image(save_dir, merged_image_name, resized_image)

    return resized_image_path, amodal_mask, leaf_mask


def generate_coco_annotation(coco_json, amodal_mask, modal_mask, leaf_mask, global_image_id, global_annotation_id, merged_image_path):
    cucumber_annotation = generate_annotation(
        amodal_mask=amodal_mask,
        modal_mask=modal_mask,
        global_id=global_annotation_id,
        image_id=global_image_id,
        category_id=1,
        occluder_segm=mask_to_polygon(leaf_mask) if leaf_mask is not None else [],
    )
    coco_json["annotations"].append(cucumber_annotation)
    global_annotation_id += 1

    leaf_annotation = generate_annotation(
        amodal_mask=leaf_mask,
        modal_mask=None,
        global_id=global_annotation_id,
        image_id=global_image_id,
        category_id=2,
    )
    coco_json["annotations"].append(leaf_annotation)
    global_annotation_id += 1

    image_info = {
        "id": global_image_id,
        "width": int(amodal_mask.shape[1]),
        "height": int(amodal_mask.shape[0]),
        "file_name": os.path.basename(str(merged_image_path)),
    }
    coco_json["images"].append(image_info)

    return coco_json, global_annotation_id


def process_amodal_images_and_masks(
    cucumber_image_path,
    leaf_cropped_image_paths,
    cucumber_mask_path,
    save_dir,
    mask_save_dir,
    coco_json,
    global_image_id,
    global_annotation_id,
    position,
    occlusion_ratio,
    initial_leaf_ratio,
    multi_leaves=0,
    target_size=(768, 1024),
):
    merged_image_path, amodal_mask, leaf_mask = synthesize_images(
        cucumber_image_path,
        cucumber_mask_path,
        leaf_cropped_image_paths,
        position,
        occlusion_ratio,
        initial_leaf_ratio,
        save_dir,
        global_image_id,
        target_size=target_size,
        multi_leaves=multi_leaves,
    )

    modal_mask, overlap_mask = get_amodal_masks(amodal_mask, leaf_mask)
    save_processed_masks(amodal_mask, overlap_mask, modal_mask, leaf_mask, os.path.basename(str(merged_image_path)), mask_save_dir)

    coco_json, global_annotation_id = generate_coco_annotation(
        coco_json,
        amodal_mask,
        modal_mask,
        leaf_mask,
        global_image_id,
        global_annotation_id,
        merged_image_path,
    )

    global_image_id += 1
    return coco_json, global_image_id, global_annotation_id

def setup_logger(level):
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="[%(levelname)s] %(message)s",
    )
    return logging.getLogger("occlusion-mask-generation")


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def read_image(path, flags, label):
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"{label} not found or unreadable: {path}")
    return image


def require_alpha(image, path):
    if image is None:
        raise ValueError(f"Leaf image unreadable: {path}")
    if len(image.shape) < 3 or image.shape[2] < 4:
        raise ValueError(f"Leaf image must be RGBA (with alpha channel): {path}")

def str2bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in ("1", "true", "t", "yes", "y"):
        return True
    if value in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")


def parse_size(value):
    if value is None:
        return None
    value = value.lower().replace("x", ",")
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Target size must be 'width,height' or 'widthxheight'.")
    return (int(parts[0]), int(parts[1]))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Amodal/modal mask generation for occluded cucumbers.")
    parser.add_argument("--dataset_type", type=str, choices=["train", "valid", "debugging"], default=None)
    parser.add_argument("--position", type=str, choices=["top", "middle", "bottom", "random"], default=None)
    parser.add_argument("--multi_leaves", type=int, choices=[0, 1, 2], default=None)
    parser.add_argument("--random_ratio", type=str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--sample_limit", type=int, default=None)
    parser.add_argument("--target_size", type=parse_size, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry_run", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")

    parser.add_argument("--data_root", type=str, default=None, help="Root folder that contains data/splitted/*")
    parser.add_argument("--out_root", type=str, default=None, help="Root folder for outputs")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--mask_save_dir", type=str, default=None)
    parser.add_argument("--json_dir", type=str, default=None)
    return parser.parse_args()


def require_dir(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def main():
    args = parse_arguments()

    logger = setup_logger(args.log_level)
    set_seed(args.seed)

    if args.dataset_type is not None:
        HYPERPARAMETERS["dataset_type"] = args.dataset_type
    if args.position is not None:
        HYPERPARAMETERS["position"] = args.position
    if args.multi_leaves is not None:
        HYPERPARAMETERS["multi_leaves"] = args.multi_leaves
    if args.random_ratio is not None:
        HYPERPARAMETERS["random_ratio"] = args.random_ratio

    dataset_type = HYPERPARAMETERS["dataset_type"]

    data_root = Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT
    out_root = Path(args.out_root) if args.out_root else DEFAULT_OUT_ROOT

    input_paths = resolve_input_paths(data_root)
    output_paths = resolve_output_paths(out_root)

    if args.save_dir:
        output_paths["save_dir"] = Path(args.save_dir)
    if args.mask_save_dir:
        output_paths["mask_save_dir"] = Path(args.mask_save_dir)
    if args.json_dir:
        output_paths["json_dir"] = Path(args.json_dir)

    cucumber_images_dir = input_paths["cucumber_images_dir"] / dataset_type
    cucumber_masks_dir = input_paths["cucumber_masks_dir"] / dataset_type
    leaf_cropped_dir = input_paths["leaf_cropped_dir"] / dataset_type

    require_dir(cucumber_images_dir, "Cucumber images dir")
    require_dir(cucumber_masks_dir, "Cucumber masks dir")
    require_dir(leaf_cropped_dir, "Leaf crops dir")

    save_dir = output_paths["save_dir"]
    mask_save_dir = output_paths["mask_save_dir"]
    json_dir = output_paths["json_dir"]

    ensure_directories_exist([str(save_dir), str(mask_save_dir), str(json_dir)])

    logger.info(f"Dataset: {dataset_type}")
    logger.info(f"Inputs: images={cucumber_images_dir}, masks={cucumber_masks_dir}, leaves={leaf_cropped_dir}")
    logger.info(f"Outputs: images={save_dir}, masks={mask_save_dir}, json={json_dir}")

    if args.dry_run:
        logger.info("Dry-run enabled. Skipping generation.")
        return
    sample_limit = args.sample_limit if args.sample_limit is not None else DEFAULT_SAMPLE_LIMITS.get(dataset_type, 5)
    if sample_limit <= 0:
        raise ValueError("sample_limit must be > 0")    position = HYPERPARAMETERS["position"]
    multi_leaves = HYPERPARAMETERS["multi_leaves"]
    random_ratio = HYPERPARAMETERS["random_ratio"]
    ratios = HYPERPARAMETERS["r_settings"]
    proportions = HYPERPARAMETERS["r_proportions"]
    initial_leaf_ratio = HYPERPARAMETERS["initial_leaf_ratio"]
    sort = HYPERPARAMETERS["sort"]
    occlusion_ratio = HYPERPARAMETERS["occlusion_ratio"]

    target_size = args.target_size if args.target_size is not None else DEFAULT_TARGET_SIZE

    if random_ratio:
        occlusion_ratio_list = create_occlusion_ratio_list(sample_limit, ratios, proportions)

    def get_cucumber_masks(mask_dir, image_name):
        cucumber_masks = []
        for mask_file in os.listdir(mask_dir):
            if mask_file.startswith(image_name) and "_0_" in mask_file:
                cucumber_masks.append(os.path.join(mask_dir, mask_file))
        return cucumber_masks

    cucumber_image_paths = get_image_paths_from_folder(str(cucumber_images_dir), sort=sort)
    coco_json = initialize_coco_json()

    valid_cucumber_paths = []
    for cucumber_image_path in cucumber_image_paths:
        image_name = os.path.splitext(os.path.basename(cucumber_image_path))[0]
        cucumber_mask_paths = get_cucumber_masks(str(cucumber_masks_dir), image_name)
        if len(cucumber_mask_paths) > 0:
            valid_cucumber_paths.append((cucumber_image_path, cucumber_mask_paths))

    total_cucumber_images = len(valid_cucumber_paths)
    if total_cucumber_images == 0:
        raise ValueError("No valid cucumber masks were found for the given images.")

    sample_per_cucumber = sample_limit // total_cucumber_images
    remaining_samples = sample_limit % total_cucumber_images

    logger.info(f"Valid cucumber images: {total_cucumber_images}")
    logger.info(f"Samples per cucumber: {sample_per_cucumber}, extra samples: {remaining_samples}")

    leaf_cropped_image_paths = get_image_paths_from_folder(str(leaf_cropped_dir))
    if not leaf_cropped_image_paths:
        raise ValueError("No leaf crop images found.")
    if multi_leaves in (1, 2) and len(leaf_cropped_image_paths) < 2:
        raise ValueError("multi_leaves requires at least two leaf images in cropped_leaves.")

    logger.info(f"Leaf crops: {len(leaf_cropped_image_paths)}")
    with tqdm(total=sample_limit, desc="Generated samples", unit="samples") as pbar:
        sample_count = 0
        global_image_id, global_annotation_id = 0, 0

        for cucumber_idx, (cucumber_image_path, cucumber_mask_paths) in enumerate(valid_cucumber_paths):
            samples_for_this_cucumber = sample_per_cucumber
            if cucumber_idx < remaining_samples:
                samples_for_this_cucumber += 1

            cucumber_sample_count = 0

            for cucumber_mask_path in cucumber_mask_paths:
                sampled_leaf_paths = random_sample_leaf_paths(leaf_cropped_image_paths, samples_for_this_cucumber)

                for idx, leaf_cropped_image_path in enumerate(sampled_leaf_paths):
                    if sample_count >= sample_limit or cucumber_sample_count >= samples_for_this_cucumber:
                        break

                    pair_idx = -(idx + 1)
                    leaves_cropped_image_paths = [leaf_cropped_image_path, sampled_leaf_paths[pair_idx]]

                    if random_ratio:
                        occlusion_ratio = occlusion_ratio_list[sample_count] / 100.0

                    coco_json, global_image_id, global_annotation_id = process_amodal_images_and_masks(
                        cucumber_image_path=cucumber_image_path,
                        leaf_cropped_image_paths=leaves_cropped_image_paths,
                        cucumber_mask_path=cucumber_mask_path,
                        save_dir=str(save_dir),
                        mask_save_dir=str(mask_save_dir),
                        coco_json=coco_json,
                        global_image_id=global_image_id,
                        global_annotation_id=global_annotation_id,
                        position=position,
                        occlusion_ratio=occlusion_ratio,
                        initial_leaf_ratio=initial_leaf_ratio,
                        multi_leaves=multi_leaves,
                        target_size=target_size,
                    )

                    sample_count += 1
                    cucumber_sample_count += 1
                    pbar.update(1)

                    if sample_count >= sample_limit:
                        break

                if sample_count >= sample_limit:
                    break

            if sample_count >= sample_limit:
                break

    output_json_path = Path(json_dir) / "dataset.json"
    save_coco_json(coco_json, str(output_json_path))


if __name__ == "__main__":
    main()










