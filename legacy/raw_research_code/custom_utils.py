# ?대?吏? 諛붿슫??諛뺤뒪瑜???踰덉뿉 濡쒕뱶?섎뒗 ?⑥닔
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json, random
import os
from typing import List, Tuple

def load_images_and_boxes(image_folder: str, label_folder: str, max_images: int = 10) -> Tuple[List[np.ndarray], List[List], List[str]]:
    images = []
    all_boxes = []
    image_names = []
    processed_count = 0

    # ?덉씠釉??뚯씪 紐⑸줉???뺣젹???곹깭濡?媛?몄샂
    label_files = sorted(os.listdir(label_folder))

    for label_file in label_files:
        if processed_count >= max_images:
            break

        # JSON ?덉씠釉??뚯씪 ?쎄린
        with open(os.path.join(label_folder, label_file), 'r', encoding='utf-8') as f:
            annotation = json.load(f)

        # ?대?吏 ?뚯씪紐?媛?몄삤湲?
        image_name = annotation['description']['image']
        image_path = os.path.join(image_folder, image_name)

        # ?대?吏 ?쎄린
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 諛붿슫??諛뺤뒪 ?뺣낫 媛?몄삤湲?
        boxes = []
        for result in annotation['result']:
            if result['type'] == 'bbox':
                x, y, w, h = result['x'], result['y'], result['w'], result['h']
                boxes.append([x, y, x + w, y + h])

        if not boxes:
            continue

        images.append(image_rgb)
        all_boxes.append(boxes)
        image_names.append(image_name)

        processed_count += 1

    return images, all_boxes, image_names

# 留덉뒪?ъ? ?대?吏瑜??쒓컖?뷀븯???⑥닔
def plot_masks_on_images(images: List[np.ndarray], masks: List[np.ndarray], image_names: List[str]):
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    
    for idx, (image, mask_set, image_name) in enumerate(zip(images, masks, image_names)):
        for i in range(mask_set.shape[0]):  # 媛??대?吏???щ윭 媛쒖쓽 留덉뒪?ш? ?덉쓣 ???덉쓬
            ax = axes[idx // 5, idx % 5]
            ax.imshow(image)
            ax.imshow(mask_set[i], alpha=0.5, cmap='jet')
            ax.set_title(f"Image: {image_name} - Mask {i + 1}")
            ax.axis('off')

    plt.tight_layout()
    plt.show()


# 留덉뒪?щ? ?대?吏 ?뚯씪濡???ν븯???⑥닔
def save_masks(masks: List[np.ndarray], image_names: List[str], output_folder: str, target_size: Tuple[int, int] = (384, 512)):
    os.makedirs(output_folder, exist_ok=True)

    for mask_set, image_name in zip(masks, image_names):
        for i, mask in enumerate(mask_set):
            # 留덉뒪??由ъ궗?댁쫰
            resized_mask = cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
            
            # 留덉뒪?????寃쎈줈 ?ㅼ젙
            mask_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_mask_{i + 1}.png")
            
            # 留덉뒪?????
            cv2.imwrite(mask_path, (resized_mask * 255).astype(np.uint8))  # 留덉뒪?щ? ?댁쭊 ?대?吏濡????

            
def resize_images_and_masks(images: List[np.ndarray], masks: List[np.ndarray], target_size: Tuple[int, int] = (384, 512)) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    resized_images = []
    resized_masks = []

    for image, mask_set in zip(images, masks):
        # ?대?吏 由ъ궗?댁쫰
        resized_image = cv2.resize(image, (target_size[1], target_size[0]))  # (width, height)濡??ㅼ젙
        resized_images.append(resized_image)

        # 留덉뒪??由ъ궗?댁쫰
        resized_mask_set = []
        for mask in mask_set:
            # 留덉뒪?щ? uint8濡?蹂?섑븯??由ъ궗?댁쫰
            mask_uint8 = mask.astype(np.uint8)  # bool ??낆쓣 uint8濡?蹂??
            resized_mask = cv2.resize(mask_uint8, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
            resized_mask_set.append(resized_mask)

        resized_masks.append(np.array(resized_mask_set))

    return resized_images, resized_masks

def ensure_directories_exist(directories):
    """
    二쇱뼱吏?寃쎈줈 紐⑸줉???대떦?섎뒗 ?붾젆?곕━媛 議댁옱?섏? ?딆쓣 寃쎌슦 ?앹꽦?⑸땲??

    :param directories: ?붾젆?곕━ 寃쎈줈 紐⑸줉 (由ъ뒪???먮뒗 ?뺤뀛?덈━)
    """
    if isinstance(directories, dict):
        directories = directories.values()  # ?뺤뀛?덈━ 媛믩뱾??紐⑸줉?쇰줈 蹂??
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"?붾젆?곕━ ?앹꽦?? {directory}")
        else:
            print(f"?붾젆?곕━媛 ?대? 議댁옱?⑸땲?? {directory}")


def save_image(save_dir, file_name, img, flag=0):
    key = 'Mask' if flag == 0 else 'Image'

    os.makedirs(save_dir, exist_ok=True)  # ???寃쎈줈媛 ?놁쑝硫??앹꽦
    image_save_path = os.path.join(save_dir, file_name)
    cv2.imwrite(image_save_path, img)
    #print(f"{key} ??λ맖: {image_save_path}")
    return image_save_path

def get_image_paths_from_folder(folder_path, extensions=['.jpg', '.png'], sort=False):
    image_paths = []
    normalized_exts = [ext.lower() for ext in extensions]
    for filename in os.listdir(folder_path):
        lower_name = filename.lower()
        if any(lower_name.endswith(ext) for ext in normalized_exts):
            image_paths.append(os.path.join(folder_path, filename))
    if sort:
        image_paths.sort()  # ?뺣젹 ?섑뻾

    return image_paths

def random_sample_leaf_paths(leaf_paths, k):
    """
    ??寃쎈줈 以묒뿉??k媛쒖쓽 ?섑뵆???쒕뜡?섍쾶 ?좏깮
    """
    if k <= 0:
        return []
    if len(leaf_paths) <= k:
        return leaf_paths  # ??寃쎈줈媛 k媛쒕낫???곸쑝硫??꾩껜 諛섑솚
    return random.sample(leaf_paths, k)

def get_bbox_from_mask(mask):
    """
    留덉뒪?ъ뿉???쇰컲?곸씤 [x_min, y_min, x_max, y_max] ?뺤떇??bbox 異붿텧
    """
    coords = np.column_stack(np.where(mask == 255))  # 留덉뒪??醫뚰몴 異붿텧
    if coords.size == 0:  # 鍮?留덉뒪??泥섎━
        return [0, 0, 0, 0]
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return [x_min, y_min, x_max, y_max]

def get_coco_bbox_from_mask(mask):
    """
    留덉뒪?ъ뿉??COCO ?щ㎎ [x_min, y_min, width, height] ?뺤떇??bbox 異붿텧
    """
    x_min, y_min, x_max, y_max = get_bbox_from_mask(mask)
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def mask_to_polygon(binary_mask, min_contour_area=10):
    """
    Convert a binary mask to a COCO segmentation polygon, filtering by area.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_contour_area:  # 理쒖냼 ?ш린 ?꾪꽣留?
            contour = contour.flatten().tolist()
            if len(contour) > 4:  # 理쒖냼?쒖쓽 ?대━怨??먯씠 ?꾩슂
                polygons.append(contour)
    #print(f"polygon len : {len(polygons)}")
    return polygons


def resize_image_and_masks(image, masks, target_size=(768, 1024)):
    target_width, target_height = target_size
    original_height, original_width = image.shape[:2]
    
    # ?대?吏 由ъ궗?댁쫰
    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    # 留덉뒪??由ъ궗?댁쫰
    resized_masks = [
        cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        for mask in masks
    ]
    
    return resized_image, resized_masks
