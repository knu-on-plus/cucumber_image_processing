import json
import os, sys
import cv2
import numpy as np
from custom_utils import *
import random
import matplotlib.pyplot as plt


def calculate_leaf_location(cucumber_bbox, location='middle'):
    x_min, y_min, x_max, y_max = cucumber_bbox

    # ?ㅼ씠 以묒떖 醫뚰몴 怨꾩궛
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    # 媛??꾩튂蹂?醫뚰몴 怨꾩궛
    top = (center_x, int(y_min + (y_max - y_min) * 0.10))      # ?곷떒 以묒떖
    middle = (center_x, center_y) # 以묒븰
    bottom = (center_x, int(y_min + (y_max - y_min) * 0.80))   # ?섎떒 以묒떖

    # ?꾩튂 寃곗젙???곕Ⅸ 寃곌낵 諛섑솚
    if location == 'top':
        return top
    elif location == 'middle':
        return middle
    elif location == 'bottom':
        return bottom
    elif location == 'random':
        return random.choice([top, bottom])  # ?곷떒 ?먮뒗 ?섎떒 以??쒕뜡
    else:
        raise ValueError(f"Invalid location option: {location}")

def leaf_size_initialization(cucumber_mask, leaf_image, ratio = (0.35, 0.5)):
    min_leaf_size_ratio, max_leaf_size_ratio = ratio  # ?롮쓽 理쒕? 諛?理쒖냼 ?ш린 鍮꾩쑉
    max_leaf_h, max_leaf_w = int(cucumber_mask.shape[0] * max_leaf_size_ratio), int(cucumber_mask.shape[1] * max_leaf_size_ratio)
    min_leaf_h, min_leaf_w = int(cucumber_mask.shape[0] * min_leaf_size_ratio), int(cucumber_mask.shape[1] * min_leaf_size_ratio)

    # ???대?吏 ?ш린 ?쒗븳
    leaf_h, leaf_w = leaf_image.shape[:2]
    if leaf_h > max_leaf_h or leaf_w > max_leaf_w:
        leaf_image = cv2.resize(leaf_image, (max_leaf_w, max_leaf_h), interpolation=cv2.INTER_LINEAR)
    elif leaf_h < min_leaf_h or leaf_w < min_leaf_w:
        leaf_image = cv2.resize(leaf_image, (min_leaf_w, min_leaf_h), interpolation=cv2.INTER_LINEAR)

    return leaf_image

def resize_leaf_to_target_ratio(cucumber_mask, leaf_image, leaf_position, target_ratio):
    loss_rate = 0.05  # ?덉슜 ?ㅼ감
    leaf_x, leaf_y = leaf_position
    cucumber_area = np.sum(cucumber_mask == 255)  # ?ㅼ씠 留덉뒪??硫댁쟻 怨꾩궛

    leaf_h, leaf_w = leaf_image.shape[:2]
    max_iterations = 10
    iterations = 0

    while True:
        # ??留덉뒪???앹꽦 (?뚰뙆 梨꾨꼸 ?쒖슜)
        leaf_mask = (leaf_image[:, :, 3] > 0).astype(np.uint8) * 255

        # 以묒떖??leaf_position?쇰줈 ?대룞
        temp_leaf_mask = np.zeros_like(cucumber_mask)
        start_y = max(0, leaf_y - leaf_h // 2)
        start_x = max(0, leaf_x - leaf_w // 2)
        end_y = min(cucumber_mask.shape[0], start_y + leaf_h)
        end_x = min(cucumber_mask.shape[1], start_x + leaf_w)

        temp_leaf_mask[start_y:end_y, start_x:end_x] = leaf_mask[0:(end_y - start_y), 0:(end_x - start_x)]

        # ?꾩옱 寃뱀묠 ?곸뿭 怨꾩궛
        overlap_area = np.sum((temp_leaf_mask > 0) & (cucumber_mask > 0))
        current_ratio = overlap_area / cucumber_area

        # ?쒓컖??
        #visualize_resizing(cucumber_mask, temp_leaf_mask, leaf_position, overlap_area, current_ratio, iterations)

        # ?붾쾭源?異쒕젰
        #print(f"Iteration {iterations}: Overlap Area: {overlap_area}, Leaf Area: {leaf_h * leaf_w}, Current Ratio: {current_ratio:.4f}")

        # 紐⑺몴 鍮꾩쑉???꾨떖?섎㈃ 醫낅즺
        if abs(current_ratio - target_ratio) < loss_rate:
            #print(f"Target ratio achieved with current ratio: {current_ratio:.4f} after {iterations} iterations.")
            break

        # 諛섎났 珥덇낵 ??醫낅즺
        if iterations >= max_iterations:
            #print(f"Error: Maximum iterations ({max_iterations}) reached. Exiting.")
            break

        # ?ш린 議곗젙 鍮꾩쑉 怨꾩궛
        if current_ratio < target_ratio:
            scale_factor = min((target_ratio / current_ratio) ** 0.5, 1.1)
        else:
            scale_factor = max((target_ratio / current_ratio) ** 0.5, 0.9)

        # Resize???ш린 怨꾩궛
        new_h = max(1, int(leaf_h * scale_factor))
        new_w = max(1, int(leaf_w * scale_factor))

        # Resize ?섑뻾
        resized_leaf = cv2.resize(leaf_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # ?낅뜲?댄듃
        leaf_image = resized_leaf
        leaf_h, leaf_w = leaf_image.shape[:2]
        iterations += 1

    #print("Leaves resized to target ratio.")
    return leaf_image

def adjust_leaves_to_occlusion(cucumber_mask, leaf_image1, leaf_image2, leaf_location1, leaf_location2, target_ratio):
    def create_leaf_mask(leaf_image, leaf_position, mask_shape):
        leaf_h, leaf_w = leaf_image.shape[:2]
        temp_mask = np.zeros(mask_shape, dtype=np.uint8)

        # ?뚰뙆 梨꾨꼸濡???留덉뒪???앹꽦
        leaf_mask = (leaf_image[:, :, 3] > 0).astype(np.uint8) * 255

        # ??以묒떖??湲곗??쇰줈 ?쒖옉/??醫뚰몴 怨꾩궛
        leaf_x, leaf_y = leaf_position
        start_y = leaf_y - leaf_h // 2
        start_x = leaf_x - leaf_w // 2
        end_y = leaf_y + leaf_h // 2
        end_x = leaf_x + leaf_w // 2

        # 寃쎄퀎 議곌굔 泥섎━: ?ㅼ씠 ?대?吏瑜?踰쀬뼱?섎뒗 寃쎌슦 ?섎씪?닿린
        crop_start_y = max(0, -start_y)
        crop_start_x = max(0, -start_x)
        crop_end_y = leaf_h - max(0, end_y - mask_shape[0])
        crop_end_x = leaf_w - max(0, end_x - mask_shape[1])

        start_y = max(0, start_y)
        start_x = max(0, start_x)
        end_y = min(mask_shape[0], end_y)
        end_x = min(mask_shape[1], end_x)

        # ?ш린 ?쇱튂 ?뺤씤 ??蹂듭궗
        region_h = end_y - start_y
        region_w = end_x - start_x
        crop_h = crop_end_y - crop_start_y
        crop_w = crop_end_x - crop_start_x

        if region_h != crop_h or region_w != crop_w:
            # ?ш린 ?쇱튂?섎룄濡??섎씪??
            region_h = min(region_h, crop_h)
            region_w = min(region_w, crop_w)

            end_y = start_y + region_h
            end_x = start_x + region_w
            crop_end_y = crop_start_y + region_h
            crop_end_x = crop_start_x + region_w

        # 留덉뒪??蹂듭궗 諛??섎씪?닿린
        temp_mask[start_y:end_y, start_x:end_x] = leaf_mask[crop_start_y:crop_end_y, crop_start_x:crop_end_x]

        # 以묒떖???붾쾭源?
        #print(f"Leaf Position: {leaf_position}, Mask Center Adjusted to: ({leaf_x}, {leaf_y})")
        return temp_mask

    def move_leaf(leaf_position, direction, step, mask_shape):
        x, y = leaf_position
        height, width = mask_shape

        if direction == "up":
            new_y = max(0, y - step)  # ?곷떒 寃쎄퀎瑜?踰쀬뼱?섏? ?딅룄濡??쒗븳
        elif direction == "down":
            new_y = min(height - 1, y + step)  # ?섎떒 寃쎄퀎瑜?踰쀬뼱?섏? ?딅룄濡??쒗븳
        else:
            raise ValueError(f"Invalid direction: {direction}. Use 'up' or 'down'.")

        return (x, new_y)

    max_iterations = 20
    loss_rate = 0.05
    step_size = 10
    iterations = 0

    cucumber_area = np.sum(cucumber_mask > 0)

    while iterations < max_iterations:
        # ??留덉뒪???앹꽦
        leaf_mask1 = create_leaf_mask(leaf_image1, leaf_location1, cucumber_mask.shape)
        leaf_mask2 = create_leaf_mask(leaf_image2, leaf_location2, cucumber_mask.shape)

        # ????留덉뒪??蹂묓빀
        combined_leaf_mask = cv2.bitwise_or(leaf_mask1, leaf_mask2)

        # 寃뱀묠 鍮꾩쑉 怨꾩궛
        overlap_area = np.sum((cucumber_mask > 0) & (combined_leaf_mask > 0))
        current_ratio = overlap_area / cucumber_area

        #print(f"Iteration {iterations}: , Current Ratio: {current_ratio:.4f}")
        #visualize_shifting(cucumber_mask, leaf_mask1, leaf_mask2, leaf_location1, leaf_location2, iterations)

        # 紐⑺몴 鍮꾩쑉 ?꾨떖 ?щ? ?뺤씤
        if abs(current_ratio - target_ratio) <= loss_rate:
            #print(f"Target ratio met: {current_ratio:.4f}")
            break

        # 鍮꾩쑉???곕씪 以묒떖???대룞
        if current_ratio < target_ratio:
            leaf_location1 = move_leaf(leaf_location1, "down", step_size, cucumber_mask.shape)
            leaf_location2 = move_leaf(leaf_location2, "up", step_size, cucumber_mask.shape)
        else:
            leaf_location1 = move_leaf(leaf_location1, "up", step_size, cucumber_mask.shape)
            leaf_location2 = move_leaf(leaf_location2, "down", step_size, cucumber_mask.shape)

        iterations += 1

    #if iterations >= max_iterations:
        #print(f"Warning: Maximum iterations reached. Final ratio: {current_ratio:.4f}")

    return leaf_location1, leaf_location2

def overlap_dual_leaves(cucumber_mask, leaf_image1, leaf_image2, initial_leaf_ratio):
    leaf_image1 = leaf_size_initialization(cucumber_mask, leaf_image1, initial_leaf_ratio)
    leaf_image2 = leaf_size_initialization(cucumber_mask, leaf_image2, initial_leaf_ratio)
    # ?대?吏 ?ш린 鍮꾧탳
    h1, w1, _ = leaf_image1.shape
    h2, w2, _ = leaf_image2.shape

    if w1 < w2:  # leaf_image2媛 ???щ떎硫??꾩튂瑜?諛붽퓞
        leaf_image1, leaf_image2 = leaf_image2, leaf_image1
        h1, w1, h2, w2 = h2, w2, h1, w1

    # ??罹붾쾭???ш린 怨꾩궛
    canvas_width = w1 + w2 // 2
    canvas_height = max(h1, h2)

    # ?덈줈??罹붾쾭???앹꽦 (RGBA)
    overlapped_leaves = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)

    # leaf_image1 諛곗튂 (?쇱そ)
    x1_start, y1_start = 0, (canvas_height - h1) // 2
    overlapped_leaves[y1_start:y1_start + h1, x1_start:x1_start + w1, :] = leaf_image1

    # leaf_image2 諛곗튂 (?ㅻⅨ履?
    x2_start = max(0, min(w1 - w2 // 2, canvas_width - w2))  # ?ㅻⅨ履???珥덇낵 諛⑹?
    y2_start = max(0, (canvas_height - h2) // 2)

    for c in range(4):  # 梨꾨꼸蹂?蹂묓빀 (RGBA)
        overlapped_leaves[y2_start:y2_start + h2, x2_start:x2_start + w2, c] = np.where(
            leaf_image2[:, :, 3] > 0,  # ?뚰뙆 梨꾨꼸???덈뒗 寃쎌슦
            leaf_image2[:, :, c],
            overlapped_leaves[y2_start:y2_start + h2, x2_start:x2_start + w2, c]
        )
    return overlapped_leaves

def merge_and_crop_leaf(cucumber_image, resized_leaf_image, leaf_position):
    # ?롮쓽 以묒떖 醫뚰몴? ?ш린
    leaf_x, leaf_y = leaf_position
    leaf_h, leaf_w = resized_leaf_image.shape[:2]

    # ???대?吏??醫뚯륫 ?곷떒 醫뚰몴 怨꾩궛
    crop_x_start = max(0, leaf_x - leaf_w // 2)
    crop_y_start = max(0, leaf_y - leaf_h // 2)

    # ???대?吏瑜?寃쎄퀎??留욊쾶 ?먮Ⅴ湲?
    leaf_crop_start_y = max(0, -leaf_y + leaf_h // 2)  # ?롮쓽 ?쒖옉 Y 醫뚰몴
    leaf_crop_end_y = leaf_h - max(0, (leaf_y + leaf_h // 2) - cucumber_image.shape[0])
    leaf_crop_start_x = max(0, -leaf_x + leaf_w // 2)  # ?롮쓽 ?쒖옉 X 醫뚰몴
    leaf_crop_end_x = leaf_w - max(0, (leaf_x + leaf_w // 2) - cucumber_image.shape[1])

    cropped_leaf_image = resized_leaf_image[
        leaf_crop_start_y:leaf_crop_end_y,
        leaf_crop_start_x:leaf_crop_end_x
    ]

    # 寃곌낵 ?대?吏? 留덉뒪??珥덇린??
    merged_image = cucumber_image.copy()
    leaf_mask = np.zeros((cucumber_image.shape[0], cucumber_image.shape[1]), dtype=np.uint8)

    # ???대?吏 蹂묓빀
    cropped_h, cropped_w = cropped_leaf_image.shape[:2]
    for i in range(cropped_h):
        for j in range(cropped_w):
            # 寃쎄퀎 珥덇낵 諛⑹? 議곌굔
            if (crop_y_start + i >= cucumber_image.shape[0]) or (crop_x_start + j >= cucumber_image.shape[1]):
                #print("====== Warning: Leaf image out of bounds. ======")
                continue  # 寃쎄퀎瑜?踰쀬뼱??寃쎌슦 ?ㅽ궢

            if cropped_leaf_image[i, j, 3] > 0:  # ?щ챸?섏? ?딆? 寃쎌슦
                merged_image[crop_y_start + i, crop_x_start + j] = cropped_leaf_image[i, j, :3]
                leaf_mask[crop_y_start + i, crop_x_start + j] = 255  # ??留덉뒪???낅뜲?댄듃

    return merged_image, leaf_mask

def save_processed_masks(amodal_mask, overlap_mask, modal_mask, occluder_mask, image_name, mask_save_dir):
    base_name = os.path.splitext(image_name)[0]

    occluder_filename = f"{base_name}_occluder_mask.png"
    save_image(mask_save_dir, occluder_filename, occluder_mask)

    amodal_filename = f"{base_name}_amodal_mask.png"
    save_image(mask_save_dir, amodal_filename, amodal_mask)

    modal_filename = f"{base_name}_modal_mask.png"
    save_image(mask_save_dir, modal_filename, modal_mask)

    overlap_filename = f"{base_name}_overlap_mask.png"
    save_image(mask_save_dir, overlap_filename, overlap_mask.astype(np.uint8) * 255)


def get_amodal_masks(cucumber_mask, leaf_mask):
    # 寃뱀튂??遺遺?(?ㅼ씠 留덉뒪?ъ? ??留덉뒪?ш? ?숈떆??255??遺遺꾩쓣 異붿텧)
    overlap_mask = (cucumber_mask == 255) & (leaf_mask == 255)
    
    # Modal 留덉뒪???앹꽦 (寃뱀튂??遺遺꾩쓣 ?쒖쇅???ㅼ씠 留덉뒪??
    modal_mask = cucumber_mask.copy()
    modal_mask[overlap_mask] = 0  # overlap ?곸뿭??0?쇰줈 留뚮뱾??寃뱀튇 遺遺??쒓굅

    return modal_mask, overlap_mask


def generate_annotation(amodal_mask, modal_mask, global_id, image_id, category_id, occluder_segm=[]):
    amodal_segm = mask_to_polygon(amodal_mask)
    amodal_bbox = get_coco_bbox_from_mask(amodal_mask)
    
    process_mask = modal_mask if modal_mask is not None else amodal_mask  # 蹂댁씠???곸뿭???놁쑝硫??꾩껜 ?곸뿭(amodal_mask) ?ъ슜
    area = float(np.sum(process_mask == 255))
    segmentation = visible_segm = mask_to_polygon(process_mask)
    visible_bbox = get_coco_bbox_from_mask(process_mask)

    annotation = {
        "id": global_id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": 0,
        "amodal_bbox": amodal_bbox,
        "visible_bbox" : visible_bbox,
        "bbox": amodal_bbox,  # COCO ?щ㎎ BBox 怨꾩궛
        "area": area,
        "amodal_area": float(np.sum(amodal_mask == 255)),
        "amodal_segm": amodal_segm,
        "segmentation": amodal_segm,  # Segmentation polygon ?앹꽦
        "visible_segm": visible_segm,
        "background_objs_segm": [],  # 湲곕낯媛?
        "occluder_segm": occluder_segm,
    }
    return annotation

def create_occlusion_ratio_list(sample_limit, ratios, proportions):
    """
    ratios: 由ъ뒪?? e.g., [50, 75, 90]
    proportions: 由ъ뒪?? e.g., [5, 4, 1]
    """
    total_proportion = sum(proportions)
    counts = [int(sample_limit * p / total_proportion) for p in proportions]
    
    # 留덉?留?鍮꾩쑉???⑥? ?섑뵆 紐⑤몢 ?좊떦
    counts[-1] = sample_limit - sum(counts[:-1])
    
    occlusion_ratios = []
    for ratio, count in zip(ratios, counts):
        occlusion_ratios.extend([ratio] * count)
    
    random.shuffle(occlusion_ratios)
    return occlusion_ratios

def visualize_resizing(cucumber_mask, temp_leaf_mask, leaf_position, overlap_area, current_ratio, iteration):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # ?먮낯 cucumber_mask ?쒓컖??
    ax[0].imshow(cucumber_mask, cmap="gray")
    ax[0].scatter(leaf_position[0], leaf_position[1], color='red', label='Leaf Position')
    ax[0].set_title("Original Cucumber Mask")
    ax[0].legend()

    # Overlap???곸뿭 ?쒓컖??
    overlap_visual = cucumber_mask.copy()
    overlap_visual[temp_leaf_mask > 0] = 255  # Overlap ?곸뿭 媛뺤“
    ax[1].imshow(overlap_visual, cmap="Reds")
    ax[1].scatter(leaf_position[0], leaf_position[1], color='red', label='Leaf Position')

    # 怨꾩궛??以묒떖???쒓컖??
    ax[1].set_title(f"Iteration {iteration}\nOverlap Area: {overlap_area}, Ratio: {current_ratio:.4f}")
    ax[1].legend()
    plt.tight_layout()
    plt.show()

def visualize_shifting(cucumber_mask, leaf_mask1, leaf_mask2, leaf_location1, leaf_location2, iteration):
    # 寃뱀묠 ?곸뿭 怨꾩궛
    overlap_area = cv2.bitwise_and(cucumber_mask, cv2.bitwise_or(leaf_mask1, leaf_mask2))
    
    # ?쒓컖??以鍮?
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # ?꾩껜 諛곌꼍 ?앹꽦
    visualization = np.zeros_like(cucumber_mask, dtype=np.uint8)
    visualization[cucumber_mask > 0] = 128  # ?ㅼ씠 媛앹껜???뚯깋
    visualization[leaf_mask1 > 0] = 200  # ??1? 諛앹? ?뚯깋
    visualization[leaf_mask2 > 0] = 255  # ??2???곗깋
    visualization[overlap_area > 0] = 50  # 寃뱀튂???곸뿭? ?대몢???됱쑝濡??쒖떆
    
    # ?쒓컖??
    ax.imshow(visualization, cmap="gray")
    ax.scatter(leaf_location1[0], leaf_location1[1], color='red', label='Leaf 1 Center')
    ax.scatter(leaf_location2[0], leaf_location2[1], color='blue', label='Leaf 2 Center')
    ax.set_title(f"Iteration {iteration}: Overlap Visualization")
    ax.legend()
    
    plt.show()


