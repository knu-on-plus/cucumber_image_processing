# Pipeline Overview

This repo contains several notebook-based stages. The main runnable entrypoint is `modal_mask_generation.py`.

## Stages

1. YOLO-based object detection (optional)
   - `yolo_detection/get_oi_bbox.ipynb`
   - `yolo_detection/get_leaves_bbox.ipynb`
   - Output: bounding boxes for cucumber/leaf candidates

2. Leaf mask generation (optional)
   - `sam2_mask_gen/sam2_mask_save.ipynb`
   - `sam2_mask_gen/sam2_mask_visualization.ipynb`
   - Output: leaf masks and cropped leaves

3. Synthetic dataset generation (main)
   - `modal_mask_generation.py`
   - Inputs: cucumber images, cucumber masks, cropped leaves
   - Output: synthesized amodal images, modal masks, COCO json

4. Post-processing / validation (optional)
   - `data_processing/*.ipynb`
   - JSON validation, split utilities, filtering

## Notes

- Notebook paths contain many legacy absolute paths. Use search/replace to update to your local layout.
- The CLI entrypoint is Windows- and Linux-friendly via `--data_root` and `--out_root`.
