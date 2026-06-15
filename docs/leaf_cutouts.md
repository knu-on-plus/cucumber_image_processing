# Leaf Cutout Preparation

The synthesis stage requires RGBA leaf assets:

```text
leaf_cutouts/*.png
leaf_assets.json
```

Each PNG must contain the leaf RGB crop and an alpha channel where non-zero
pixels mark the visible leaf. The quickstart already includes prepared leaf
cutouts, so this guide is only needed when building a dataset from new raw
images.

## Recommended Flow

1. Train or provide a leaf detector.
2. Run the detector on raw images and keep leaf bounding boxes.
3. Use each leaf bounding box as a SAM2 box prompt.
4. Convert the SAM2 mask into an RGBA crop.
5. Place the resulting PNG files under the dataset's `leaf_cutouts/` directory.
6. Create `leaf_assets.json` with one entry per PNG.

## YOLOv10 Setup

YOLO weights are user supplied. The repository does not redistribute the
project-specific detector weights because the source training dataset cannot
be redistributed.

Clone the official YOLOv10 implementation into `external/`:

```bash
mkdir -p external
git clone https://github.com/THU-MIG/yolov10.git external/yolov10
python -m pip install -e external/yolov10
```

Use your trained weights to export leaf detections. The normalized preparation
contract only needs image path, leaf class, confidence, and `xyxy` bbox
coordinates. A simple intermediate file can look like this:

```json
[
  {
    "image_file": "images/scene_001.jpg",
    "class_name": "leaf",
    "score": 0.91,
    "bbox_xyxy": [102, 215, 456, 668]
  }
]
```

## SAM2 Mask To RGBA Crop

Use the same SAM2 installation created by `./scripts/setup_omg.sh`. For each
leaf bbox:

1. Load the source RGB image.
2. Run SAM2 with the bbox as a box prompt.
3. Crop the RGB image to the bbox.
4. Crop the SAM2 mask to the same bbox.
5. Save an RGBA PNG where `alpha = mask * 255`.

The output PNG should have a transparent background outside the leaf. Keep at
least two valid leaf cutouts because paper Conditions 2 and 3 use two leaf
instances.

## Leaf Asset Manifest

The current quickstart manifest format is:

```json
{
  "leaf_assets": [
    {
      "id": 1,
      "file_name": "leaf_cutouts/leaf_001.png",
      "width": 256,
      "height": 256,
      "alpha_nonzero_fraction": 0.42
    }
  ]
}
```

`python pipeline.py validate-inputs` checks that every listed file exists and
has a usable alpha channel.
