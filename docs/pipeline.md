# Public pipeline contract

The public interface is limited to three stages. Internally, each stage may
contain smaller reproducibility steps.

Supported release scope:

- Prepared annotation/RGBA input path: implemented
- Cucumber SAM2 segmentation: implemented
- Representative and full-grid synthesis: implemented
- Exact `5:4:1` weighted synthesis preset: implemented

## Stage 1: Detection and input preparation

Stage 1's required final output is **images + cucumber bboxes + leaf assets**.
How those outputs are obtained is selectable.

Implemented cucumber path:

- Existing bbox annotations, as used by the quickstart sample

Planned cucumber path:

- User-trained YOLO detection output

Implemented leaf path:

- Existing RGBA cutouts, as used by the quickstart sample

Planned leaf path:

- User-trained YOLO detections, then SAM2 cutout generation

See [`leaf_cutouts.md`](leaf_cutouts.md) for the expected RGBA leaf asset
contract and a YOLO + SAM2 preparation guide.

Normalized outputs:

- Images with an explicit EXIF orientation policy
- COCO-style cucumber bbox annotations
- A leaf asset manifest
- Human-readable bbox overlays and an input summary

```bash
python pipeline.py prepare-sample
python pipeline.py validate-inputs
```

The quickstart therefore begins after cucumber detection and after leaf
segmentation. It does not begin after cucumber segmentation.

Generated masks and synthetic datasets are written under `outputs/quickstart/`
so the provided sample inputs remain unchanged.

## Stage 2: SAM2 mask generation

For the quickstart, this stage accepts each normalized cucumber bbox as a SAM2
box prompt and produces:

- One full cucumber mask for each source image
- Visual mask overlays and mask quality notes

The planned full raw-data path will also receive normalized leaf detections and
produce RGBA leaf cutouts.

```bash
./scripts/setup_omg.sh
conda activate OMG
./scripts/download_sam2_checkpoint.sh
python pipeline.py segment-cucumbers \
  --checkpoint checkpoints/sam2_hiera_large.pt \
  --model-config sam2_hiera_l.yaml
python pipeline.py validate-masks
```

The project-specific YOLO model weights are user supplied because their source
training datasets cannot be redistributed. SAM2 should be installed from its
official repository and its official checkpoint supplied separately.

## Stage 3: Synthesis and AISFormer-compatible export

This stage reproduces every paper condition:

- Baseline, conditions 1, 2, and 3
- Occlusion ratios `50%`, `75%`, and `90%`
- Paper target distribution `5:4:1` through `weighted-sample` presets
- Gamma values `0.3`, `1.0`, and `3.0`
- Amodal and visible/modal masks
- Separate annotations for separate occluding leaves
- AISFormer-compatible `inmodal_seg` and `inmodal_bbox` fields

The stage also records achieved occlusion ratios so generated samples can
be rejected or regenerated when they fall outside tolerance.

```bash
python pipeline.py synthesize \
  --config configs/quickstart.json \
  --overwrite
python pipeline.py validate-synthesis
```

Representative mode creates one layout for every condition and scene, cycles
the target ratios across the three scenes, and emits dark/original/bright
gamma variants. Full-grid mode evaluates every condition/ratio combination:

```bash
python pipeline.py synthesize \
  --config configs/paper-full.json \
  --overwrite
```

The JSON preset can select a subset of paper conditions and tune ratios,
gamma values, target size, tolerance, positions, leaf-size search ranges,
separation constraints, and attachment overlap. See
[`configuration.md`](configuration.md).

Weighted mode creates exact `5:4:1` ratio distributions for Conditions 1-3:

```bash
python pipeline.py synthesize \
  --config configs/paper-weighted.json \
  --overwrite
```

AISFormer compatibility follows the official loader contract:

- `segmentation`, `bbox`: amodal cucumber
- `inmodal_seg`, `inmodal_bbox`: visible cucumber
- Leaf instances remain separate for Conditions 2 and 3

This repository currently stops at the export boundary.
