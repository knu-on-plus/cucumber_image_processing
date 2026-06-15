# Synthesis Configuration

The synthesis engine uses a JSON preset so the quickstart command stays small
while condition-level experiments remain explicit and reproducible.

## Presets

| Preset | Purpose | Output |
|---|---|---|
| `configs/quickstart.json` | One representative ratio per scene and condition | `outputs/quickstart/synthesis` |
| `configs/paper-full.json` | Every requested ratio for every non-baseline condition | `outputs/quickstart/synthesis_paper_full` |
| `configs/paper-weighted.json` | Exact `50/75/90 = 5:4:1` sampling for Conditions 1-3 | `outputs/quickstart/synthesis_paper_weighted` |

Run either preset with:

```bash
conda activate OMG
# --overwrite replaces an existing generated output directory.
python pipeline.py synthesize \
  --config configs/paper-full.json \
  --overwrite
```

Use `--output-root PATH` to override the configured output directory.

## Common Fields

| Field | Meaning |
|---|---|
| `dataset_root` | Normalized Stage 1 and Stage 2 input directory |
| `mask_root` | Stage 2 mask manifest and mask directory |
| `output_root` | Stage 3 output directory |
| `mode` | `representative`, `full-grid`, or `weighted-sample` |
| `conditions` | Ordered subset of the four implemented paper conditions |
| `occlusion_ratios` | Targets used by non-baseline conditions |
| `ratio_sampling_weights` | Weights aligned with `occlusion_ratios` |
| `sample_count` | Total weighted layout count, excluding gamma expansion |
| `samples_per_condition` | Weighted layout count per selected condition |
| `baseline_occlusion_ratio` | Target used by the centered baseline |
| `gamma_values` | Positive gamma augmentation values |
| `target_size` | Final `[width, height]` |
| `occlusion_tolerance` | Maximum accepted absolute target error |
| `seed` | Reproducible leaf-selection offset |

In `representative` mode, requested ratios cycle across input scenes. In
`full-grid` mode, every requested ratio is evaluated for every selected
non-baseline condition.

In `weighted-sample` mode, baseline is intentionally excluded because it uses
one fixed ratio. Conditions 1-3 are sampled with an exact weight distribution,
so `samples_per_condition` must be a multiple of
`sum(ratio_sampling_weights)`. With the paper weights `[5, 4, 1]`, use `10`,
`20`, `100`, and so on.

## Condition Parameters

All size and position values are relative to the cucumber bounding box.

### Baseline and Condition 1

- `leaf_height_range`: minimum and maximum leaf-height scale searched
- `search_steps`: number of candidate sizes evaluated
- `positions`: ordered position cycle, normally `top` and `bottom`
- `position_fractions`: vertical bbox fractions corresponding to `positions`

### Condition 2

- `leaf_height_ratio`: fixed height of each separate leaf
- `outer_fraction_range`: searched top-leaf fraction; the bottom is mirrored
- `search_steps`: number of candidate positions evaluated
- `max_leaf_overlap_ratio`: maximum overlap permitted between leaf instances

### Condition 3

- `positions` and `position_fractions`: attached-pair placement cycle
- `leaf_height_range`: minimum and maximum pair-height scale searched
- `search_steps`: number of candidate sizes evaluated
- `attachment_overlap_ratio`: horizontal overlap used to attach the two leaves

The loader rejects unknown conditions, invalid ranges, non-positive gamma
values, and malformed condition parameters before synthesis starts.

## Custom Example

This example runs only Conditions 1 and 3 at severe occlusion and uses the
original brightness:

```json
{
  "name": "severe-occlusion",
  "dataset_root": "sample_data/quickstart",
  "mask_root": "outputs/quickstart/stage2",
  "output_root": "outputs/quickstart/synthesis_severe",
  "synthesis": {
    "mode": "full-grid",
    "conditions": [
      "condition_1_single_top_or_bottom",
      "condition_3_two_attached_top_or_bottom"
    ],
    "occlusion_ratios": [0.75, 0.9],
    "gamma_values": [1.0],
    "target_size": [768, 1024],
    "occlusion_tolerance": 0.05,
    "seed": 0
  }
}
```

Omitted condition parameters use the defaults declared by the synthesis
engine. Increase `samples_per_condition` in `configs/paper-weighted.json` to
scale up while preserving the exact paper ratio distribution.
