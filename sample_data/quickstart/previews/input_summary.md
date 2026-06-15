# Quickstart input summary

The cucumber boxes are supplied annotations and replace YOLO output for this sample.
Leaf cutouts already contain alpha masks and replace leaf YOLO + SAM output.

| Scene | Source ID | Cucumber bbox | EXIF orientation |
|---|---|---|---|
| scene_001 | `V003_3_3_1_2_4_2_2_1_0_0_20221019_5107_20240422195047` | `[1353, 1212, 451, 1916]` | 6 |
| scene_002 | `V003_3_3_1_2_4_2_2_1_0_0_20221019_5143_20240422195049` | `[925, 1224, 386, 2021]` | 6 |
| scene_003 | `V003_3_3_1_2_4_2_2_1_0_0_20221019_5284_20240422195057` | `[1107, 1212, 468, 1775]` | 6 |

Visual bbox checks:

- [scene_001](bbox_scene_001.svg)
- [scene_002](bbox_scene_002.svg)
- [scene_003](bbox_scene_003.svg)

Usable leaf cutouts: **8**

Next step: generate one full cucumber mask per scene with SAM2.
