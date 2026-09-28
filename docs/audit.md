# Repository Audit

This document records the state of the original research working directory before
reorganization into `clean_repo/`. It exists so a technical reviewer can trust that the
reorganized result faithfully preserves the original functionality, and can see exactly
which issues were found and addressed.

---

## 1. Original Repository Structure

The original repository was a research working directory built on top of a
PointNet / PointNet++ template and progressively extended for a 3D point-cloud damage
detection study. The meaningful file groups are summarized below.

### Training scripts (root)

| File | Purpose |
|---|---|
| `train_dmgseg.py` | **Obsolete.** First training iteration, 100% commented-out legacy code. |
| `train_dmgseg2.py` | Second iteration. Leading import block is commented out (dead code). |
| `train_dmgseg3.py` – `train_dmgseg8.py` | Successive training iterations adding loss functions, balancing, augmentation, transfer learning. |
| `train_dmgseg9.py` | **Most complete / reproducible** training script. Basis for `src/training/trainer.py`. |
| `train_classification.py`, `train_partseg.py`, `train_semseg.py`, `train_cube_segmentation.py` | Template scripts inherited from the base PointNet framework; not part of the damage-detection pipeline. |
| `train_*.sh` | Shell drivers for the 49-experiment grid. |

### Inference / evaluation scripts (root)

| File | Purpose |
|---|---|
| `infer_from_stl.py` | STL → point cloud → partition → model inference → colorized viewer. Basis for `src/evaluation/inference.py`. |
| `infer_interactive.py` | Interactive patch-level inference viewer. |
| `seg_inf.py` | Segmentation inference helper. |
| `eval_point_and_bbox.py` | Per-point + bounding-box (DBSCAN/AABB) evaluation. Basis for `src/evaluation/bbox_eval.py`. |
| `eval_val_split.py` | Evaluation over the validation split. |
| `figure_pipeline.py` | Generates pipeline figures for the paper. |
| `deploy_operating_regime.py`, `infer_pr_curves.py`, `compute_pr_operating_regimes.py` | PR-curve / operating-regime analysis. |

### Supporting code

| Path | Purpose |
|---|---|
| `models/` | All network architectures (see section 4). |
| `data_utils/` | Data loaders. **Note the `DmgCompDataLoader.py` issue in section 2.** |
| `provider.py` | Point-cloud augmentation functions (scale, shift, jitter, rotate, dropout). |
| `visualizer/` | Rendering utilities inherited from the base framework. |
| `data/damaged_components/` | NPZ point-cloud patches + `synsetoffset2category.txt` + `train_test_split/`. |
| `data/raw_meshes/` | Raw STL component meshes. |
| `model_experiments/` | Experiment results (CSV/JSON) and the analysis report. |

---

## 2. Critical Issue — `DmgCompDataLoader.py` Repurposing

`data_utils/DmgCompDataLoader.py` was **repurposed as a standalone inference script** and
**no longer exports the `DamageDataset` class** that every training script imports from it
(`from data_utils.DmgCompDataLoader import DamageDataset`).

In its current state the file:
- Contains hardcoded absolute paths (`NPZ_DIR`, `MODEL_PATH`).
- Loads a model and runs patch-level stochastic-voting inference at import time.
- Would raise `ImportError` for `DamageDataset` if any training script were run.

**Resolution in the clean repo:** `DamageDataset` is reconstructed from its usage
signatures across `train_dmgseg4.py`–`train_dmgseg9.py` and placed in
`src/data/dataset.py`, fully documented.

---

## 3. Absolute Paths Found

The following machine-specific absolute paths (`/home/miguelcruz/...`) were found in the
original source files. These prevent a third party from running the code and are removed
in the clean repo (replaced by CLI arguments / config parameters).

| File | Absolute path(s) |
|---|---|
| `data_utils/DmgCompDataLoader.py` | `NPZ_DIR = /home/miguelcruz/Documents/AS4RCSD/...`; `MODEL_PATH = /home/miguelcruz/Documents/3DL-Based/...` |
| `infer_from_stl.py` | `DAMAGED_STL_PATH = /home/miguelcruz/Documents/3DL-Based/data/raw_meshes/001001.STL`; `MODEL_PATH = /home/miguelcruz/Documents/3DL-Based/...` |
| `seg_inf.py` | `/home/miguelcruz/...` model/data paths |
| `figure_pipeline.py` | `DATA_ROOT = /home/miguelcruz/Documents/3DL-Based/data/damaged_components`; `MODEL_PATH = /home/miguelcruz/Documents/3DL-Based/...` |
| `infer_interactive.py` | `DATA_ROOT = /home/miguelcruz/Documents/3DL-Based/data/damaged_components`; `MODEL_PATH = /home/miguelcruz/Documents/3DL-Based/...` |
| `eval_val_split.py` | `DATA_ROOT = /home/miguelcruz/Documents/3DL-Based /data/damaged_components` |
| `deploy_operating_regime.py` | `NPZ_DIR` / `MODEL_PATH` under `/home/miguelcruz/...` (active and commented). |

---

## 4. Model Architectures

Architectures found in the original `models/` directory (one-line descriptions):

| Model file | Description |
|---|---|
| `pointnet_dmg_seg.py` | PointNet damage segmentation — global-feature MLP over the point set. |
| `pointnet2_dmg_seg_ssg.py` | PointNet++ single-scale grouping — hierarchical set abstraction. |
| `dgcnn_dmg_seg.py` | Dynamic Graph CNN — dynamic edge-convolution over k-NN graphs. |
| `kpconv_dmg_seg*.py` | Kernel Point Convolution variants (`_simple`, `_lite`, `_ultra_lite`, `_fixed`, `_v2`). `_v2` is the final version. |
| `point_transformer_dmg_seg*.py` | Point Transformer variants (`_simple`, `_fixed`, `_v2`, `_v3`, `_ultra_lite`). `_v3` is the final version. |
| `minkowski_dmg_seg.py` | Minkowski sparse-convolution segmentation (experimental, not in final grid). |
| `pointnet_transfer_dmg_seg.py` | PointNet with S3DIS encoder transfer learning. |
| `pointnet2_transfer_dmg_seg.py` | PointNet++ with S3DIS encoder transfer learning. |
| `pointnet2_utils.py` | Shared set-abstraction / feature-propagation layers. |
| `pointnet_utils.py` | Shared PointNet T-Net / feature-transform utilities. |
| `pointnet2_cls_*.py`, `pointnet2_part_seg_*.py`, `pointnet2_sem_seg*.py`, `pointnet_cls.py`, `pointnet_part_seg.py`, `pointnet_sem_seg.py` | Template models inherited from the base PointNet framework; not part of the damage-detection pipeline. |

The clean repo keeps the five final architectures plus the two transfer variants (see
`src/models/README.md`).

---

## 5. Obsolete vs. Active Training Scripts

| Script | Status |
|---|---|
| `train_dmgseg.py` | **Obsolete** — file is 100% commented-out legacy code. Not carried into the clean repo. |
| `train_dmgseg2.py` | Contains a fully commented-out leading import block (dead code); superseded by later iterations. |
| `train_dmgseg3.py` – `train_dmgseg8.py` | Active but superseded — intermediate iterations. |
| `train_dmgseg9.py` | **Active / canonical** — most complete and reproducible; extracted into `src/training/trainer.py`. |
