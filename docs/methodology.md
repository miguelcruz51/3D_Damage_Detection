# Methodology

This document describes the research pipeline, the class-imbalance challenge that shaped the
loss-function design, and the dataset used for training and evaluation.

---

## Research Pipeline

The system detects surface damage on metallic engineering components. It operates end-to-end
from raw geometry to a localized damage prediction:

```
STL mesh
   │  (Open3D: load + uniform point sampling, up to ~200K points)
   ▼
Point cloud (XYZ + estimated normals)
   │  (spatial grid-based partitioning into fixed-size local patches)
   ▼
Local patches (e.g. 2048 points each)
   │  (pc_normalize: center at centroid, scale to unit sphere)
   ▼
Normalized patch  [B, C, N]   (C = 3 for XYZ, 6 with normals)
   │  (deep model: PointNet / PointNet++ / DGCNN / KPConv / PointTransformer)
   ▼
Per-point log-probabilities  [B, N, 2]  (0 = healthy, 1 = damaged)
   │  (threshold → per-point binary labels; stitch patches back to full cloud)
   ▼
Per-point damage segmentation
   │  (DBSCAN clustering of damaged points → axis-aligned bounding boxes)
   ▼
Bounding-box damage localization
```

**Stage notes:**

1. **STL → point cloud.** The raw component is a triangle mesh. It is converted to an
   unordered point set by uniform surface sampling, and per-point surface normals are
   estimated.
2. **Local partition.** A full component cloud is far larger than a model input. It is
   divided into fixed-size local patches on a 3D grid so each patch fits the network.
3. **Normalize.** Each patch is centered on its centroid and scaled so its farthest point
   lies on the unit sphere (`pc_normalize`). The **same** normalization is applied during
   training and inference to keep the input distribution consistent.
4. **Model.** A point-cloud segmentation network produces a per-point log-probability over
   the two classes.
5. **Per-point segmentation.** During inference, stochastic voting (multiple sub-samples per
   patch) averages probabilities per point, then a threshold yields binary predictions.
6. **Bounding-box localization.** Damaged points are clustered with DBSCAN, and each cluster
   is summarized by an axis-aligned bounding box (AABB), giving a macro-level, region-based
   result that is evaluated with IoU-based box matching.

---

## The Class-Imbalance Problem

Damage covers only a small fraction of a component's surface. In this dataset roughly
**1.5–2% of points are labeled damaged**, with the remaining ~98% healthy.

Under such extreme imbalance, a plain cross-entropy classifier can reach very high accuracy
by predicting "healthy" almost everywhere, while completely missing the damage that actually
matters. Accuracy is therefore misleading, and specialized objectives are required.

**Why specialized loss functions were needed:**

- **Weighted cross-entropy** raises the penalty on the rare damaged class.
- **Focal loss** down-weights easy, well-classified healthy points so training focuses on the
  hard, rare damaged points.
- **Tversky loss** generalizes Dice and lets us tune the false-positive / false-negative
  trade-off directly.
- **Lovász-Softmax** optimizes IoU directly, which aligns with the imbalanced segmentation
  metric of interest.
- **Boundary-aware loss** up-weights points near the healthy/damaged boundary, where errors
  are most common and most costly. This was the single best-performing objective.
- **Hybrid losses** (CE+Dice, Lovász+Boundary) combine complementary signals.

Because accuracy is uninformative here, the **Precision-Recall AUC (PR-AUC)** is used as the
primary evaluation metric, supported by F1, per-class IoU, and mIoU.

---

## Dataset Structure

- **16 component types.** Categories `Component1`–`Component16`, mapped from
  `synsetoffset2category.txt` to numeric ids `001`–`016`.
- **NPZ patches.** Each patch is a `.npz` file with:
  - `points`  — `float32 (N, 3)` XYZ coordinates
  - `normals` — `float32 (N, 3)` surface normals
  - `labels`  — `int32 (N,)` per-point labels (0 = healthy, 1 = damaged)
- **Filename convention.** `{component_id}{version}{patch_index}{damage_flag}.npz`
  (e.g. `001001000901.npz` → component 001, version 001, patch 0009, damage_flag 1). A
  trailing `1` indicates the patch contains damaged points; a trailing `0` indicates a
  predominantly healthy patch.
- **Train / test split.** ShapeNet-style JSON lists under
  `data/damaged_components/train_test_split/`:
  - `shuffled_train_file_list.json`
  - `shuffled_val_file_list.json`
  - `shuffled_test_file_list.json`

  Each file holds relative paths, so splits are disjoint by construction and loaded directly
  by `DamageDataset`.
