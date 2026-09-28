# Models

This directory contains all model architectures used in the 3D damage detection experiments.
Each model performs per-point binary segmentation: given a point cloud patch `[B, C, N]`, it outputs
log-probabilities `[B, N, 2]` (class 0 = healthy, class 1 = damaged).

## Architecture Comparison

Results are taken from the 49-experiment grid (7 architectures × 7 loss functions).
The **Best Loss** and **Best PR-AUC** columns reflect each architecture's single best-performing run.

| Model | Architecture Type | Best Loss | Best PR-AUC | Training Epochs | Notes |
|---|---|---|---|---|---|
| `pointnet_dmg_seg.py` | PointNet (global feature MLP) | boundary_aware | 0.8591 | 118 | Fastest convergence among baselines |
| `pointnet2_dmg_seg_ssg.py` | PointNet++ SSG (hierarchical SA) | boundary_aware | **0.9201** | 172 | Best overall; F1 0.8543, mIoU 0.8358 |
| `dgcnn_dmg_seg.py` | DGCNN (dynamic edge-conv graph) | lovasz_boundary | 0.9058 | 187 | Strong second; best non-transfer model after PN2 |
| `kpconv_dmg_seg_v2.py` | KPConv (kernel point convolution) | boundary_aware | 0.8893 | 183 | Good spatial locality; higher memory use |
| `point_transformer_dmg_seg_v3.py` | Point Transformer (self-attention) | boundary_aware | 0.8254 | 182 | Lowest among finals; attention overhead |
| `pointnet2_transfer_dmg_seg.py` | PointNet++ SSG + Transfer (S3DIS) | transfer_boundary_aware | 0.9123 | 192 | Encoder pre-trained on S3DIS sem-seg |
| `pointnet_transfer_dmg_seg.py` | PointNet + Transfer (S3DIS) | transfer_lovasz_boundary | 0.7044 | 193 | Transfer did not help PointNet baseline |

## Loss Functions

All loss functions are in `losses.py`. The table below shows average PR-AUC across all architectures.

| Loss | Avg PR-AUC | Description |
|---|---|---|
| `boundary_aware` | 0.8795 | Up-weights points near damage boundaries |
| `lovasz_boundary` | 0.8717 | Lovász-Softmax + boundary emphasis |
| `crossentropy` | 0.8281 | Weighted cross-entropy baseline |
| `ce_dice` | 0.8223 | Cross-entropy + Dice hybrid |
| `focal` | — | Focusing parameter for class imbalance |
| `tversky` | — | Generalized Dice; controls FP/FN trade-off |
| `lovasz` | — | Directly optimizes IoU |

## Files

| File | Description |
|---|---|
| `pointnet_dmg_seg.py` | PointNet damage segmentation model |
| `pointnet2_dmg_seg_ssg.py` | PointNet++ SSG damage segmentation model |
| `dgcnn_dmg_seg.py` | DGCNN damage segmentation model |
| `kpconv_dmg_seg_v2.py` | KPConv v2 damage segmentation model |
| `point_transformer_dmg_seg_v3.py` | Point Transformer v3 damage segmentation model |
| `pointnet_transfer_dmg_seg.py` | PointNet with S3DIS transfer learning |
| `pointnet2_transfer_dmg_seg.py` | PointNet++ with S3DIS transfer learning |
| `losses.py` | All loss function implementations |
| `pointnet2_utils.py` | Set Abstraction and Feature Propagation layers |
