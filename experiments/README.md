# Experiments

This directory documents the model-comparison study that underpins the results reported in
this repository. Full per-run numbers are in
[`../results/metrics/experiment_full_parameters_and_results.csv`](../results/metrics/experiment_full_parameters_and_results.csv).

---

## The 49-Experiment Grid

The study evaluates **7 architectures × 7 loss functions = 49 experiments**, each trained to
convergence with early stopping on PR-AUC.

**Architectures (7):**

1. PointNet
2. PointNet++ (SSG)
3. DGCNN
4. KPConv (v2)
5. Point Transformer (v3)
6. PointNet + Transfer (S3DIS encoder)
7. PointNet++ + Transfer (S3DIS encoder)

**Loss functions (7):**

1. `crossentropy` — weighted cross-entropy
2. `focal` — focal loss for class imbalance
3. `tversky` — generalized Dice with FP/FN control
4. `ce_dice` — cross-entropy + Dice hybrid
5. `lovasz` — Lovász-Softmax (direct IoU optimization)
6. `boundary_aware` — up-weights healthy/damaged boundary points
7. `lovasz_boundary` — Lovász + boundary hybrid

> For the two transfer-learning architectures, the loss variants are the `transfer_*`
> counterparts (e.g. `transfer_boundary_aware`), applied on top of a pre-trained S3DIS
> encoder.

Every combination trains a model and is evaluated with the same per-point metric suite
(PR-AUC, F1, per-class IoU, mIoU, Dice). PR-AUC is the primary comparison metric because of
the extreme class imbalance (~1.5–2% damaged points).

---

## Reproducing an Experiment

Each experiment corresponds to a `model` + `loss_function` pairing that can be expressed as a
YAML config and launched through the training entry point:

```bash
python scripts/train.py --config configs/pointnet2_boundary_aware.yaml
```

`configs/pointnet2_boundary_aware.yaml` reproduces the best-performing run
(PointNet++ + boundary_aware). To reproduce another cell of the grid, copy that config and
change the `model` and `loss_function` fields. Evaluate any trained run with:

```bash
python scripts/evaluate.py --log_dir logs/<experiment> --macro-eval
```

---

## Hardware Requirements

- **GPU:** NVIDIA RTX 2080 (8 GB VRAM).
- The 8 GB VRAM budget constrained batch size and point count. KPConv and Point Transformer
  are the most memory-intensive architectures; the "lite" model variants in the original
  codebase were created to fit within this budget.
- A CUDA-capable GPU is strongly recommended for training. Inference on a single component is
  feasible on CPU but slow.

---

## Full Results

The complete parameter-and-results table for all 49 runs is stored at:

```
results/metrics/experiment_full_parameters_and_results.csv
```

See [`../results/README.md`](../results/README.md) for the summarized key results.
