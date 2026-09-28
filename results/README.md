# Results

Summary of the model-comparison study. Full per-run parameters and metrics for all 49
experiments are in
[`metrics/experiment_full_parameters_and_results.csv`](metrics/experiment_full_parameters_and_results.csv).

The primary metric is **PR-AUC** (area under the Precision-Recall curve), chosen because the
task is extremely imbalanced (~1.5–2% damaged points), which makes accuracy uninformative.

---

## Best Overall Model

**PointNet++ (SSG) + boundary_aware loss**

| Metric | Value |
|---|---|
| PR-AUC | **0.9201** |
| F1-Score | 0.8543 |
| mIoU | 0.8358 |
| Training epochs | 172 |

---

## Best Result per Architecture

| Architecture | Best Loss | PR-AUC |
|---|---|---|
| PointNet++ | boundary_aware | **0.9201** |
| PointNet++ (Transfer) | transfer_boundary_aware | 0.9123 |
| DGCNN | lovasz_boundary | 0.9058 |
| KPConv | boundary_aware | 0.8893 |
| PointNet | boundary_aware | 0.8591 |
| Point Transformer | boundary_aware | 0.8254 |
| PointNet (Transfer) | transfer_lovasz_boundary | 0.7044 |

---

## Loss Function Ranking (Average PR-AUC across architectures)

| Rank | Loss | Avg PR-AUC |
|---|---|---|
| 1 | boundary_aware | 0.8795 ± 0.0376 |
| 2 | lovasz_boundary | 0.8717 ± 0.0418 |
| 3 | crossentropy | 0.8281 ± 0.0830 |
| 4 | ce_dice | 0.8223 ± 0.0813 |

---

## Key Takeaways

- The **boundary_aware** loss was the most reliable objective, achieving both the best single
  result and the highest average across architectures — confirming that boundary points are
  where damage segmentation is hardest.
- **PointNet++ (SSG)** was the strongest architecture; its hierarchical set abstraction
  captured local damage geometry better than the alternatives at the 8 GB VRAM budget.
- Transfer learning helped PointNet++ modestly but hurt the PointNet baseline.
- Metric correlations: PR-AUC↔F1 = 0.901, PR-AUC↔mIoU = 0.812.
