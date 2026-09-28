"""
Plotting and 3D visualization utilities for damage-detection evaluation.

Matplotlib plot-savers are consolidated from ``test_single_component.py`` and
``train_dmgseg8.py`` / ``train_dmgseg9.py``:
    - plot_point_pr_curve
    - plot_point_roc_curve
    - plot_confusion_matrix
    - save_history_plots

Open3D-based interactive visualizers are consolidated from
``eval_point_and_bbox.py``:
    - visualize_points_and_bboxes

Open3D is imported lazily/guarded so this module remains importable in headless
environments where Open3D is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for saving plots
import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

# Guard Open3D so headless environments can still import this module.
try:
    import open3d as o3d  # type: ignore
    _HAS_OPEN3D = True
except Exception:  # pragma: no cover - import guard
    o3d = None  # type: ignore
    _HAS_OPEN3D = False

ArrayLike = np.ndarray


def plot_point_pr_curve(
    gt_labels: ArrayLike,
    pred_scores: ArrayLike,
    save_path: Optional[str] = None,
) -> None:
    """Plot the per-point precision-recall curve.

    Args:
        gt_labels: (N,) binary ground-truth labels.
        pred_scores: (N,) predicted damage probabilities/scores.
        save_path: If given, save the figure to this path; otherwise show it.
    """
    precision, recall, _ = precision_recall_curve(gt_labels, pred_scores)
    ap = average_precision_score(gt_labels, pred_scores)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Per-Point Precision-Recall Curve (AP={ap:.3f})')
    plt.grid(True, linestyle='--', alpha=0.6)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_point_roc_curve(
    gt_labels: ArrayLike,
    pred_scores: ArrayLike,
    save_path: Optional[str] = None,
) -> None:
    """Plot the per-point ROC curve.

    Args:
        gt_labels: (N,) binary ground-truth labels.
        pred_scores: (N,) predicted damage probabilities/scores.
        save_path: If given, save the figure to this path; otherwise show it.
    """
    fpr, tpr, _ = roc_curve(gt_labels, pred_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Per-Point ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    gt_labels: ArrayLike,
    pred_labels: ArrayLike,
    class_names: Optional[Sequence[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot the confusion matrix for binary damage predictions.

    Args:
        gt_labels: (N,) binary ground-truth labels.
        pred_labels: (N,) binary predicted labels.
        class_names: Optional display labels for the two classes.
        save_path: If given, save the figure to this path; otherwise show it.
    """
    cm = confusion_matrix(gt_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_history_plots(history: dict, save_dir: str) -> None:
    """Save training-history plots (loss/accuracy/F1/PR-AUC/IoU/LR/time).

    Produces a combined ``training_summary.png`` grid plus one individual
    ``{metric}_curve.png`` per tracked metric.

    Args:
        history: Dict mapping metric names (e.g. ``train_loss``, ``test_f1``,
            ``learning_rate``, ``epoch_time``) to per-epoch value lists.
        save_dir: Directory to write plots to (created if missing).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_color = '#1f77b4'
    test_color = '#ff7f0e'
    system_color = '#2ca02c'

    try:
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))

        # 1. Loss comparison
        if len(history.get("train_loss", [])) > 0 and len(history.get("test_loss", [])) > 0:
            epochs = range(1, len(history["train_loss"]) + 1)
            axes[0, 0].plot(epochs, history["train_loss"], color=train_color, linewidth=2,
                            marker='o', markersize=3, label="Training")
            axes[0, 0].plot(epochs, history["test_loss"], color=test_color, linewidth=2,
                            marker='s', markersize=3, label="Test")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].set_title("Loss Comparison", fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Accuracy comparison
        if len(history.get("train_accuracy", [])) > 0 and len(history.get("test_accuracy", [])) > 0:
            epochs = range(1, len(history["train_accuracy"]) + 1)
            axes[0, 1].plot(epochs, history["train_accuracy"], color=train_color, linewidth=2,
                            marker='o', markersize=3, label="Training")
            axes[0, 1].plot(epochs, history["test_accuracy"], color=test_color, linewidth=2,
                            marker='s', markersize=3, label="Test")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Accuracy")
            axes[0, 1].set_title("Accuracy Comparison", fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Test F1 and PR-AUC
        if len(history.get("test_f1", [])) > 0 and len(history.get("test_pr_auc", [])) > 0:
            epochs = range(1, len(history["test_f1"]) + 1)
            ax1 = axes[1, 0]
            ax2 = ax1.twinx()
            line1 = ax1.plot(epochs, history["test_f1"], color='red', linewidth=2,
                             marker='o', markersize=3, label="F1-Score")
            line2 = ax2.plot(epochs, history["test_pr_auc"], color='blue', linewidth=2,
                             marker='s', markersize=3, label="PR-AUC")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("F1-Score", color='red')
            ax2.set_ylabel("PR-AUC", color='blue')
            ax1.set_title("Test F1-Score & PR-AUC", fontweight='bold')
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
            ax1.grid(True, alpha=0.3)

        # 4. IoU metrics
        if len(history.get("test_iou_damaged", [])) > 0:
            epochs = range(1, len(history["test_iou_damaged"]) + 1)
            axes[1, 1].plot(epochs, history["test_iou_damaged"], color='red', linewidth=2,
                            marker='o', markersize=3, label="IoU Damaged")
            axes[1, 1].plot(epochs, history["test_iou_intact"], color='green', linewidth=2,
                            marker='s', markersize=3, label="IoU Intact")
            axes[1, 1].plot(epochs, history["test_miou"], color=test_color, linewidth=2,
                            marker='^', markersize=3, label="Mean IoU")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("IoU Score")
            axes[1, 1].set_title("IoU Metrics", fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        # 5. Learning Rate
        if len(history.get("learning_rate", [])) > 0:
            epochs = range(1, len(history["learning_rate"]) + 1)
            axes[2, 0].plot(epochs, history["learning_rate"], color='purple', linewidth=2,
                            marker='o', markersize=3)
            axes[2, 0].set_xlabel("Epoch")
            axes[2, 0].set_ylabel("Learning Rate")
            axes[2, 0].set_title("Learning Rate Schedule", fontweight='bold')
            axes[2, 0].set_yscale('log')
            axes[2, 0].grid(True, alpha=0.3)

        # 6. Epoch Time
        if len(history.get("epoch_time", [])) > 0:
            epochs = range(1, len(history["epoch_time"]) + 1)
            axes[2, 1].plot(epochs, history["epoch_time"], color=system_color, linewidth=2,
                            marker='o', markersize=3)
            axes[2, 1].set_xlabel("Epoch")
            axes[2, 1].set_ylabel("Time (seconds)")
            axes[2, 1].set_title("Training Time per Epoch", fontweight='bold')
            axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / "training_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating training summary: {e}")

    # Individual metric plots
    for metric, values in history.items():
        try:
            if len(values) == 0:
                continue
            if metric.startswith('train_'):
                color = train_color
            elif metric.startswith('test_'):
                color = test_color
            else:
                color = system_color
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, len(values) + 1), values, color=color, linewidth=2,
                     marker='o', markersize=4)
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
            plt.title(f"{metric.replace('_', ' ').title()} vs Epochs", fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_dir / f"{metric}_curve.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error creating {metric} plot: {e}")


def visualize_points_and_bboxes(
    points: ArrayLike,
    pred_boxes: list,
    gt_boxes: list,
    pred_labels: Optional[ArrayLike] = None,
    gt_labels: Optional[ArrayLike] = None,
) -> None:
    """Visualize a point cloud with predicted and GT bounding boxes (Open3D).

    Predicted damage points are colored red, healthy points gray. Ground-truth
    boxes are green, predicted boxes are red.

    Args:
        points: (N, 3) array of coordinates.
        pred_boxes: List of predicted box dicts (from ``cluster_to_aabbs``).
        gt_boxes: List of ground-truth box dicts (from ``cluster_to_aabbs``).
        pred_labels: Optional (N,) label array used to color damaged points.
        gt_labels: Optional (N,) label array (reserved for alternate schemes).

    Raises:
        RuntimeError: If Open3D is not installed in the current environment.
    """
    if not _HAS_OPEN3D:
        raise RuntimeError(
            "Open3D is required for visualize_points_and_bboxes but is not "
            "installed. Install open3d to use interactive 3D visualization."
        )

    geoms = []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if pred_labels is not None:
        pred_labels = np.asarray(pred_labels)
        colors = np.zeros_like(points, dtype=float)
        colors[pred_labels == 1] = [1, 0, 0]
        colors[pred_labels == 0] = [0.6, 0.6, 0.6]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.paint_uniform_color([0.7, 0.7, 0.7])

    geoms.append(pcd)

    for box in gt_boxes:
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=box['min'], max_bound=box['max'])
        bbox.color = (0, 1, 0)
        geoms.append(bbox)

    for box in pred_boxes:
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=box['min'], max_bound=box['max'])
        bbox.color = (1, 0, 0)
        geoms.append(bbox)

    o3d.visualization.draw_geometries(geoms)
