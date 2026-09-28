"""
Per-point evaluation metrics for binary damage segmentation.

Consolidated from ``train_dmgseg9.py`` (the most complete version) and
``eval_point_and_bbox.py``. Provides the full metrics suite used to evaluate
per-point predictions: precision, recall, F1, per-class IoU, mean IoU, Dice
coefficient, and PR-AUC.

Labels are binary: 0 = healthy (intact), 1 = damaged.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)

ArrayLike = np.ndarray


def jaccard_manual(y_true: ArrayLike, y_pred: ArrayLike, cls: int) -> float:
    """Compute the IoU (Jaccard index) for a single class manually.

    Args:
        y_true: 1-D array of ground-truth labels.
        y_pred: 1-D array of predicted labels.
        cls: The class value to compute IoU for.

    Returns:
        Intersection-over-union for `cls`, or 0.0 when the union is empty.
    """
    y_true_c = (np.asarray(y_true) == cls).astype(np.int32)
    y_pred_c = (np.asarray(y_pred) == cls).astype(np.int32)
    inter = np.sum(y_true_c & y_pred_c)
    union = np.sum(y_true_c | y_pred_c)
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def calculate_metrics(
    pred_labels: ArrayLike,
    true_labels: ArrayLike,
    pred_scores: Optional[ArrayLike] = None,
) -> tuple[float, float, float, float, Optional[float], dict]:
    """Compute the full metrics suite for binary damage segmentation.

    This is the complete version consolidated from ``train_dmgseg9.py``. It
    returns both a tuple of headline scalars (for backwards-compatible logging)
    and a detailed results dictionary.

    Args:
        pred_labels: Predicted binary labels, shape (N,) or any shape (flattened).
        true_labels: Ground-truth binary labels, same shape as `pred_labels`.
        pred_scores: Optional per-point damage probabilities/scores, shape (N,).
            When provided, PR-AUC (average precision) is computed.

    Returns:
        A tuple ``(precision, recall, f1, miou, pr_auc, results)`` where
        ``results`` is a dict with keys: ``precision``, ``recall``, ``f1``,
        ``iou_intact``, ``iou_damaged``, ``miou``, ``dice``, ``dice_damaged``,
        ``pr_auc`` (may be ``None``), and ``support``.
    """
    pred_labels = np.asarray(pred_labels).flatten()
    true_labels = np.asarray(true_labels).flatten()

    # Guard for empty arrays
    if pred_labels.size == 0 or true_labels.size == 0:
        empty = {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "iou_intact": 0.0, "iou_damaged": 0.0, "miou": 0.0,
            "dice": 0.0, "dice_damaged": 0.0, "pr_auc": None, "support": {},
        }
        return 0.0, 0.0, 0.0, 0.0, None, empty

    # Basic metrics
    precision = precision_score(true_labels, pred_labels, average='binary', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='binary', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='binary', zero_division=0)

    # IoU (per class + mean IoU) - handle if a class missing
    try:
        iou_per_class = jaccard_score(true_labels, pred_labels, average=None, labels=[0, 1])
        if len(iou_per_class) == 2:
            iou_intact, iou_damaged = iou_per_class
        else:
            iou_intact, iou_damaged = (
                (float(iou_per_class[0]), 0.0)
                if 0 in np.unique(true_labels)
                else (0.0, float(iou_per_class[0]))
            )
    except Exception:
        iou_intact = jaccard_manual(true_labels, pred_labels, 0)
        iou_damaged = jaccard_manual(true_labels, pred_labels, 1)
    miou = float(np.mean([iou_intact, iou_damaged]))

    # Dice (for damaged class)
    denom = np.sum(pred_labels == 1) + np.sum(true_labels == 1)
    dice_damaged = (2.0 * np.sum((pred_labels == 1) & (true_labels == 1))) / (denom + 1e-6)
    # Approximate overall dice from mean IoU (kept for logging parity)
    dice = (2 * miou) / (1 + miou + 1e-6)

    # PR AUC
    pr_auc: Optional[float] = None
    if pred_scores is not None:
        try:
            pr_auc = float(average_precision_score(true_labels, np.asarray(pred_scores).flatten()))
        except Exception:
            pr_auc = 0.0

    # Class support (predictions)
    unique, counts = np.unique(pred_labels, return_counts=True)
    support = dict(zip(map(int, unique), map(int, counts)))

    results = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou_intact": float(iou_intact),
        "iou_damaged": float(iou_damaged),
        "miou": float(miou),
        "dice": float(dice),
        "dice_damaged": float(dice_damaged),
        "pr_auc": float(pr_auc) if pr_auc is not None else None,
        "support": support,
    }
    return float(precision), float(recall), float(f1), miou, pr_auc, results


def per_point_metrics(pred_labels: ArrayLike, true_labels: ArrayLike) -> dict:
    """Compute a compact set of per-point metrics for the damaged class.

    A simpler alternative to :func:`calculate_metrics` intended for use by
    ``scripts/evaluate.py`` when only the headline damaged-class scores are
    needed.

    Args:
        pred_labels: Predicted binary labels (any shape, flattened).
        true_labels: Ground-truth binary labels (any shape, flattened).

    Returns:
        Dict with keys ``precision``, ``recall``, ``f1``, and ``iou`` for the
        damaged (positive) class.
    """
    pred = np.asarray(pred_labels).flatten()
    true = np.asarray(true_labels).flatten()
    return {
        "precision": float(precision_score(true, pred, average='binary', zero_division=0)),
        "recall": float(recall_score(true, pred, average='binary', zero_division=0)),
        "f1": float(f1_score(true, pred, average='binary', zero_division=0)),
        "iou": float(jaccard_score(true, pred, average='binary', zero_division=0)),
    }
