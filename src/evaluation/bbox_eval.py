"""
Bounding-box-level (macro) evaluation for damage detection.

Consolidated from ``eval_point_and_bbox.py``. Provides the "macro-level"
evaluation that goes beyond per-point metrics: it clusters damaged points with
DBSCAN, fits axis-aligned bounding boxes (AABBs) to each cluster, matches
predicted boxes to ground-truth boxes by IoU, and reports box-level
precision/recall/F1.

Open3D is only required by the interactive visualization helpers, which live in
``visualization.py``; this module therefore has no hard dependency on Open3D.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from sklearn.neighbors import NearestNeighbors

ArrayLike = np.ndarray
Box = dict


def per_point_metrics(pred_labels: ArrayLike, true_labels: ArrayLike) -> dict:
    """Compute per-point precision/recall/F1/IoU for the damaged class.

    Args:
        pred_labels: Predicted binary labels (any shape, flattened).
        true_labels: Ground-truth binary labels (any shape, flattened).

    Returns:
        Dict with keys ``precision``, ``recall``, ``f1``, ``iou``.
    """
    pred = np.asarray(pred_labels).flatten()
    true = np.asarray(true_labels).flatten()
    precision = precision_score(true, pred, average='binary', zero_division=0)
    recall = recall_score(true, pred, average='binary', zero_division=0)
    f1 = f1_score(true, pred, average='binary', zero_division=0)
    iou = jaccard_score(true, pred, average='binary', zero_division=0)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou}


def adaptive_eps_from_points(points: ArrayLike, k: int = 5, multiplier: float = 1.0) -> float:
    """Estimate a reasonable DBSCAN ``eps`` from k-NN distances.

    Args:
        points: (M, 3) array of coordinates.
        k: Use the distance to the k-th nearest neighbor.
        multiplier: Safety factor applied to the median k-NN distance.

    Returns:
        Estimated ``eps`` value (float).
    """
    if len(points) <= k:
        bbox = np.max(points, axis=0) - np.min(points, axis=0)
        scale = np.linalg.norm(bbox) / np.sqrt(len(points) + 1e-12)
        return max(1e-3, scale * multiplier)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
    distances, _ = nbrs.kneighbors(points)
    kth = distances[:, k]
    eps = np.median(kth) * multiplier
    if eps <= 0:
        bbox = np.max(points, axis=0) - np.min(points, axis=0)
        eps = np.linalg.norm(bbox) * 1e-3
    return float(eps)


def cluster_damage_points(
    points: ArrayLike,
    labels: ArrayLike,
    eps: Optional[float] = None,
    min_samples: int = 10,
    adaptive_k: int = 5,
    adaptive_multiplier: float = 1.0,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Cluster the damaged points (``labels == 1``) using DBSCAN.

    Args:
        points: (N, 3) array of coordinates.
        labels: (N,) binary label array (1 = damaged).
        eps: DBSCAN neighborhood radius. If None, estimated adaptively.
        min_samples: Minimum cluster size for DBSCAN.
        adaptive_k: k for adaptive eps estimation.
        adaptive_multiplier: Multiplier for adaptive eps estimation.

    Returns:
        Tuple ``(clusters, cluster_labels, indices_damage)`` where ``clusters``
        is a list of arrays of *global* point indices (one per cluster),
        ``cluster_labels`` is the raw DBSCAN label array over the damaged
        points, and ``indices_damage`` are the global indices of damaged points.
    """
    labels = np.asarray(labels).flatten()
    indices_damage = np.where(labels == 1)[0]
    if len(indices_damage) == 0:
        return [], np.array([]), indices_damage

    pts = points[indices_damage, :]

    if eps is None:
        eps = adaptive_eps_from_points(pts, k=adaptive_k, multiplier=adaptive_multiplier)

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    clabels = db.labels_  # -1 are noise
    clusters: list[np.ndarray] = []
    for cl in sorted(set(clabels)):
        if cl == -1:
            continue
        idx_in_damage = np.where(clabels == cl)[0]
        clusters.append(indices_damage[idx_in_damage])
    return clusters, clabels, indices_damage


def cluster_to_aabbs(points: ArrayLike, clusters: list[np.ndarray]) -> list[Box]:
    """Compute axis-aligned bounding boxes for a list of clusters.

    Args:
        points: (N, 3) array of coordinates.
        clusters: List of arrays of point indices (one per cluster).

    Returns:
        List of box dicts, each with keys ``min``, ``max``, ``centroid``,
        ``volume``, and ``indices``.
    """
    boxes: list[Box] = []
    for inds in clusters:
        pts = points[inds, :]
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        centroid = pts.mean(axis=0)
        vol = np.prod(np.maximum(mx - mn, 0.0))
        boxes.append({'min': mn, 'max': mx, 'centroid': centroid, 'volume': vol, 'indices': inds})
    return boxes


def aabb_iou(boxA: Box, boxB: Box) -> float:
    """Compute the IoU of two axis-aligned 3D bounding boxes.

    Args:
        boxA: Box dict with ``min`` and ``max`` (length-3 arrays).
        boxB: Box dict with ``min`` and ``max`` (length-3 arrays).

    Returns:
        Intersection-over-union in [0, 1].
    """
    mnA, mxA = np.array(boxA['min']), np.array(boxA['max'])
    mnB, mxB = np.array(boxB['min']), np.array(boxB['max'])
    inter_min = np.maximum(mnA, mnB)
    inter_max = np.minimum(mxA, mxB)
    inter_wh = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = np.prod(inter_wh)
    volA = np.prod(np.maximum(mxA - mnA, 0.0))
    volB = np.prod(np.maximum(mxB - mnB, 0.0))
    union = volA + volB - inter_vol
    if union <= 0:
        return 0.0
    return float(inter_vol / union)


def match_boxes_by_iou(
    gt_boxes: list[Box],
    pred_boxes: list[Box],
    iou_threshold: float = 0.1,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedily match predicted boxes to ground-truth boxes by IoU.

    Args:
        gt_boxes: List of ground-truth box dicts.
        pred_boxes: List of predicted box dicts.
        iou_threshold: Minimum IoU for a pair to count as a match.

    Returns:
        Tuple ``(matched, unmatched_gt, unmatched_pred)`` where ``matched`` is a
        list of ``(gt_idx, pred_idx, iou)`` tuples, and the other two are lists
        of unmatched indices.
    """
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return [], [], []
    if len(gt_boxes) == 0:
        return [], [], list(range(len(pred_boxes)))
    if len(pred_boxes) == 0:
        return [], list(range(len(gt_boxes))), []

    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)), dtype=float)
    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            iou_matrix[i, j] = aabb_iou(g, p)

    matched: list[tuple[int, int, float]] = []
    # Greedy: pick max IoU until none > 0
    while True:
        i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        max_iou = iou_matrix[i, j]
        if max_iou <= 0:
            break
        matched.append((int(i), int(j), float(max_iou)))
        iou_matrix[i, :] = -1
        iou_matrix[:, j] = -1

    # Filter by threshold
    matched = [m for m in matched if m[2] >= iou_threshold]
    matched_gt = [m[0] for m in matched]
    matched_pred = [m[1] for m in matched]

    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    unmatched_pred = [j for j in range(len(pred_boxes)) if j not in matched_pred]

    return matched, unmatched_gt, unmatched_pred


def box_level_metrics(
    gt_boxes: list[Box],
    pred_boxes: list[Box],
    matched: list[tuple[int, int, float]],
    unmatched_gt: list[int],
    unmatched_pred: list[int],
) -> dict:
    """Compute box-level precision/recall/F1 and average matched IoU.

    Args:
        gt_boxes: List of ground-truth boxes (used for interface parity).
        pred_boxes: List of predicted boxes (used for interface parity).
        matched: List of ``(gt_idx, pred_idx, iou)`` matched tuples.
        unmatched_gt: Indices of unmatched ground-truth boxes (false negatives).
        unmatched_pred: Indices of unmatched predicted boxes (false positives).

    Returns:
        Dict with keys ``TP``, ``FP``, ``FN``, ``precision``, ``recall``,
        ``f1``, ``avg_iou``.
    """
    TP = len(matched)
    FP = len(unmatched_pred)
    FN = len(unmatched_gt)
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    avg_iou = float(np.mean([m[2] for m in matched])) if len(matched) > 0 else 0.0
    return {'TP': TP, 'FP': FP, 'FN': FN, 'precision': prec, 'recall': rec, 'f1': f1, 'avg_iou': avg_iou}


def evaluate_point_and_bboxes(
    points: ArrayLike,
    pred_labels: ArrayLike,
    gt_labels: ArrayLike,
    dbscan_params_pred: Optional[dict] = None,
    dbscan_params_gt: Optional[dict] = None,
    adaptive_k: int = 100,
    adaptive_multiplier: float = 1.0,
    min_samples: int = 5,
    iou_threshold: float = 0.1,
) -> dict:
    """Run per-point and bounding-box-level evaluation end to end.

    Args:
        points: (N, 3) array of coordinates.
        pred_labels: (N,) array of predicted 0/1 labels.
        gt_labels: (N,) array of ground-truth 0/1 labels.
        dbscan_params_pred: Optional dict with ``eps`` and ``min_samples`` for
            clustering predicted damage points.
        dbscan_params_gt: Optional dict with ``eps`` and ``min_samples`` for
            clustering ground-truth damage points.
        adaptive_k: k for adaptive eps estimation.
        adaptive_multiplier: Multiplier for adaptive eps estimation.
        min_samples: Default minimum cluster size.
        iou_threshold: Minimum IoU to count a box match as a true positive.

    Returns:
        Dict with per-point metrics, clusters, boxes, matches, and box metrics.
    """
    points = np.asarray(points)
    pred = np.asarray(pred_labels).flatten()
    gt = np.asarray(gt_labels).flatten()

    # 1) per-point metrics
    pt_metrics = per_point_metrics(pred, gt)

    # 2) cluster GT and predicted damaged points
    db_pred = dbscan_params_pred or {}
    db_gt = dbscan_params_gt or {}

    pred_eps = db_pred.get('eps', None)
    pred_min = db_pred.get('min_samples', min_samples)

    gt_eps = db_gt.get('eps', None)
    gt_min = db_gt.get('min_samples', min_samples)

    pred_clusters, _, _ = cluster_damage_points(
        points, pred, eps=pred_eps, min_samples=pred_min,
        adaptive_k=adaptive_k, adaptive_multiplier=adaptive_multiplier,
    )
    gt_clusters, _, _ = cluster_damage_points(
        points, gt, eps=gt_eps, min_samples=gt_min,
        adaptive_k=adaptive_k, adaptive_multiplier=adaptive_multiplier,
    )

    # 3) convert to boxes
    pred_boxes = cluster_to_aabbs(points, pred_clusters)
    gt_boxes = cluster_to_aabbs(points, gt_clusters)

    # 4) match boxes by IoU
    matched, unmatched_gt, unmatched_pred = match_boxes_by_iou(
        gt_boxes, pred_boxes, iou_threshold=iou_threshold
    )
    box_metrics = box_level_metrics(gt_boxes, pred_boxes, matched, unmatched_gt, unmatched_pred)

    return {
        'per_point': pt_metrics,
        'pred_clusters': pred_clusters,
        'gt_clusters': gt_clusters,
        'pred_boxes': pred_boxes,
        'gt_boxes': gt_boxes,
        'matched': matched,
        'unmatched_gt': unmatched_gt,
        'unmatched_pred': unmatched_pred,
        'box_metrics': box_metrics,
        'params': {
            'pred_eps': pred_eps, 'gt_eps': gt_eps,
            'adaptive_k': adaptive_k, 'adaptive_multiplier': adaptive_multiplier,
            'min_samples': min_samples, 'iou_threshold': iou_threshold,
        },
    }


def precision_recall_curve_by_iou(
    gt_boxes: list[Box],
    pred_boxes: list[Box],
    thresholds: Optional[ArrayLike] = None,
) -> dict:
    """Compute box-level precision/recall/F1 across a range of IoU thresholds.

    Args:
        gt_boxes: List of ground-truth boxes.
        pred_boxes: List of predicted boxes.
        thresholds: IoU thresholds to sweep. Defaults to 0.1..1.0 step 0.1.

    Returns:
        Dict with keys ``thresholds``, ``precision``, ``recall``, ``f1``.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.1, 1.01, 0.1), 2)

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for thr in thresholds:
        matched, unmatched_gt, unmatched_pred = match_boxes_by_iou(
            gt_boxes, pred_boxes, iou_threshold=float(thr)
        )
        metrics = box_level_metrics(gt_boxes, pred_boxes, matched, unmatched_gt, unmatched_pred)
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        f1s.append(metrics['f1'])

    return {'thresholds': thresholds, 'precision': precisions, 'recall': recalls, 'f1': f1s}
