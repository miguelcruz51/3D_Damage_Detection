"""
STL-based inference pipeline for point-cloud damage detection.

Consolidated from ``infer_from_stl.py``. Loads a raw STL mesh, uniformly
samples a point cloud, estimates normals, spatially partitions the cloud into
local patches, runs a model over each patch with stochastic voting, and
stitches per-point damage probabilities back onto the full cloud.

All hardcoded constants (``SAMPLE_POINTS``, ``PARTITION_SIZE``, ``NPOINTS``,
``DAMAGE_THRESHOLD``) and absolute paths (``DAMAGED_STL_PATH``, ``MODEL_PATH``)
from the original script have been replaced with function parameters. The core
inference function returns numpy arrays and never launches an Open3D viewer.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import torch

# Guard Open3D: only load_damaged_stl needs it, and the core inference function
# does not. This keeps the module importable in headless test environments.
try:
    import open3d as o3d  # type: ignore
    _HAS_OPEN3D = True
except Exception:  # pragma: no cover - import guard
    o3d = None  # type: ignore
    _HAS_OPEN3D = False

from data.preprocessing import partition_point_cloud

ArrayLike = np.ndarray


def load_damaged_stl(stl_path: str, n_points: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
    """Load an STL mesh and uniformly sample a point cloud with normals.

    Args:
        stl_path: Path to the STL file.
        n_points: Number of points to sample uniformly from the mesh surface.

    Returns:
        Tuple ``(features, points)`` where ``features`` is an (n_points, 6)
        array of concatenated XYZ + normals and ``points`` is the (n_points, 3)
        XYZ array.

    Raises:
        RuntimeError: If Open3D is unavailable or the mesh has no vertices.
    """
    if not _HAS_OPEN3D:
        raise RuntimeError(
            "Open3D is required to load STL meshes but is not installed."
        )

    mesh = o3d.io.read_triangle_mesh(stl_path)
    if len(mesh.vertices) == 0:
        raise RuntimeError(f"Loaded mesh has no vertices: {stl_path}")
    mesh.compute_vertex_normals()

    pcd = mesh.sample_points_uniformly(n_points)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    features = np.concatenate([points, normals], axis=1)
    return features, points


def aggregate_votes(
    prob_acc: dict[int, list[float]],
    n_points: int,
) -> np.ndarray:
    """Aggregate accumulated per-point probability votes by mean.

    Args:
        prob_acc: Mapping from global point index to a list of predicted
            damage probabilities gathered across sub-samples.
        n_points: Total number of points in the full cloud.

    Returns:
        (n_points,) float array of aggregated (mean) damage probabilities.
        Points that received no votes remain 0.0.
    """
    final_probs = np.zeros(n_points, dtype=float)
    for idx, probs in prob_acc.items():
        final_probs[idx] = float(np.mean(probs))
    return final_probs


def run_inference(
    model: torch.nn.Module,
    features: ArrayLike,
    xyz: ArrayLike,
    partition_size: int = 2048,
    npoints: int = 1024,
    damage_threshold: float = 0.1,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run damage inference over a full point cloud with stochastic voting.

    The cloud is partitioned into local patches. Within each patch, ``npoints``
    points are randomly sub-sampled and passed to the model; the resulting
    per-point damage probabilities are accumulated per global index. Final
    per-point probability is the mean of all votes for that point.

    Args:
        model: A trained segmentation model returning ``(log_probs, features)``
            with ``log_probs`` of shape ``[B, N, 2]``.
        features: (M, 6) array of XYZ + normals for the full cloud.
        xyz: (M, 3) array of XYZ coordinates for the full cloud.
        partition_size: Points per spatial patch.
        npoints: Number of points sub-sampled from each patch for the model.
        damage_threshold: Probability threshold for the binary decision.
        device: Torch device. Defaults to CUDA if available, else CPU.

    Returns:
        Tuple ``(final_probs, final_preds)`` of numpy arrays, each length M:
        aggregated damage probabilities and thresholded binary predictions.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = np.asarray(features)
    xyz = np.asarray(xyz)

    model = model.to(device)
    model.eval()

    patches = partition_point_cloud(xyz, partition_size=partition_size)

    prob_acc: dict[int, list[float]] = defaultdict(list)

    with torch.no_grad():
        for patch_idx in patches:
            patch_feat = features[patch_idx]
            n_available = len(patch_idx)
            sample_n = min(npoints, n_available)
            sel = np.random.choice(n_available, sample_n, replace=False)
            inp = patch_feat[sel]

            batch = torch.tensor(inp, dtype=torch.float32)
            batch = batch.transpose(1, 0).unsqueeze(0).to(device)

            logits, _ = model(batch)
            probs = torch.exp(logits)[0, :, 1].cpu().numpy()

            for li, gi in enumerate(patch_idx[sel]):
                prob_acc[int(gi)].append(float(probs[li]))

    final_probs = aggregate_votes(prob_acc, len(xyz))
    final_preds = (final_probs >= damage_threshold).astype(np.int64)
    return final_probs, final_preds
