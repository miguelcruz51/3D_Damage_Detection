#!/usr/bin/env python
"""
Inference entry point: predict surface damage on a raw STL component.

Loads a mesh, uniformly samples a point cloud, estimates normals, spatially
partitions the cloud, runs a trained model with stochastic voting, and produces
per-point damage probabilities and binary predictions.

Usage:
    # Save results to disk (.npz probabilities + .txt binary predictions):
    python scripts/inference.py --stl_path part.stl --model_path best_model.pth \
        --output_path results/part --npoints 1024 --threshold 0.1 --n_votes 3

    # Launch the interactive Open3D viewer (no --output_path):
    python scripts/inference.py --stl_path part.stl --model_path best_model.pth

No CLI argument may be a hardcoded absolute path to a non-existent location.
"""

import argparse
import importlib
import os
import sys

import numpy as np
import torch

# Make clean_repo/src importable (top-level packages: data, models, evaluation, ...).
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from evaluation.inference import load_damaged_stl, run_inference

MODELS_WITH_ONEHOT = ["kpconv_dmg_seg", "kpconv_dmg_seg_v2", "pointnet2_dmg_seg_ssg"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run damage inference on a raw STL file.")
    parser.add_argument("--stl_path", type=str, required=True, help="Path to the input STL mesh.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .pth checkpoint.")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output path prefix. If given, saves <prefix>.npz and <prefix>.txt; "
                             "otherwise launches the Open3D viewer.")
    parser.add_argument("--model", type=str, default="pointnet2_dmg_seg_ssg",
                        help="Model architecture module name under src/models/.")
    parser.add_argument("--normal", action="store_true", default=True,
                        help="Feed surface normals to the model (6-channel input).")
    parser.add_argument("--sample_points", type=int, default=200_000,
                        help="Number of points uniformly sampled from the mesh surface.")
    parser.add_argument("--partition_size", type=int, default=2048,
                        help="Number of points per spatial patch.")
    parser.add_argument("--npoints", type=int, default=1024,
                        help="Points sub-sampled from each patch for the model.")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Damage probability threshold for the binary decision.")
    parser.add_argument("--n_votes", type=int, default=1,
                        help="Number of stochastic voting passes to average.")
    return parser.parse_args()


def _validate_input_path(arg_name: str, path: str) -> None:
    """Reject absolute paths that point to a non-existent location.

    This guards against hardcoded absolute paths (e.g. from a developer's
    machine) that would silently fail on another system.
    """
    if os.path.isabs(path) and not os.path.exists(path):
        raise FileNotFoundError(
            f"--{arg_name} is an absolute path that does not exist on this "
            f"system: '{path}'. Provide a path that exists locally."
        )


def _validate_output_path(path: str) -> None:
    """Ensure an absolute output prefix lives in an existing directory."""
    parent = os.path.dirname(os.path.abspath(path))
    if os.path.isabs(path) and not os.path.isdir(parent):
        raise FileNotFoundError(
            f"--output_path directory does not exist: '{parent}'. "
            "Create it first or provide a relative path."
        )


def _build_model(model_name: str, normal: bool, device: torch.device) -> torch.nn.Module:
    module = importlib.import_module(f"models.{model_name}")
    try:
        model = module.get_model(normal_channel=normal)
    except TypeError:
        model = module.get_model()
    return model.to(device)


def main() -> None:
    args = parse_args()

    # --- path validation --------------------------------------------------
    _validate_input_path("stl_path", args.stl_path)
    _validate_input_path("model_path", args.model_path)
    if args.output_path is not None:
        _validate_output_path(args.output_path)

    if not os.path.isfile(args.stl_path):
        raise FileNotFoundError(f"STL file not found: {args.stl_path}")
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load model + weights --------------------------------------------
    model = _build_model(args.model, args.normal, device)
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # --- load + sample the mesh ------------------------------------------
    features, xyz = load_damaged_stl(args.stl_path, n_points=args.sample_points)

    # --- run inference with stochastic voting ----------------------------
    prob_votes = []
    for _ in range(max(1, args.n_votes)):
        probs, _ = run_inference(
            model, features, xyz,
            partition_size=args.partition_size,
            npoints=args.npoints,
            damage_threshold=args.threshold,
            device=device,
        )
        prob_votes.append(probs)
    final_probs = np.mean(np.stack(prob_votes, axis=0), axis=0)
    final_preds = (final_probs >= args.threshold).astype(np.int64)

    print(f"Points: {len(xyz)} | Predicted damaged: {int(final_preds.sum())} "
          f"({100.0 * final_preds.mean():.2f}%)")

    # --- output ----------------------------------------------------------
    if args.output_path is not None:
        npz_path = f"{args.output_path}.npz"
        txt_path = f"{args.output_path}.txt"
        np.savez(npz_path, points=xyz, probabilities=final_probs)
        np.savetxt(txt_path, final_preds, fmt="%d")
        print(f"Saved probabilities to: {npz_path}")
        print(f"Saved binary predictions to: {txt_path}")
    else:
        from evaluation.visualization import visualize_points_and_bboxes
        visualize_points_and_bboxes(xyz, pred_boxes=[], gt_boxes=[], pred_labels=final_preds)


if __name__ == "__main__":
    main()
