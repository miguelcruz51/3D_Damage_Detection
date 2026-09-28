#!/usr/bin/env python
"""
Evaluation entry point for a trained damage-segmentation model.

Loads the best checkpoint from an experiment directory, runs per-point
evaluation over the test split, optionally runs bounding-box-level (macro)
evaluation, and saves precision-recall, ROC, and confusion-matrix plots.

Usage:
    python scripts/evaluate.py --log_dir log/part_seg/my_run
    python scripts/evaluate.py --log_dir log/part_seg/my_run \
        --num_point 1024 --normal --gpu 0 --macro_eval

The experiment directory is expected to contain:
    checkpoints/best_model.pth
    logs/meta.json          (used to recover the model architecture name)
"""

import argparse
import importlib
import json
import os
import sys

import numpy as np
import torch

# Make clean_repo/src importable (top-level packages: data, models, evaluation, ...).
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from data.dataset import DamageDataset
from evaluation.metrics import calculate_metrics
from evaluation.bbox_eval import evaluate_point_and_bboxes
from evaluation.visualization import (
    plot_confusion_matrix,
    plot_point_pr_curve,
    plot_point_roc_curve,
)
from utils.checkpoint import load_checkpoint

NUM_PART = 2
MODELS_WITH_ONEHOT = ["kpconv_dmg_seg", "kpconv_dmg_seg_v2", "pointnet2_dmg_seg_ssg"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained damage-segmentation model.")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Experiment directory containing checkpoints/best_model.pth.")
    parser.add_argument("--num_point", type=int, default=1024, help="Points sampled per patch.")
    parser.add_argument("--normal", action="store_true", help="Use surface normals (6-channel input).")
    parser.add_argument("--gpu", type=str, default=None, help="CUDA device index (e.g. '0').")
    parser.add_argument("--macro_eval", action="store_true",
                        help="Also run bounding-box-level (DBSCAN + AABB IoU) evaluation.")
    parser.add_argument("--data_root", type=str, default="data/damaged_components/",
                        help="Root directory of the NPZ patch dataset.")
    parser.add_argument("--model", type=str, default=None,
                        help="Model module name. If omitted, read from logs/meta.json.")
    return parser.parse_args()


def _resolve_model_name(log_dir: str, cli_model: str | None) -> str:
    """Determine the model architecture name from CLI or the saved meta.json."""
    if cli_model:
        return cli_model
    meta_path = os.path.join(log_dir, "logs", "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        model = meta.get("model") or meta.get("config", {}).get("model")
        if model:
            return model
    raise ValueError(
        "Could not determine the model architecture. Pass --model explicitly "
        f"or ensure {meta_path} exists with a 'model' field."
    )


def _build_model(model_name: str, normal_channel: bool, device: torch.device) -> torch.nn.Module:
    module = importlib.import_module(f"models.{model_name}")
    try:
        model = module.get_model(normal_channel=normal_channel)
    except TypeError:
        model = module.get_model()
    return model.to(device)


def _to_categorical(y: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    y_np = y.cpu().numpy().squeeze()
    if y_np.ndim == 0:
        y_np = np.array([y_np])
    return torch.eye(num_classes)[y_np].to(device)


def main() -> None:
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not os.path.isdir(args.log_dir):
        raise FileNotFoundError(f"Experiment directory not found: {args.log_dir}")

    ckpt_path = os.path.join(args.log_dir, "checkpoints", "best_model.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Best checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = _resolve_model_name(args.log_dir, args.model)
    model = _build_model(model_name, args.normal, device)
    load_checkpoint(ckpt_path, model, map_location="cuda" if device.type == "cuda" else "cpu")
    model.eval()

    use_one_hot = model_name in MODELS_WITH_ONEHOT

    test_dataset = DamageDataset(
        root=args.data_root, npoints=args.num_point, split="test", normal_channel=args.normal
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=8, shuffle=False, num_workers=4
    )

    # Component-category count, used only for models that take a cls_label.
    num_categories = max(1, len(test_dataset.cat2id))

    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_points: list[np.ndarray] = []

    with torch.no_grad():
        for points, label, target in test_loader:
            xyz = points[:, :, 0:3].reshape(-1, 3).cpu().numpy()
            points_t = points.to(device).float().transpose(2, 1)
            label = label.long().to(device)

            if use_one_hot:
                one_hot = _to_categorical(label.squeeze(), num_categories, device)
                seg_pred, _ = model(points_t, one_hot)
            else:
                seg_pred, _ = model(points_t)

            seg_pred = seg_pred.contiguous().view(-1, NUM_PART)
            pred_choice = seg_pred.max(1)[1]
            pred_scores = torch.softmax(seg_pred, dim=1)[:, 1]

            all_preds.append(pred_choice.cpu().numpy())
            all_targets.append(target.view(-1).cpu().numpy())
            all_scores.append(pred_scores.cpu().numpy())
            all_points.append(xyz)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_scores = np.concatenate(all_scores)
    all_points = np.concatenate(all_points, axis=0)

    precision, recall, f1, miou, pr_auc, results = calculate_metrics(
        all_preds, all_targets, all_scores
    )

    print("=== Per-point evaluation ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"mIoU:      {miou:.4f}")
    print(f"PR-AUC:    {pr_auc if pr_auc is not None else 0.0:.4f}")
    print(f"IoU (damaged): {results['iou_damaged']:.4f} | IoU (intact): {results['iou_intact']:.4f}")

    # --- save plots -------------------------------------------------------
    plots_dir = os.path.join(args.log_dir, "eval_plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_point_pr_curve(all_targets, all_scores, os.path.join(plots_dir, "pr_curve.png"))
    plot_point_roc_curve(all_targets, all_scores, os.path.join(plots_dir, "roc_curve.png"))
    plot_confusion_matrix(
        all_targets, all_preds,
        class_names=["healthy", "damaged"],
        save_path=os.path.join(plots_dir, "confusion_matrix.png"),
    )
    print(f"Plots saved to: {plots_dir}")

    # --- optional macro (bbox-level) evaluation ---------------------------
    if args.macro_eval:
        macro = evaluate_point_and_bboxes(all_points, all_preds, all_targets)
        box_metrics = macro["box_metrics"]
        print("=== Bounding-box-level evaluation ===")
        print(f"TP: {box_metrics['TP']} | FP: {box_metrics['FP']} | FN: {box_metrics['FN']}")
        print(f"Box Precision: {box_metrics['precision']:.4f}")
        print(f"Box Recall:    {box_metrics['recall']:.4f}")
        print(f"Box F1:        {box_metrics['f1']:.4f}")
        print(f"Avg matched IoU: {box_metrics['avg_iou']:.4f}")
        with open(os.path.join(plots_dir, "macro_eval.json"), "w") as f:
            json.dump(box_metrics, f, indent=2)


if __name__ == "__main__":
    main()
