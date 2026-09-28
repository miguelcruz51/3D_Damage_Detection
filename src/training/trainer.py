"""
Training loop for binary point-cloud damage segmentation.

Extracted and cleaned from ``train_dmgseg9.py`` — the most complete and
reproducible training script in the original codebase. All debug ``print``
statements (e.g. ``print("points.shape:", ...)``) have been removed.

Retained capabilities
----------------------
- Reproducible seeding (Python / NumPy / Torch / CUDA + deterministic cuDNN)
- Class-balancing methods: ``none``, ``fixed_weights``, ``auto_weights``,
  ``weighted_sampler``
- Configurable loss selection (crossentropy / focal / ce_dice / tversky /
  lovasz / boundary_aware / lovasz_boundary)
- Step and cosine learning-rate schedulers
- Transfer learning: partial pretrained-weight loading + encoder freezing
- TensorBoard logging
- Augmentation levels: ``none`` / ``basic`` / ``advanced``
- Early stopping with configurable metric (accuracy / recall / pr_auc)
- ``meta.json`` saving for reproducibility

Configuration is passed as a plain object (dict or ``argparse.Namespace``)
rather than being parsed from ``sys.argv`` inside the trainer.
"""

from __future__ import annotations

import datetime
import importlib
import json
import logging
import os
import random
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import augmentation
from data.dataset import DamageDataset
from evaluation.metrics import calculate_metrics
from evaluation.visualization import save_history_plots
from models.losses import (
    BoundaryAwareLoss,
    CrossEntropyDiceLoss,
    FocalLoss,
    LovaszBoundaryLoss,
    LovaszSoftmax,
    TverskyLoss,
)
from utils.checkpoint import load_pretrained_weights, save_checkpoint

# Number of segmentation classes (0 = healthy, 1 = damaged).
NUM_PART = 2
# Models whose forward pass expects a one-hot class label as a second argument.
MODELS_WITH_ONEHOT = ["kpconv_dmg_seg", "kpconv_dmg_seg_v2", "pointnet2_dmg_seg_ssg"]

# Default configuration mirroring the argparse defaults of train_dmgseg9.py.
DEFAULT_CONFIG = {
    "model": "pointnet2_dmg_seg_ssg",
    "batch_size": 16,
    "epoch": 251,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "log_dir": None,
    "decay_rate": 1e-4,
    "npoint": 2048,
    "normal": False,
    "step_size": 20,
    "lr_decay": 0.5,
    "start_epoch": 0,
    "seed": 42,
    "pretrained_path": "",
    "freeze_encoder": False,
    "freeze_ratio": 1.0,
    "balancing_method": "none",
    "fixed_class_weights": [1.0, 55.0],
    "oversample_factor": 3.0,
    "loss_function": "crossentropy",
    "focal_alpha": [0.25, 0.75],
    "focal_gamma": 1.5,
    "dice_lambda": 0.3,
    "tversky_alpha": 0.3,
    "tversky_beta": 0.7,
    "early_stop_metric": "accuracy",
    "early_stop_patience": 10,
    "scheduler": "step",
    "augment": "none",
    "lovasz_weight": 0.5,
    "boundary_weight": 0.5,
    "boundary_emphasis": 2.0,
    "data_root": "data/damaged_components/",
    "resume": "",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_namespace(config: Union[dict, SimpleNamespace]) -> SimpleNamespace:
    """Normalise a dict or namespace into a namespace with defaults filled in."""
    if isinstance(config, dict):
        merged = {**DEFAULT_CONFIG, **config}
    else:
        merged = {**DEFAULT_CONFIG, **vars(config)}
    return SimpleNamespace(**merged)


def _set_seed(seed: int) -> None:
    """Seed all RNGs and enable deterministic cuDNN for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _inplace_relu(m: torch.nn.Module) -> None:
    if isinstance(m, torch.nn.ReLU):
        m.inplace = True


def _to_categorical(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """One-hot encode class labels, preserving device."""
    y_np = y.cpu().data.numpy().squeeze()
    if y_np.ndim == 0:
        y_np = np.array([y_np])
    new_y = torch.eye(num_classes)[y_np]
    return new_y.cuda() if y.is_cuda else new_y


def _freeze_layers(model: torch.nn.Module, freeze_ratio: float = 1.0, logger=None) -> torch.nn.Module:
    """Freeze encoder/backbone layers for transfer learning.

    A parameter is eligible for freezing if its name contains a known encoder
    token (``mlp``, ``conv1``, ``sa1``, ``encoder``). Eligible parameters are
    frozen with probability ``freeze_ratio``.
    """
    frozen, total = 0, 0
    for name, param in model.named_parameters():
        total += 1
        if any(tok in name for tok in ("mlp", "conv1", "sa1", "encoder")):
            if random.random() <= freeze_ratio:
                param.requires_grad = False
                frozen += 1
    if logger:
        logger.info(f"[INFO] Frozen {frozen}/{total} parameters ({(frozen / total) * 100:.1f}%).")
    return model


def _compute_class_weights(dataset: DamageDataset) -> torch.Tensor:
    """Compute inverse-frequency class weights from a dataset's labels."""
    all_labels = []
    for i in range(len(dataset)):
        _, _, seg = dataset[i]
        all_labels.append(seg)
    all_labels = np.concatenate(all_labels, axis=0)
    target = torch.tensor(all_labels, dtype=torch.long)
    _, counts = torch.unique(target, return_counts=True)
    weights = 1.0 / counts.float()
    weights = weights / weights.sum()
    return weights


def _build_loss(args: SimpleNamespace, model_module, class_weights: Optional[torch.Tensor], device: torch.device):
    """Construct the loss function selected in the config."""
    if args.loss_function == "focal":
        alpha = torch.tensor(args.focal_alpha).to(device) if args.focal_alpha else None
        return FocalLoss(alpha=alpha, gamma=args.focal_gamma).to(device)
    if args.loss_function == "ce_dice":
        return CrossEntropyDiceLoss(
            ce_weight=1 - args.dice_lambda,
            dice_weight=args.dice_lambda,
            class_weights=class_weights,
        ).to(device)
    if args.loss_function == "tversky":
        return TverskyLoss(alpha=args.tversky_alpha, beta=args.tversky_beta).to(device)
    if args.loss_function == "lovasz":
        return LovaszSoftmax().to(device)
    if args.loss_function == "boundary_aware":
        return BoundaryAwareLoss(boundary_weight=args.boundary_emphasis).to(device)
    if args.loss_function == "lovasz_boundary":
        return LovaszBoundaryLoss(
            lovasz_weight=args.lovasz_weight,
            boundary_weight=args.boundary_weight,
            boundary_emphasis=args.boundary_emphasis,
        ).to(device)
    # Default: model-provided weighted NLL loss.
    return model_module.get_loss(class_weights=class_weights).to(device)


def _new_history() -> dict:
    return {
        "test_accuracy": [], "test_precision": [], "test_recall": [], "test_f1": [],
        "test_iou_damaged": [], "test_iou_intact": [], "test_miou": [],
        "test_dice": [], "test_dice_damaged": [], "test_pr_auc": [], "test_loss": [],
        "train_accuracy": [], "train_loss": [],
        "learning_rate": [], "epoch_time": [],
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def train(config: Union[dict, SimpleNamespace]) -> Path:
    """Run the full training loop.

    Args:
        config: A dict or namespace of hyperparameters. Missing keys fall back
            to :data:`DEFAULT_CONFIG`. Recognised keys mirror the CLI arguments
            of the original ``train_dmgseg9.py`` (see :data:`DEFAULT_CONFIG`).

    Returns:
        Path to the experiment directory containing checkpoints, logs, and
        TensorBoard events.
    """
    args = _as_namespace(config)
    _set_seed(args.seed)
    torch.cuda.empty_cache()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # --- experiment directories -------------------------------------------
    timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    exp_dir = Path("./log/part_seg")
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = exp_dir / (args.log_dir or timestr)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir / "checkpoints"
    log_dir = exp_dir / "logs"
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    # --- logging ----------------------------------------------------------
    logger = logging.getLogger("Trainer")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_dir / f"{args.model}.txt")
    file_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(file_handler)

    def log_string(s):
        logger.info(s)
        print(s)

    tensorboard_dir = exp_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    log_string(f"TensorBoard logging to: {tensorboard_dir}")
    log_string("PARAMETERS:")
    log_string(vars(args))

    # --- meta.json for reproducibility ------------------------------------
    try:
        with open(log_dir / "meta.json", "w") as f:
            meta = {"seed": args.seed, "model": args.model, "config": vars(args)}
            try:
                meta["git_commit"] = (
                    subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                    .decode()
                    .strip()
                )
            except Exception:
                meta["git_commit"] = None
            json.dump(meta, f, indent=2, default=str)
    except Exception:
        pass

    # --- datasets ---------------------------------------------------------
    def worker_init_fn(worker_id):
        worker_seed = args.seed + worker_id + int(torch.initial_seed() % 1_000_000)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_dataset = DamageDataset(root=args.data_root, npoints=args.npoint, split="train", normal_channel=args.normal)
    test_dataset = DamageDataset(root=args.data_root, npoints=args.npoint, split="test", normal_channel=args.normal)

    # --- class balancing --------------------------------------------------
    class_weights: Optional[torch.Tensor] = None
    train_sampler: Optional[WeightedRandomSampler] = None
    if args.balancing_method == "fixed_weights":
        class_weights = torch.tensor(args.fixed_class_weights, dtype=torch.float32).to(device)
    elif args.balancing_method == "auto_weights":
        class_weights = _compute_class_weights(train_dataset).to(device)
    elif args.balancing_method == "weighted_sampler":
        targets = []
        for rel_path in train_dataset.datapath:
            fpath = os.path.join(train_dataset.root, rel_path)
            labels = np.load(fpath)["labels"]
            targets.append(int(np.bincount(labels).argmax()))
        class_sample_counts = np.bincount(targets)
        class_weights_np = 1.0 / class_sample_counts
        if len(class_weights_np) > 1:
            class_weights_np[1] *= args.oversample_factor
        sample_weights = torch.DoubleTensor([class_weights_np[t] for t in targets])
        train_sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=worker_init_fn,
    )

    # Number of component categories (e.g. Component1-Component16). Used only to
    # one-hot encode the category label for models whose forward() accepts a
    # cls_label argument. Derived from the dataset so it cannot fall out of sync
    # with synsetoffset2category.txt.
    num_classes = max(1, len(train_dataset.cat2id))

    # Whether the selected architecture's forward() takes a cls_label argument.
    use_one_hot = args.model in MODELS_WITH_ONEHOT

    # --- model ------------------------------------------------------------
    # Import the selected architecture module from the top-level ``models``
    # package (``clean_repo/src`` is expected to be on ``sys.path``).
    model_module = importlib.import_module(f"models.{args.model}")
    try:
        classifier = model_module.get_model(normal_channel=args.normal)
    except TypeError:
        classifier = model_module.get_model()
    classifier = classifier.to(device)

    if args.pretrained_path:
        log_string(f"[INFO] Loading pretrained model from {args.pretrained_path}")
        classifier = load_pretrained_weights(classifier, args.pretrained_path, logger)

    if args.freeze_encoder:
        classifier = _freeze_layers(classifier, args.freeze_ratio, logger)

    classifier.apply(_inplace_relu)

    # --- loss / optimizer / scheduler -------------------------------------
    criterion = _build_loss(args, model_module, class_weights, device)

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=args.decay_rate,
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-5)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)

    start_epoch = args.start_epoch
    best_metric = 0.0
    early_stop_counter = 0
    history = _new_history()

    # --- optional resume --------------------------------------------------
    if args.resume:
        from utils.checkpoint import load_checkpoint
        ckpt = load_checkpoint(args.resume, classifier, optimizer,
                               map_location="cuda" if use_cuda else "cpu")
        start_epoch = ckpt.get("epoch", 0) + 1
        log_string(f"[INFO] Resumed from {args.resume} at epoch {start_epoch}")

    # --- training epochs --------------------------------------------------
    epoch = start_epoch
    for epoch in range(start_epoch, args.epoch):
        epoch_start_time = time.time()
        log_string(f"\nEpoch {epoch + 1}/{args.epoch}")
        mean_correct = []
        epoch_loss = 0.0

        classifier.train()
        for points, label, target in tqdm(train_loader, total=len(train_loader)):
            points = points.numpy()

            if args.augment == "basic":
                points[:, :, 0:3] = augmentation.random_scale_point_cloud(points[:, :, 0:3])
                points[:, :, 0:3] = augmentation.shift_point_cloud(points[:, :, 0:3])
            elif args.augment == "advanced":
                points[:, :, 0:3] = augmentation.random_scale_point_cloud(points[:, :, 0:3])
                points[:, :, 0:3] = augmentation.shift_point_cloud(points[:, :, 0:3])
                points[:, :, 0:3] = augmentation.jitter_point_cloud(points[:, :, 0:3])

            points = torch.Tensor(points).float().to(device).transpose(2, 1)
            label, target = label.long().to(device), target.long().to(device)

            if use_one_hot:
                one_hot = _to_categorical(label.squeeze(), num_classes)
                seg_pred, trans_feat = classifier(points, one_hot)
            else:
                seg_pred, trans_feat = classifier(points)

            seg_pred = seg_pred.contiguous().view(-1, NUM_PART)
            target = target.view(-1)
            loss = criterion(seg_pred, target)
            epoch_loss += loss.item()

            pred_choice = seg_pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        train_instance_acc = float(np.mean(mean_correct))
        mean_epoch_loss = epoch_loss / len(train_loader)
        log_string(f"Train Accuracy: {train_instance_acc:.4f} | Loss: {mean_epoch_loss:.4f}")

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Training/Loss", mean_epoch_loss, epoch)
        writer.add_scalar("Training/Accuracy", train_instance_acc, epoch)
        writer.add_scalar("Training/Learning_Rate", current_lr, epoch)

        # --- evaluation ---------------------------------------------------
        classifier.eval()
        with torch.no_grad():
            all_preds, all_targets, all_scores = [], [], []
            total_correct, total_seen = 0, 0
            test_epoch_loss = 0.0
            for points, label, target in tqdm(test_loader, total=len(test_loader)):
                points = points.to(device).float().transpose(2, 1)
                label, target = label.long().to(device), target.long().to(device)

                if use_one_hot:
                    one_hot = _to_categorical(label.squeeze(), num_classes)
                    seg_pred, _ = classifier(points, one_hot)
                else:
                    seg_pred, _ = classifier(points)

                test_loss = criterion(seg_pred.contiguous().view(-1, NUM_PART), target.view(-1))
                test_epoch_loss += test_loss.item()

                pred_choice = seg_pred.contiguous().view(-1, NUM_PART).max(1)[1]
                pred_scores = torch.softmax(seg_pred.reshape(-1, NUM_PART), dim=1)[:, 1]

                correct = pred_choice.eq(target.view(-1)).sum()
                total_correct += correct.item()
                total_seen += args.batch_size * args.npoint

                all_preds.append(pred_choice.cpu().numpy())
                all_targets.append(target.view(-1).cpu().numpy())
                all_scores.append(pred_scores.cpu().numpy())

            mean_test_loss = test_epoch_loss / len(test_loader)
            instance_acc = float(total_correct) / float(total_seen)
            all_preds = np.concatenate(all_preds) if all_preds else np.array([])
            all_targets = np.concatenate(all_targets) if all_targets else np.array([])
            all_scores = np.concatenate(all_scores) if all_scores else np.array([])

            precision, recall, f1, miou, pr_auc, extra = calculate_metrics(
                all_preds, all_targets, all_scores
            )

            log_string(
                f"Test Acc: {instance_acc:.4f} | Precision: {precision:.4f} | "
                f"Recall: {recall:.4f} | F1: {f1:.4f} | mIoU: {miou:.4f} | "
                f"PR AUC: {pr_auc if pr_auc is not None else 0.0:.4f}"
            )

            epoch_time = time.time() - epoch_start_time

            writer.add_scalar("Test/Accuracy", instance_acc, epoch)
            writer.add_scalar("Test/Precision", precision, epoch)
            writer.add_scalar("Test/Recall", recall, epoch)
            writer.add_scalar("Test/F1_Score", f1, epoch)
            writer.add_scalar("Test/PR_AUC", pr_auc if pr_auc is not None else 0.0, epoch)
            writer.add_scalar("Test/mIoU", miou, epoch)
            writer.add_scalars("Loss_Comparison", {"Train": mean_epoch_loss, "Test": mean_test_loss}, epoch)

            history["test_accuracy"].append(instance_acc)
            history["test_precision"].append(precision)
            history["test_recall"].append(recall)
            history["test_f1"].append(f1)
            history["test_iou_damaged"].append(extra["iou_damaged"])
            history["test_iou_intact"].append(extra["iou_intact"])
            history["test_miou"].append(miou)
            history["test_dice"].append(extra["dice"])
            history["test_dice_damaged"].append(extra["dice_damaged"])
            history["test_pr_auc"].append(extra["pr_auc"] if extra["pr_auc"] is not None else 0.0)
            history["test_loss"].append(mean_test_loss)
            history["train_accuracy"].append(train_instance_acc)
            history["train_loss"].append(mean_epoch_loss)
            history["learning_rate"].append(current_lr)
            history["epoch_time"].append(epoch_time)

            current_metric = {
                "accuracy": instance_acc,
                "recall": recall,
                "pr_auc": pr_auc if pr_auc is not None else 0.0,
            }[args.early_stop_metric]

            if current_metric >= best_metric:
                best_metric = current_metric
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": classifier.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        args.early_stop_metric: current_metric,
                    },
                    str(checkpoints_dir / "best_model.pth"),
                )
                log_string(f"Best model saved (metric={args.early_stop_metric}={current_metric:.4f})")

                save_history_plots(history, exp_dir / "eval_plots")
                try:
                    (exp_dir / "eval_plots").mkdir(parents=True, exist_ok=True)
                    with open(exp_dir / "eval_plots" / "history.json", "w") as hfile:
                        json.dump(history, hfile, indent=2)
                except Exception:
                    pass

                early_stop_counter = 0
            else:
                early_stop_counter += 1

            writer.add_scalar("Training/Early_Stop_Counter", early_stop_counter, epoch)
            writer.add_scalar("Training/Best_Metric", best_metric, epoch)

            if early_stop_counter >= args.early_stop_patience:
                log_string(f"Early stopping at epoch {epoch + 1}")
                break

    # --- final save -------------------------------------------------------
    save_checkpoint(
        {
            "epoch": epoch,
            "model_state_dict": classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        str(checkpoints_dir / "last_model.pth"),
    )

    writer.close()
    log_string(f"TensorBoard logs saved to: {tensorboard_dir}")
    return exp_dir
