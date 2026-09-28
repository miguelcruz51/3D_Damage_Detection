"""
Checkpoint save/load utilities for damage-segmentation models.

Provides:
    - save_checkpoint    : Persist model/optimizer state (and arbitrary extras).
    - load_checkpoint    : Restore a full checkpoint into a model/optimizer.
    - load_pretrained_weights : Partial (shape-tolerant) weight loading for
      transfer learning, reconstructed from ``train_dmgseg9.py``.

The partial-loading behaviour mirrors the original training script: only keys
that exist in both the checkpoint and the target model *and* whose tensor
shapes match are copied. All other keys are skipped, allowing a checkpoint
trained on a related task (e.g. S3DIS semantic segmentation) to initialise a
damage-segmentation model.
"""

from __future__ import annotations

import os
from typing import Optional

import torch


def save_checkpoint(state: dict, path: str) -> None:
    """Save a checkpoint dictionary to disk.

    Creates the parent directory if it does not already exist.

    Args:
        state: Checkpoint contents. Typically contains ``epoch``,
            ``model_state_dict``, ``optimizer_state_dict`` and optional metrics
            such as ``pr_auc`` or ``accuracy``.
        path: Destination ``.pth`` file path.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> dict:
    """Load a full checkpoint into a model (and optionally an optimizer).

    Args:
        path: Path to a ``.pth`` checkpoint file.
        model: Model whose parameters will be restored in-place.
        optimizer: Optional optimizer whose state will be restored in-place.
        map_location: Device mapping for ``torch.load`` (default ``"cpu"``).

    Returns:
        The raw checkpoint dictionary that was loaded, so callers can read
        extra fields such as ``epoch`` or stored metrics.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = torch.load(path, map_location=map_location)

    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)

    if optimizer is not None and isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint if isinstance(checkpoint, dict) else {"model_state_dict": checkpoint}


def load_pretrained_weights(
    model: torch.nn.Module,
    pretrained_path: str,
    logger=None,
    map_location: str = "cpu",
) -> torch.nn.Module:
    """Partially load pretrained weights, ignoring shape/key mismatches.

    Reconstructed from ``train_dmgseg9.py``'s ``load_pretrained_weights``. Only
    keys present in both the checkpoint and the target model whose tensor
    shapes match are copied; every other key is skipped. This enables transfer
    learning from a checkpoint with a different output head or input dimension.

    Args:
        model: Target model to initialise (modified in-place).
        pretrained_path: Path to a pretrained ``.pth`` file. May be a raw
            state dict or a dict containing ``model_state_dict``.
        logger: Optional logger for progress messages. Falls back to ``print``.
        map_location: Device mapping for ``torch.load`` (default ``"cpu"``).

    Returns:
        The (in-place modified) model. If the file is missing or loading fails,
        the model is returned unchanged.
    """
    def _log(msg: str, level: str = "info") -> None:
        if logger is not None:
            getattr(logger, level, logger.info)(msg)
        else:
            print(msg)

    if not os.path.isfile(pretrained_path):
        _log(f"[WARN] Pretrained file not found: {pretrained_path}", "warning")
        return model

    try:
        checkpoint = torch.load(pretrained_path, map_location=map_location)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

        model_dict = model.state_dict()
        # Keep only matching keys with matching shapes.
        pretrained_dict = {
            k: v
            for k, v in checkpoint.items()
            if k in model_dict and v.size() == model_dict[k].size()
        }
        missing = set(model_dict.keys()) - set(pretrained_dict.keys())
        _log(
            f"[INFO] Loading pretrained weights: {len(pretrained_dict)} layers "
            f"matched, {len(missing)} skipped."
        )
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    except Exception as e:  # pragma: no cover - defensive parity with original
        _log(f"[ERROR] Failed to load pretrained weights: {e}", "error")

    return model
