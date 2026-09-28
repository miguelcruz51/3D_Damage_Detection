#!/usr/bin/env python
"""
Training entry point for point-cloud damage segmentation.

Usage:
    python scripts/train.py --config configs/pointnet2_boundary_aware.yaml
    python scripts/train.py --config configs/pointnet2_boundary_aware.yaml \
        --log_dir my_run --gpu 0 --resume log/part_seg/my_run/checkpoints/last_model.pth

The full set of hyperparameters is read from the YAML config file. The optional
CLI overrides (``--log_dir``, ``--gpu``, ``--resume``) take precedence over the
corresponding config values.
"""

import argparse
import os
import sys

import yaml

# Make clean_repo/src importable (top-level packages: data, models, training, ...).
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from training.trainer import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a damage-segmentation model from a YAML config.")
    # --config is required so the script cannot be run without a config file.
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--log_dir", type=str, default=None, help="Override the experiment log directory name.")
    parser.add_argument("--gpu", type=str, default=None, help="Override the CUDA device index (e.g. '0').")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume training from.")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load a YAML config file into a dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    return config


def main() -> None:
    args = parse_args()

    config = load_config(args.config)

    # Apply CLI overrides.
    if args.log_dir is not None:
        config["log_dir"] = args.log_dir
    if args.resume is not None:
        config["resume"] = args.resume
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    exp_dir = train(config)
    print(f"Training complete. Experiment directory: {exp_dir}")


if __name__ == "__main__":
    main()
