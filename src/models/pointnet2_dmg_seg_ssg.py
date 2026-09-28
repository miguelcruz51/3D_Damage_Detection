"""
PointNet++ (SSG) damage segmentation model.

Architecture: PointNet++ Single-Scale Grouping (Qi et al., 2017) adapted for
binary per-point semantic segmentation of surface damage on metallic components.

Input format:  [B, C, N]  — B=batch, C=3 (XYZ) or 6 (XYZ+normals), N=points
Output format: [B, N, 2]  — per-point log-probabilities (healthy / damaged)

The forward pass returns (log_probs, global_features) where global_features
is the bottleneck feature tensor [B, 1024, 1].

Best result: PR-AUC 0.9201 with BoundaryAwareLoss (172 epochs).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation


def get_model(normal_channel=False):
    """Factory function — returns a get_model_class instance."""
    return get_model_class(normal_channel=normal_channel)


class get_model_class(nn.Module):
    """
    PointNet++ SSG segmentation backbone for binary damage detection.

    Args:
        normal_channel: if True, concatenate surface normals to XYZ input
                        (total input channels = 6); default False (3 channels)
    """

    def __init__(self, normal_channel=False):
        super(get_model_class, self).__init__()
        self.normal_channel = normal_channel
        additional_channel = 3 if normal_channel else 0

        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=6 + additional_channel, mlp=[64, 64, 128], group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3, mlp=[128, 128, 256], group_all=False
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True
        )

        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(
            in_channel=128 + 6 + additional_channel, mlp=[128, 128, 128]
        )

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, 2, 1)

    def forward(self, xyz, cls_label=None):
        """
        Args:
            xyz:       [B, C, N]  input point cloud (C=3 or 6)
            cls_label: ignored — kept for API compatibility with the training
                       framework; pass None or omit

        Returns:
            x:          [B, N, 2]     per-point log-probabilities
            l3_points:  [B, 1024, 1]  global bottleneck features
        """
        B, C, N = xyz.shape

        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz
            l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)

        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points


class get_loss(nn.Module):
    """
    Weighted NLL loss for PointNet++ SSG.

    Args:
        class_weights: per-class weight list, e.g. [1.0, 45.0] for imbalance
    """

    def __init__(self, class_weights=None):
        super(get_loss, self).__init__()
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )

    def forward(self, pred, target, _=None):
        """
        Args:
            pred:   [B, N, 2]  log-probabilities
            target: [B, N]     ground-truth labels (0/1)
        """
        pred = pred.view(-1, 2)
        target = target.view(-1)

        if self.class_weights is not None:
            if pred.is_cuda:
                self.class_weights = self.class_weights.to(pred.device)
            return F.nll_loss(pred, target, weight=self.class_weights)
        else:
            return F.nll_loss(pred, target)
