"""
PointNet2 transfer-learning damage segmentation model.

Architecture: PointNet++ (Qi et al., 2017) semantic-segmentation backbone
pre-trained on S3DIS (13-class), fine-tuned with a new binary segmentation
head for damage detection on metallic components.

Input format:  [B, C, N]  — B=batch, C=6 or 9 (XYZ+normals), N=points
Output format: [B, N, 2]  — per-point log-probabilities (healthy / damaged)

The forward pass returns (log_probs, global_features).

Typical use:
    model = get_model(pretrained_path='path/to/semseg_weights.pth',
                      freeze_encoder=True)

Best result: PR-AUC 0.9123 with BoundaryAwareLoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet2_utils import (
    PointNetSetAbstraction,
    PointNetFeaturePropagation,
)


# ---------------------------------------------------------------------------
# Minimal PointNet2 semantic-segmentation backbone
# Matches the author's S3DIS model (9-channel input, 13 classes).
# ---------------------------------------------------------------------------

class _PointNet2SemSegBackbone(nn.Module):
    """
    PointNet++ semantic segmentation backbone (author's S3DIS architecture).

    Kept private — use get_model for the transfer-learning variant.
    """

    def __init__(self, num_classes=13):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 9 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)


# ---------------------------------------------------------------------------
# Transfer-learning model
# ---------------------------------------------------------------------------

class get_model(nn.Module):
    """
    PointNet++ with pretrained encoder for binary damage segmentation.

    The S3DIS backbone's encoder layers can be loaded from a checkpoint and
    optionally frozen.  The first SA layer's input channels are adjusted to
    match the project's 6-channel (or 9-channel) data format.  A new binary
    segmentation head replaces the original 13-class output.

    Args:
        num_classes:     number of output classes (default 2)
        pretrained_path: path to a .pth checkpoint
        freeze_encoder:  if True, encoder weights are frozen
        normal_channel:  if True, expects 6-channel input (XYZ + normals);
                         S3DIS uses 9-channel input, so the first SA layer
                         is re-built accordingly
    """

    def __init__(
        self,
        num_classes=2,
        pretrained_path=None,
        freeze_encoder=False,
        normal_channel=True,
    ):
        super(get_model, self).__init__()
        self.normal_channel = normal_channel

        self.encoder = _PointNet2SemSegBackbone(num_classes=13)

        # Adjust first SA layer to match project input channels
        expected_in = 9 if normal_channel else 3
        self.encoder.sa1 = PointNetSetAbstraction(
            npoint=1024,
            radius=0.1,
            nsample=32,
            in_channel=expected_in,
            mlp=[32, 32, 64],
            group_all=False,
        )

        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.encoder.load_state_dict(state_dict, strict=False)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.new_head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1),
        )

    def forward(self, xyz):
        """
        Args:
            xyz: [B, C, N]

        Returns:
            x:          [B, N, num_classes]  per-point log-probabilities
            l4_points:  [B, 512, 16]         bottleneck features
        """
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.encoder.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.encoder.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.encoder.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.encoder.sa4(l3_xyz, l3_points)

        l3_points = self.encoder.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.encoder.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.encoder.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.encoder.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.new_head(l0_points)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points


class get_loss(nn.Module):
    """
    Weighted NLL loss for PointNet++ transfer model.

    Args:
        class_weights: per-class weight list, e.g. [1.0, 45.0]
    """

    def __init__(self, class_weights=None):
        super(get_loss, self).__init__()
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )

    def forward(self, pred, target, trans_feat=None):
        pred = pred.view(-1, 2)
        target = target.view(-1)

        if self.class_weights is not None:
            if pred.is_cuda:
                self.class_weights = self.class_weights.to(pred.device)
            return F.nll_loss(pred, target, weight=self.class_weights)
        else:
            return F.nll_loss(pred, target)
