"""
PointNet transfer-learning damage segmentation model.

Architecture: PointNet (Qi et al., 2017) encoder pre-trained on ShapeNet
part segmentation (13-class), fine-tuned with a new binary segmentation head
for damage detection on metallic components.

Input format:  [B, C, N]  — B=batch, C=6 (XYZ+normals, normal_channel=True), N=points
Output format: [B, N, 2]  — per-point log-probabilities (healthy / damaged)

The forward pass returns (log_probs, trans_feat) where trans_feat is the
feature-transform matrix for the optional regularization loss.

Typical use:
    model = get_model(pretrained_path='path/to/partseg_weights.pth',
                      freeze_encoder=True)

Best result: PR-AUC 0.7044 with LovaszBoundaryLoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet_utils import STN3d, STNkd, feature_transform_reguliarzer


# ---------------------------------------------------------------------------
# Minimal PointNet part-segmentation backbone (author's original architecture)
# Used only as the pretrained encoder; weights are loaded from a .pth file.
# ---------------------------------------------------------------------------

class _PointNetPartSegBackbone(nn.Module):
    """
    PointNet part-segmentation backbone matching the author's original
    architecture (part_num=50 or 13, normal_channel=True).

    Kept private — use get_model for the transfer-learning variant.
    """

    def __init__(self, part_num=13, normal_channel=True):
        super().__init__()
        channel = 6 if normal_channel else 3
        self.part_num = part_num
        self.stn = STN3d(channel)
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, 512, 1)
        self.conv5 = nn.Conv1d(512, 2048, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fstn = STNkd(k=128)
        # Original head uses 4944 = 2048 + 16 (one-hot label) + local features
        # We only need the encoder layers; the head is replaced.
        self.convs1 = nn.Conv1d(4944, 256, 1)
        self.convs2 = nn.Conv1d(256, 256, 1)
        self.convs3 = nn.Conv1d(256, 128, 1)
        self.convs4 = nn.Conv1d(128, part_num, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)


# ---------------------------------------------------------------------------
# Transfer-learning model
# ---------------------------------------------------------------------------

class get_model(nn.Module):
    """
    PointNet with pretrained encoder for binary damage segmentation.

    The encoder layers (STN, conv1–conv5, fstn) can be loaded from a
    ShapeNet part-segmentation checkpoint and optionally frozen. A new
    segmentation head is trained for the damage detection task.

    Args:
        part_num:        number of output classes (default 2)
        pretrained_path: path to a .pth checkpoint containing encoder weights
                         (state_dict or {'model_state_dict': ...})
        freeze_encoder:  if True, encoder weights are frozen during training
    """

    def __init__(self, part_num=2, pretrained_path=None, freeze_encoder=False):
        super(get_model, self).__init__()
        self.part_num = part_num

        self.encoder = _PointNetPartSegBackbone(part_num=13, normal_channel=True)

        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.encoder.load_state_dict(state_dict, strict=False)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.new_head = nn.Sequential(
            nn.Conv1d(4928, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, part_num, 1),
        )

    def forward(self, point_cloud):
        """
        Args:
            point_cloud: [B, C, N]  (C=6 for XYZ+normals)

        Returns:
            x:          [B, N, part_num]  per-point log-probabilities
            trans_feat: feature-transform matrix for regularization
        """
        B, D, N = point_cloud.size()

        grad_enabled = not (
            hasattr(self.encoder.conv1, 'weight')
            and not self.encoder.conv1.weight.requires_grad
        )

        with torch.set_grad_enabled(grad_enabled):
            trans = self.encoder.stn(point_cloud)
            pc = point_cloud.transpose(2, 1)

            if D > 3:
                pc, feature = pc.split(3, dim=2)
            pc = torch.bmm(pc, trans)
            if D > 3:
                pc = torch.cat([pc, feature], dim=2)
            pc = pc.transpose(2, 1)

            out1 = F.relu(self.encoder.bn1(self.encoder.conv1(pc)))
            out2 = F.relu(self.encoder.bn2(self.encoder.conv2(out1)))
            out3 = F.relu(self.encoder.bn3(self.encoder.conv3(out2)))

            trans_feat = self.encoder.fstn(out3)
            net_transformed = torch.bmm(out3.transpose(2, 1), trans_feat).transpose(2, 1)

            out4 = F.relu(self.encoder.bn4(self.encoder.conv4(net_transformed)))
            out5 = self.encoder.bn5(self.encoder.conv5(out4))

            out_max = torch.max(out5, 2, keepdim=True)[0].view(-1, 2048)
            expand = out_max.view(-1, 2048, 1).repeat(1, 1, N)
            concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)

        x = self.new_head(concat)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x, dim=-1)

        return x, trans_feat


class get_loss(nn.Module):
    """
    Weighted NLL loss with optional feature-transform regularization.

    Args:
        mat_diff_loss_scale: weight for the feature-transform regularizer
        class_weights:       per-class weight list, e.g. [1.0, 45.0]
    """

    def __init__(self, mat_diff_loss_scale=0.001, class_weights=None):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
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
            loss = F.nll_loss(pred, target, weight=self.class_weights)
        else:
            loss = F.nll_loss(pred, target)

        if trans_feat is not None:
            loss = loss + feature_transform_reguliarzer(trans_feat) * self.mat_diff_loss_scale

        return loss
