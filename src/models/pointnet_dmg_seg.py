"""
PointNet damage segmentation model.

Architecture: PointNet (Qi et al., 2017) adapted for binary per-point
semantic segmentation of surface damage on metallic components.

Input format:  [B, C, N]  — B=batch, C=3 (XYZ) or 6 (XYZ+normals), N=points
Output format: [B, N, 2]  — per-point log-probabilities (healthy / damaged)

The forward pass returns (log_probs, trans_feat) where trans_feat is the
feature-transform matrix used for the optional regularization loss.

Source: Xu Yan, "Pointnet_Pointnet2_pytorch" (2019), adapted for this project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet_utils import STN3d, STNkd, feature_transform_reguliarzer


class get_model(nn.Module):
    """
    PointNet segmentation model adapted for binary damage detection.

    Args:
        part_num:       number of output classes (default 2: healthy/damaged)
        normal_channel: if True, input is XYZ + surface normals (6 channels);
                        if False, input is XYZ only (3 channels)
    """

    def __init__(self, part_num=2, normal_channel=True):
        super(get_model, self).__init__()
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

        # Segmentation head: concatenation of global + local features → 4928 channels
        self.convs1 = nn.Conv1d(4928, 256, 1)
        self.convs2 = nn.Conv1d(256, 256, 1)
        self.convs3 = nn.Conv1d(256, 128, 1)
        self.convs4 = nn.Conv1d(128, part_num, 1)

        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, point_cloud):
        """
        Args:
            point_cloud: [B, C, N]  (C=3 or 6)

        Returns:
            net:        [B, N, part_num]  per-point log-probabilities
            trans_feat: [B, 128, 128]     feature-transform matrix for regularization
        """
        B, D, N = point_cloud.size()

        trans = self.stn(point_cloud)
        point_cloud = point_cloud.transpose(2, 1)

        if D > 3:
            point_cloud, feature = point_cloud.split(3, dim=2)
        point_cloud = torch.bmm(point_cloud, trans)
        if D > 3:
            point_cloud = torch.cat([point_cloud, feature], dim=2)

        point_cloud = point_cloud.transpose(2, 1)

        out1 = F.relu(self.bn1(self.conv1(point_cloud)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))

        trans_feat = self.fstn(out3)
        x = out3.transpose(2, 1)
        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)

        out4 = F.relu(self.bn4(self.conv4(net_transformed)))
        out5 = self.bn5(self.conv5(out4))

        out_max = torch.max(out5, 2, keepdim=True)[0].view(-1, 2048)
        expand = out_max.view(-1, 2048, 1).repeat(1, 1, N)
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)

        net = F.relu(self.bns1(self.convs1(concat)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)

        net = net.transpose(2, 1).contiguous()
        net = F.log_softmax(net.view(-1, self.part_num), dim=-1)
        net = net.view(B, N, self.part_num)

        return net, trans_feat


class get_loss(torch.nn.Module):
    """
    Weighted NLL loss with optional feature-transform regularization.

    Args:
        mat_diff_loss_scale: weight for the feature-transform regularizer
        class_weights:       per-class weight list, e.g. [1.0, 45.0] for imbalance
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
        """
        Args:
            pred:       [B, N, 2]  log-probabilities
            target:     [B, N]     ground-truth labels (0/1)
            trans_feat: optional feature-transform matrix for regularization
        """
        pred = pred.view(-1, 2)
        target = target.view(-1)

        if self.class_weights is not None:
            if pred.is_cuda:
                self.class_weights = self.class_weights.to(pred.device)
            loss = F.nll_loss(pred, target, weight=self.class_weights)
        else:
            loss = F.nll_loss(pred, target)

        if trans_feat is not None:
            mat_diff_loss = feature_transform_reguliarzer(trans_feat)
            loss = loss + mat_diff_loss * self.mat_diff_loss_scale

        return loss
