"""
PointTransformer (v3) damage segmentation model.

Architecture: Point Transformer (Zhao et al., 2021) adapted for binary
per-point semantic segmentation of surface damage on metallic components.
Implements a hierarchical encoder-decoder with local self-attention blocks
and learned 3-D position encodings.

Input format:  [B, C, N]  — B=batch, C=3 (XYZ) or 6 (XYZ+normals), N=points
Output format: [B, N, 2]  — per-point log-probabilities (healthy / damaged)

The forward pass returns (log_probs, None); the second element is kept for
API consistency.

Best result: PR-AUC 0.8254 with BoundaryAwareLoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def knn(x, k):
    """
    k-nearest neighbours via pairwise distances.

    Args:
        x: [B, 3, N]
        k: number of neighbours

    Returns:
        idx: [B, N, k]
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def index_points(points, idx):
    """
    Gather points by index tensor.

    Args:
        points: [B, N, C]
        idx:    [B, ...]

    Returns:
        [B, ..., C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    return points[batch_indices, idx, :]


# ---------------------------------------------------------------------------
# Attention building blocks
# ---------------------------------------------------------------------------

class PositionEncoding(nn.Module):
    """
    Learnable 3-D position encoding MLP.

    Args:
        channels: output embedding dimension
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.pos_mlp = nn.Sequential(
            nn.Conv1d(3, channels // 2, 1),
            nn.BatchNorm1d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 2, channels, 1),
            nn.BatchNorm1d(channels),
        )

    def forward(self, xyz):
        """
        Args:
            xyz: [B, 3, N]

        Returns:
            [B, channels, N]
        """
        return self.pos_mlp(xyz)


class LocalAttentionBlock(nn.Module):
    """
    Local self-attention block over k-NN neighbours with relative position encoding.

    Args:
        channels: feature dimension
        k:        neighbourhood size
    """

    def __init__(self, channels, k=16):
        super().__init__()
        self.channels = channels
        self.k = k

        self.q_conv = nn.Conv1d(channels, channels, 1)
        self.k_conv = nn.Conv1d(channels, channels, 1)
        self.v_conv = nn.Conv1d(channels, channels, 1)

        self.pos_enc = PositionEncoding(channels)

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
        )

        self.out_conv = nn.Conv1d(channels, channels, 1)
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x, xyz):
        """
        Args:
            x:   [B, C, N]
            xyz: [B, 3, N]

        Returns:
            [B, C, N]
        """
        B, C, N = x.shape

        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        pos_enc = self.pos_enc(xyz)
        k = k + pos_enc

        knn_idx = knn(xyz, self.k)

        k_grouped = index_points(k.transpose(1, 2), knn_idx)
        v_grouped = index_points(v.transpose(1, 2), knn_idx)
        xyz_grouped = index_points(xyz.transpose(1, 2), knn_idx)

        xyz_relative = xyz_grouped - xyz.transpose(1, 2).unsqueeze(2)
        pos_enc_rel = self.pos_enc(
            xyz_relative.permute(0, 3, 1, 2).contiguous().view(B, 3, -1)
        ).view(B, C, N, self.k).permute(0, 2, 3, 1)

        k_grouped = k_grouped + pos_enc_rel

        q_expanded = q.transpose(1, 2).unsqueeze(2)
        attn_input = (q_expanded - k_grouped).permute(0, 3, 1, 2)
        attn_weights = F.softmax(self.attn_mlp(attn_input), dim=-1)

        out = torch.sum(attn_weights * v_grouped.permute(0, 3, 1, 2), dim=-1)
        out = self.norm(self.out_conv(out) + x)
        return F.relu(out)


class TransitionDown(nn.Module):
    """
    Downsample points by stride-based selection and increase feature channels.

    Args:
        in_channels:  input feature channels
        out_channels: output feature channels
        stride:       downsampling factor
    """

    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.stride = stride
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, xyz):
        x_down = x[:, :, :: self.stride]
        xyz_down = xyz[:, :, :: self.stride]
        return self.conv(x_down), xyz_down


class TransitionUp(nn.Module):
    """
    Upsample features by nearest-neighbour interpolation and skip-connection.

    Args:
        in_channels:  combined (interpolated + skip) feature channels
        out_channels: output feature channels
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_low, xyz_low, x_high, xyz_high):
        B, C_low, N_low = x_low.shape
        _, _, N_high = x_high.shape

        if N_low == N_high:
            x_concat = torch.cat([x_low, x_high], dim=1)
        else:
            x_low_interp = F.interpolate(x_low, size=N_high, mode='nearest')
            x_concat = torch.cat([x_low_interp, x_high], dim=1)

        return self.conv(x_concat)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class get_model(nn.Module):
    """
    PointTransformer hierarchical attention model for binary damage segmentation.

    Args:
        num_classes:    number of output classes (default 2)
        normal_channel: if True, input uses 6 channels (XYZ + normals);
                        if False, 3 channels (XYZ only)
    """

    def __init__(self, num_classes=2, normal_channel=True):
        super().__init__()
        self.normal_channel = normal_channel
        in_channels = 6 if normal_channel else 3

        self.input_conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.enc_conv1 = nn.Sequential(
            nn.Conv1d(32, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True)
        )
        self.enc_attn1 = LocalAttentionBlock(64, k=16)
        self.enc_down1 = TransitionDown(64, 128, stride=2)

        self.enc_conv2 = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True)
        )
        self.enc_attn2 = LocalAttentionBlock(128, k=16)
        self.enc_down2 = TransitionDown(128, 256, stride=2)

        self.enc_conv3 = nn.Sequential(
            nn.Conv1d(256, 256, 1), nn.BatchNorm1d(256), nn.ReLU(inplace=True)
        )
        self.enc_attn3 = LocalAttentionBlock(256, k=16)
        self.enc_down3 = TransitionDown(256, 512, stride=2)

        self.bottleneck_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True)
        )
        self.bottleneck_attn = LocalAttentionBlock(512, k=8)

        self.dec_up3 = TransitionUp(512 + 256, 256)
        self.dec_attn3 = LocalAttentionBlock(256, k=16)

        self.dec_up2 = TransitionUp(256 + 128, 128)
        self.dec_attn2 = LocalAttentionBlock(128, k=16)

        self.dec_up1 = TransitionUp(128 + 64, 64)
        self.dec_attn1 = LocalAttentionBlock(64, k=16)

        self.classifier = nn.Sequential(
            nn.Conv1d(64 + 32, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(64, num_classes, 1),
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, N]

        Returns:
            out:  [B, N, num_classes]  per-point log-probabilities
            None: placeholder for API consistency
        """
        B, C, N = x.shape
        xyz = x[:, :3, :]

        x0 = self.input_conv(x)
        xyz0 = xyz

        x1 = self.enc_attn1(self.enc_conv1(x0), xyz0)
        x1_down, xyz1 = self.enc_down1(x1, xyz0)

        x2 = self.enc_attn2(self.enc_conv2(x1_down), xyz1)
        x2_down, xyz2 = self.enc_down2(x2, xyz1)

        x3 = self.enc_attn3(self.enc_conv3(x2_down), xyz2)
        x3_down, xyz3 = self.enc_down3(x3, xyz2)

        x_bottle = self.bottleneck_attn(self.bottleneck_conv(x3_down), xyz3)

        x_up3 = self.dec_attn3(self.dec_up3(x_bottle, xyz3, x3, xyz2), xyz2)
        x_up2 = self.dec_attn2(self.dec_up2(x_up3, xyz2, x2, xyz1), xyz1)
        x_up1 = self.dec_attn1(self.dec_up1(x_up2, xyz1, x1, xyz0), xyz0)

        x_final = torch.cat([x_up1, x0], dim=1)
        out = self.classifier(x_final)
        out = out.transpose(2, 1).contiguous()
        out = F.log_softmax(out, dim=-1)

        return out, None


class get_loss(nn.Module):
    """
    Weighted NLL loss for PointTransformer.

    Args:
        class_weights: per-class weight list, e.g. [1.0, 45.0] for imbalance
    """

    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )

    def forward(self, pred, target, trans_feat=None):
        pred = pred.view(-1, pred.size(-1))
        target = target.view(-1)

        if self.class_weights is not None:
            if pred.is_cuda:
                self.class_weights = self.class_weights.to(pred.device)
            return F.nll_loss(pred, target, weight=self.class_weights)
        else:
            return F.nll_loss(pred, target)
