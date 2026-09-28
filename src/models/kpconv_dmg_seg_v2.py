"""
KPConv (v2) damage segmentation model.

Architecture: Kernel Point Convolution (Thomas et al., 2019) adapted for
binary per-point semantic segmentation of surface damage on metallic components.
Implements a U-Net style encoder-decoder with KPConv layers using a Gaussian
influence function and an icosahedron-based kernel point layout.

Input format:  [B, C, N]  — B=batch, C=3 (XYZ) or 6 (XYZ+normals), N=points
Output format: [B, N, 2]  — per-point log-probabilities (healthy / damaged)

The forward pass returns (log_probs, None); the second element is kept for
API consistency.

Best result: PR-AUC 0.8893 with BoundaryAwareLoss.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def farthest_point_sample(xyz, npoint):
    """
    Farthest Point Sampling.

    Args:
        xyz:    [B, N, 3]
        npoint: number of points to sample

    Returns:
        centroids: [B, npoint]  sampled indices
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def square_distance(src, dst):
    """
    Squared Euclidean distance between two point sets.

    Args:
        src: [B, N, 3]
        dst: [B, M, 3]

    Returns:
        [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Gather points by index.

    Args:
        points: [B, N, C]
        idx:    [B, ...] index tensor

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


def ball_query(radius, nsample, xyz, new_xyz):
    """
    Ball query with adaptive k-NN fallback and padding.

    Args:
        radius:  neighbourhood radius (used conceptually; implementation uses kNN)
        nsample: number of neighbours to return
        xyz:     all points   [B, N, 3]
        new_xyz: query points [B, S, 3]

    Returns:
        group_idx: [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    k = min(nsample, xyz.shape[1])
    _, group_idx = torch.topk(sqrdists, k, dim=-1, largest=False, sorted=False)

    if k < nsample:
        padding = group_idx[:, :, -1:].repeat(1, 1, nsample - k)
        group_idx = torch.cat([group_idx, padding], dim=-1)

    return group_idx


# ---------------------------------------------------------------------------
# KPConv building blocks
# ---------------------------------------------------------------------------

class KernelPointConvolution(nn.Module):
    """
    Kernel Point Convolution layer.

    Places K kernel points in a spherical arrangement and aggregates
    neighbour features weighted by a Gaussian influence function.

    Args:
        in_channels:  input feature dimension
        out_channels: output feature dimension
        kernel_size:  number of kernel points (default 15)
        radius:       kernel radius (same as neighbourhood radius)
        sigma:        Gaussian bandwidth
    """

    def __init__(self, in_channels, out_channels, kernel_size=15, radius=0.1, sigma=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.radius = radius
        self.sigma = sigma

        self.kernel_points = nn.Parameter(self._init_kernel_points())
        self.weights = nn.Parameter(
            torch.randn(kernel_size, in_channels, out_channels) * 0.02
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def _init_kernel_points(self):
        """Place kernel points using an icosahedron + random fill."""
        if self.kernel_size == 1:
            return torch.zeros(1, 3)

        kernel_points = [[0.0, 0.0, 0.0]]

        phi = (1 + math.sqrt(5)) / 2
        vertices = [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
        ]
        normalized = []
        for v in vertices:
            norm = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            normalized.append([v[0] / norm, v[1] / norm, v[2] / norm])

        for v in normalized[: self.kernel_size - 1]:
            kernel_points.append(v)

        while len(kernel_points) < self.kernel_size:
            u, v_r = np.random.uniform(0, 1), np.random.uniform(0, 1)
            theta = 2 * math.pi * u
            phi_r = math.acos(2 * v_r - 1)
            kernel_points.append([
                math.sin(phi_r) * math.cos(theta),
                math.sin(phi_r) * math.sin(theta),
                math.cos(phi_r),
            ])

        kp = torch.tensor(kernel_points[: self.kernel_size], dtype=torch.float32)
        return kp * self.radius

    def forward(self, xyz, features, new_xyz=None):
        """
        Args:
            xyz:      input point coordinates  [B, N, 3]
            features: input point features     [B, N, in_channels]
            new_xyz:  output coordinates       [B, M, 3]  (None → same as xyz)

        Returns:
            output_features: [B, M, out_channels]
        """
        if new_xyz is None:
            new_xyz = xyz

        B, N, _ = xyz.shape
        _, M, _ = new_xyz.shape

        neighbor_idx = ball_query(self.radius * 2, 32, xyz, new_xyz)
        neighbor_xyz = index_points(xyz, neighbor_idx)
        neighbor_features = index_points(features, neighbor_idx)

        relative_pos = neighbor_xyz - new_xyz.unsqueeze(2)

        kp = self.kernel_points.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        distances = torch.norm(relative_pos.unsqueeze(3) - kp, dim=-1)
        influences = torch.exp(-0.5 * (distances / self.sigma) ** 2)

        output_features = torch.zeros(B, M, self.out_channels, device=features.device)
        for k in range(self.kernel_size):
            weight = self.weights[k]
            influence = influences[:, :, :, k]
            weighted = neighbor_features * influence.unsqueeze(-1)
            aggregated = weighted.sum(dim=2)
            output_features = output_features + torch.matmul(aggregated, weight)

        output_features = output_features.transpose(1, 2)
        output_features = self.bn(output_features)
        output_features = F.relu(output_features)
        output_features = output_features.transpose(1, 2)

        return output_features


class KPConvBlock(nn.Module):
    """KPConv block with optional farthest-point subsampling."""

    def __init__(self, in_channels, out_channels, subsample_ratio=4, radius=0.1):
        super().__init__()
        self.subsample_ratio = subsample_ratio
        self.kpconv = KernelPointConvolution(in_channels, out_channels, radius=radius)

    def forward(self, xyz, features):
        B, N, _ = xyz.shape
        if self.subsample_ratio > 1:
            npoint = N // self.subsample_ratio
            fps_idx = farthest_point_sample(xyz, npoint)
            new_xyz = index_points(xyz, fps_idx)
        else:
            new_xyz = xyz

        new_features = self.kpconv(xyz, features, new_xyz)
        return new_xyz, new_features


class KPConvUpBlock(nn.Module):
    """KPConv decoder block: nearest-neighbour interpolation + skip connection."""

    def __init__(self, in_channels, skip_channels, out_channels, radius=0.2):
        super().__init__()
        self.kpconv = KernelPointConvolution(in_channels + skip_channels, out_channels, radius=radius)

    def forward(self, xyz1, features1, xyz2, features2):
        B, N1, _ = xyz1.shape
        _, N2, _ = xyz2.shape

        if N1 == N2:
            interpolated = features1
        else:
            dists = square_distance(xyz2, xyz1)
            idx = torch.argmin(dists, dim=-1)
            interpolated = index_points(features1, idx)

        combined = torch.cat([interpolated, features2], dim=-1)
        return self.kpconv(xyz2, combined, xyz2)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class get_model(nn.Module):
    """
    KPConv U-Net for binary damage segmentation.

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
            nn.ReLU()
        )

        self.enc_block1 = KPConvBlock(32, 64, subsample_ratio=1, radius=0.05)
        self.enc_block2 = KPConvBlock(64, 128, subsample_ratio=4, radius=0.1)
        self.enc_block3 = KPConvBlock(128, 256, subsample_ratio=4, radius=0.2)
        self.enc_block4 = KPConvBlock(256, 512, subsample_ratio=4, radius=0.4)

        self.bottleneck = KernelPointConvolution(512, 512, radius=0.8)

        self.dec_block4 = KPConvUpBlock(512, 256, 256, radius=0.4)
        self.dec_block3 = KPConvUpBlock(256, 128, 128, radius=0.2)
        self.dec_block2 = KPConvUpBlock(128, 64, 64, radius=0.1)
        self.dec_block1 = KPConvUpBlock(64, 32, 64, radius=0.05)

        self.classifier = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(64, num_classes, 1)
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
        xyz = x[:, :3, :].transpose(1, 2).contiguous()

        features = self.input_conv(x).transpose(1, 2).contiguous()

        xyz1, f1 = self.enc_block1(xyz, features)
        xyz2, f2 = self.enc_block2(xyz1, f1)
        xyz3, f3 = self.enc_block3(xyz2, f2)
        xyz4, f4 = self.enc_block4(xyz3, f3)

        f_bottle = self.bottleneck(xyz4, f4, xyz4)

        up4 = self.dec_block4(xyz4, f_bottle, xyz3, f3)
        up3 = self.dec_block3(xyz3, up4, xyz2, f2)
        up2 = self.dec_block2(xyz2, up3, xyz1, f1)
        up1 = self.dec_block1(xyz1, up2, xyz, features)

        out = self.classifier(up1.transpose(1, 2).contiguous())
        out = out.transpose(2, 1).contiguous()
        out = F.log_softmax(out, dim=-1)

        return out, None


class get_loss(nn.Module):
    """
    Weighted NLL loss for KPConv.

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
