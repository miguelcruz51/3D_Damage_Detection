"""
DGCNN damage segmentation model.

Architecture: Dynamic Graph CNN (Wang et al., 2019) adapted for binary
per-point semantic segmentation of surface damage on metallic components.
The model dynamically constructs k-NN graphs in feature space at each layer.

Input format:  [B, C, N]  — B=batch, C=3 (XYZ) or 6 (XYZ+normals), N=points
Output format: [B, N, 2]  — per-point log-probabilities (healthy / damaged)

The forward pass returns (log_probs, None); the second element is kept for
API consistency with other models that return a feature tensor.

Best result: PR-AUC 0.9058 with LovaszBoundaryLoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    """
    Find k nearest neighbours for each point using pairwise distances.

    Args:
        x: [B, C, N]
        k: number of neighbours

    Returns:
        idx: [B, N, k]  neighbour indices
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    Construct edge features for each point and its k-NN neighbours.

    For each point, the feature is (neighbour - centre) concatenated with
    the centre features, yielding 2C edge features per edge.

    Args:
        x:   [B, C, N]
        k:   number of neighbours
        idx: optional pre-computed neighbour indices [B, N, k]

    Returns:
        feature: [B, 2C, N, k]
    """
    B, C, N = x.size()
    if idx is None:
        idx = knn(x, k=k)

    device = x.device
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()
    feature = x.view(B * N, -1)[idx, :]
    feature = feature.view(B, N, k, C)

    x = x.view(B, N, 1, C).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature


class get_model(nn.Module):
    """
    DGCNN segmentation model for binary damage detection.

    Args:
        num_classes:    number of output classes (default 2)
        k:              number of nearest neighbours for graph construction
        normal_channel: if True, input uses 6 channels (XYZ + normals);
                        if False, 3 channels (XYZ only)
    """

    def __init__(self, num_classes=2, k=20, normal_channel=False):
        super(get_model, self).__init__()
        self.k = k
        self.normal_channel = normal_channel
        in_channels = 6 if normal_channel else 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(64 + 64 + 128 + 256, 1024, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)

        # Input to conv6: local (512) + global (1024) = 1536
        self.conv6 = nn.Sequential(
            nn.Conv1d(1536, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv8 = nn.Conv1d(256, num_classes, kernel_size=1, bias=False)

    def forward(self, x):
        """
        Args:
            x: [B, C, N]

        Returns:
            x:    [B, N, num_classes]  per-point log-probabilities
            None: placeholder for API consistency
        """
        batch_size, _, num_points = x.size()

        x1 = self.conv1(get_graph_feature(x, k=self.k)).max(dim=-1)[0]
        x2 = self.conv2(get_graph_feature(x1, k=self.k)).max(dim=-1)[0]
        x3 = self.conv3(get_graph_feature(x2, k=self.k)).max(dim=-1)[0]
        x4 = self.conv4(get_graph_feature(x3, k=self.k)).max(dim=-1)[0]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x5 = self.conv5(x_cat)
        x_max = x5.max(dim=-1, keepdim=True)[0].repeat(1, 1, num_points)

        x = torch.cat((x_cat, x_max), dim=1)

        x = self.conv6(x)
        x = self.dp1(x)
        x = self.conv7(x)
        x = self.dp2(x)
        x = self.conv8(x)

        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x, dim=-1)

        return x, None


class get_loss(nn.Module):
    """
    Weighted NLL loss for DGCNN.

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

    def forward(self, pred, target, trans_feat=None):
        """
        Args:
            pred:   [B, N, num_classes]  log-probabilities
            target: [B, N]               ground-truth labels (0/1)
        """
        pred = pred.view(-1, pred.size(-1))
        target = target.view(-1)

        if self.class_weights is not None:
            if pred.is_cuda:
                self.class_weights = self.class_weights.to(pred.device)
            return F.nll_loss(pred, target, weight=self.class_weights)
        else:
            return F.nll_loss(pred, target)
