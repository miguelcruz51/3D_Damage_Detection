"""
Loss functions for binary point-cloud damage segmentation.

This module consolidates all loss function implementations used across the
experiment suite. The task is a highly imbalanced binary segmentation problem
(~1.5–2% damaged points), so several specialised losses are provided.

Available losses
----------------
- get_loss              : Default weighted NLL loss (used by all baseline models)
- FocalLoss             : Focuses training on hard/minority examples
- CrossEntropyDiceLoss  : CE + soft Dice hybrid
- TverskyLoss           : Generalised Dice with separate FP/FN penalties
- LovaszSoftmax         : Directly optimises mean IoU via the Lovász extension
- BoundaryAwareLoss     : Up-weights points near the healthy/damaged boundary
- LovaszBoundaryLoss    : Hybrid of LovászSoftmax and BoundaryAwareLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Default loss (used by all models as MODEL.get_loss())
# ---------------------------------------------------------------------------

class get_loss(nn.Module):
    """
    Weighted negative log-likelihood loss — the default loss used by all
    baseline damage-segmentation models.

    The model's forward pass returns log-softmax probabilities, so NLLLoss is
    the correct pairing (equivalent to cross-entropy when applied to
    log-probabilities).

    Parameters
    ----------
    class_weights : list[float] | None
        Per-class weights ``[w_healthy, w_damaged]``.  Set ``w_damaged`` high
        (e.g. 45.0) to compensate for class imbalance.  If ``None``, uniform
        weights are used.
    """

    def __init__(self, class_weights=None):
        super(get_loss, self).__init__()
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted NLL loss.

        Parameters
        ----------
        pred : Tensor[B*N, 2]
            Log-softmax probabilities for each point (output of ``log_softmax``).
        target : Tensor[B*N]
            Ground-truth class indices (0 = healthy, 1 = damaged).

        Returns
        -------
        Tensor
            Scalar loss value.
        """
        weights = (
            self.class_weights.to(pred.device)
            if self.class_weights is not None
            else None
        )
        return F.nll_loss(pred, target, weight=weights)


# ---------------------------------------------------------------------------
# FocalLoss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for class-imbalanced segmentation.

    Down-weights easy (well-classified) examples and focuses training on hard
    examples — particularly useful when damaged points are rare.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Parameters
    ----------
    alpha : Tensor | list[float] | None
        Per-class weighting factor applied *before* the focal modulation.
        Typically set to ``[0.25, 0.75]`` to further up-weight the damaged
        class.  If ``None``, no per-class weighting is applied.
    gamma : float
        Focusing exponent.  ``gamma=0`` reduces to standard cross-entropy.
        Typical values: 1.5–2.0.
    reduction : str
        ``'mean'`` (default), ``'sum'``, or ``'none'``.
    """

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : Tensor[B*N, C]
            Raw (unnormalised) logits.
        targets : Tensor[B*N]
            Ground-truth class indices.

        Returns
        -------
        Tensor
            Scalar (or per-element) focal loss.
        """
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ---------------------------------------------------------------------------
# CrossEntropyDiceLoss
# ---------------------------------------------------------------------------

class CrossEntropyDiceLoss(nn.Module):
    """
    Hybrid Cross-Entropy + soft-Dice loss.

    Combines the per-point discrimination of cross-entropy with the overlap
    optimisation of the Dice coefficient.  Particularly effective when both
    precision and recall on the minority class matter.

    Parameters
    ----------
    ce_weight : float
        Contribution of cross-entropy term (default ``0.7``).
    dice_weight : float
        Contribution of soft-Dice term (default ``0.3``).
    class_weights : Tensor | list[float] | None
        Per-class weights forwarded to ``CrossEntropyLoss``.
    """

    def __init__(
        self,
        ce_weight: float = 0.7,
        dice_weight: float = 0.3,
        class_weights=None,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        weights = (
            torch.tensor(class_weights, dtype=torch.float32)
            if isinstance(class_weights, list)
            else class_weights
        )
        self.ce = nn.CrossEntropyLoss(weight=weights)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : Tensor[B*N, C]
            Raw logits.
        targets : Tensor[B*N]
            Ground-truth class indices.

        Returns
        -------
        Tensor
            Scalar combined loss.
        """
        ce = self.ce(inputs, targets)

        probs = torch.softmax(inputs, dim=1)
        targets_1hot = F.one_hot(targets, num_classes=inputs.size(1)).float().to(inputs.device)

        intersection = (probs * targets_1hot).sum(dim=0)
        dice = 1 - (2 * intersection + 1e-6) / (
            probs.sum(dim=0) + targets_1hot.sum(dim=0) + 1e-6
        )
        dice = dice.mean()

        return self.ce_weight * ce + self.dice_weight * dice


# ---------------------------------------------------------------------------
# TverskyLoss
# ---------------------------------------------------------------------------

class TverskyLoss(nn.Module):
    """
    Tversky Loss — a generalisation of Dice that allows separate control over
    false-positive and false-negative penalties.

    Setting ``alpha < beta`` penalises false negatives more heavily, which is
    desirable when missing damaged points is more costly than false alarms.

    Reference: Salehi et al., "Tversky loss function for image segmentation
    using 3D fully convolutional deep networks", MICCAI 2017.

    Parameters
    ----------
    alpha : float
        False-positive penalty weight (default ``0.3``).
    beta : float
        False-negative penalty weight (default ``0.7``).
        ``alpha + beta`` does not need to equal 1, but typically does.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : Tensor[B*N, C]
            Raw logits.
        targets : Tensor[B*N]
            Ground-truth class indices.

        Returns
        -------
        Tensor
            Scalar Tversky loss in [0, 1].
        """
        probs = torch.softmax(inputs, dim=1)
        targets_1hot = F.one_hot(targets, num_classes=inputs.size(1)).float().to(inputs.device)

        TP = (probs * targets_1hot).sum()
        FP = (probs * (1 - targets_1hot)).sum()
        FN = ((1 - probs) * targets_1hot).sum()
        tversky = TP / (TP + self.alpha * FP + self.beta * FN + 1e-6)
        return 1 - tversky


# ---------------------------------------------------------------------------
# LovaszSoftmax
# ---------------------------------------------------------------------------

def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute gradient of the Lovász extension w.r.t sorted errors."""
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


class LovaszSoftmax(nn.Module):
    """
    Lovász-Softmax loss — directly optimises the mean IoU.

    Unlike Dice/Tversky which use soft surrogates, the Lovász extension
    provides a tight convex surrogate for the IoU.  This tends to give
    better-calibrated precision/recall trade-offs on imbalanced datasets.

    Reference: Berman et al., "The Lovász-Softmax loss: A tractable surrogate
    for the optimisation of the intersection-over-union measure in neural
    networks", CVPR 2018.

    Input shapes
    ------------
    Accepts logits in several formats:
    - ``[B*N, C]`` — already flattened
    - ``[B, N, C]`` — batch × points × classes
    - ``[B, C, N]`` — batch × classes × points (detected automatically)
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : Tensor
            Raw logits (see class docstring for accepted shapes).
        labels : Tensor
            Ground-truth class indices.

        Returns
        -------
        Tensor
            Scalar mean Lovász loss across all classes present in the batch.
        """
        # Normalise to [B*N, C] and [B*N]
        if logits.dim() == 3:
            if logits.size(1) == 2:  # [B, C, N] — C is the class dimension
                logits = logits.permute(0, 2, 1).contiguous()  # → [B, N, C]
            B, N, C = logits.shape
            logits = logits.view(-1, C)
            labels = labels.view(-1)

        probs = F.softmax(logits, dim=1)
        C = probs.size(1)
        losses = []

        for c in range(C):
            fg = (labels == c).float()
            if fg.sum() == 0:
                continue
            errors = (fg - probs[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            grad = _lovasz_grad(fg_sorted)
            losses.append(torch.dot(errors_sorted, grad))

        if not losses:
            return torch.tensor(0.0, device=logits.device)
        return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# BoundaryAwareLoss
# ---------------------------------------------------------------------------

class BoundaryAwareLoss(nn.Module):
    """
    Boundary-Aware Loss — up-weights points near healthy/damaged boundaries.

    Damage regions in real components have gradual, ambiguous boundaries.
    This loss detects boundary points (those with at least one neighbour of a
    different class in a local window) and multiplies their per-point CE loss
    by a boundary emphasis factor, encouraging the model to be precise at
    transition regions.

    Parameters
    ----------
    boundary_weight : float
        Multiplicative emphasis for boundary points.  A value of ``2.0`` means
        boundary points contribute twice as much to the total loss.
    k_neighbors : int
        Half-window size for boundary detection.  Point ``i`` is considered a
        boundary point if any point in ``[i-k, i+k]`` has a different label.
    """

    def __init__(self, boundary_weight: float = 2.0, k_neighbors: int = 5):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.k_neighbors = k_neighbors
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def _detect_boundaries(self, labels: torch.Tensor, k: int = 5) -> torch.Tensor:
        """
        Detect boundary points in a batch of label sequences.

        Parameters
        ----------
        labels : Tensor[B, N]
            Per-point class labels.
        k : int
            Half-window size.

        Returns
        -------
        Tensor[B, N]
            Float mask; 1.0 at boundary points, 0.0 elsewhere.
        """
        B, N = labels.shape
        boundary_mask = torch.zeros_like(labels, dtype=torch.float)
        for b in range(B):
            for i in range(N):
                start = max(0, i - k)
                end = min(N, i + k + 1)
                neighbors = labels[b, start:end]
                if (neighbors != labels[b, i]).any():
                    boundary_mask[b, i] = 1.0
        return boundary_mask

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : Tensor
            Raw logits.  Accepted shapes: ``[B*N, C]``, ``[B, N, C]``, or
            ``[B, C, N]``.
        target : Tensor
            Ground-truth class indices.  Accepted shapes: ``[B*N]`` or ``[B, N]``.

        Returns
        -------
        Tensor
            Scalar boundary-weighted CE loss.
        """
        if pred.dim() == 3:
            if pred.size(1) == 2:  # [B, C, N] → [B, N, C]
                pred = pred.permute(0, 2, 1).contiguous()
            B, N, C = pred.shape
            target_2d = target.view(B, N) if target.dim() == 1 else target
            pred = pred.view(-1, C)
            target_flat = target.view(-1)
        else:
            B, N = 1, pred.size(0)
            target_2d = target.view(B, N)
            target_flat = target.view(-1)

        ce_loss = self.ce(pred, target_flat)
        boundary_mask = self._detect_boundaries(target_2d, k=self.k_neighbors)
        boundary_mask = boundary_mask.view(-1)

        weights = 1.0 + self.boundary_weight * boundary_mask
        return (ce_loss * weights).mean()


# ---------------------------------------------------------------------------
# LovaszBoundaryLoss
# ---------------------------------------------------------------------------

class LovaszBoundaryLoss(nn.Module):
    """
    Lovász-Boundary hybrid loss.

    Combines the global IoU optimisation of :class:`LovaszSoftmax` with the
    local boundary precision of :class:`BoundaryAwareLoss`.  This was the
    best-performing loss function in the experiment suite, achieving PR-AUC
    0.9201 with PointNet2-SSG.

    Parameters
    ----------
    lovasz_weight : float
        Weight for the LovászSoftmax component (default ``0.5``).
    boundary_weight : float
        Weight for the BoundaryAwareLoss component (default ``0.5``).
    boundary_emphasis : float
        Boundary emphasis factor forwarded to :class:`BoundaryAwareLoss`
        (default ``2.0``).
    """

    def __init__(
        self,
        lovasz_weight: float = 0.5,
        boundary_weight: float = 0.5,
        boundary_emphasis: float = 2.0,
    ):
        super().__init__()
        self.lovasz = LovaszSoftmax()
        self.boundary = BoundaryAwareLoss(boundary_weight=boundary_emphasis)
        self.lovasz_weight = lovasz_weight
        self.boundary_weight = boundary_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : Tensor
            Raw logits (shapes accepted by :class:`LovaszSoftmax`).
        target : Tensor
            Ground-truth class indices.

        Returns
        -------
        Tensor
            Scalar combined loss.
        """
        lovasz_loss = self.lovasz(pred, target)
        boundary_loss = self.boundary(pred, target)
        return self.lovasz_weight * lovasz_loss + self.boundary_weight * boundary_loss
