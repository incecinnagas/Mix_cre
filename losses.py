from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = pred - target
        abs_err = torch.abs(err)
        quadratic = torch.minimum(abs_err, torch.tensor(self.delta, device=pred.device))
        linear = abs_err - quadratic
        return 0.5 * quadratic.pow(2) + self.delta * linear


def pearson_corr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = pred - pred.mean()
    target = target - target.mean()
    num = (pred * target).sum()
    den = torch.sqrt((pred.pow(2).sum() + eps) * (target.pow(2).sum() + eps))
    return num / den


def spearman_surrogate(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # 退化为 Pearson 近似（若需要更精确的排序近似，可引入 torchsort）
    return pearson_corr(pred, target)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_pt = -F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(log_pt)
        loss = -(1 - pt) ** self.gamma * log_pt
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def classification_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    focal: bool = False,
    class_weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if focal:
        return FocalLoss()(logits, target)
    try:
        return F.cross_entropy(logits, target, weight=class_weight, label_smoothing=float(label_smoothing))
    except TypeError:
        # 兼容旧版 PyTorch 无 label_smoothing 参数
        return F.cross_entropy(logits, target, weight=class_weight)


class MultiTaskLoss(nn.Module):
    def __init__(self, spearman_weight: float = 0.2, cls_weight: float = 0.3, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.huber = HuberLoss(delta=1.0)
        self.spearman_weight = spearman_weight
        self.cls_weight = cls_weight
        self.label_smoothing = float(label_smoothing)

    def forward(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        logits: torch.Tensor,
        species: torch.Tensor,
        class_weight: Optional[torch.Tensor] = None,
        use_focal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l_reg_huber = self.huber(y_hat, y).mean()
        # 让相关性近似项参与反向传播，促进相关性指标提升
        l_reg_spear = 1 - spearman_surrogate(y_hat, y)  # maximize correlation -> minimize 1 - r
        l_reg = l_reg_huber + self.spearman_weight * l_reg_spear
        l_cls = classification_loss(logits, species, focal=use_focal, class_weight=class_weight, label_smoothing=self.label_smoothing)
        total = l_reg + self.cls_weight * l_cls
        return total, l_reg, l_cls