import torch
import torch.nn as nn


def dice_score(logits, target, eps=1e-6):
    p = (torch.sigmoid(logits) > 0.5).float()
    inter = (p * target).sum((1, 2, 3))
    return ((2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)).mean().item()


def iou_score(logits, target, eps=1e-6):
    p = (torch.sigmoid(logits) > 0.5).float()
    inter = (p * target).sum((1, 2, 3))
    union = p.sum((1, 2, 3)) + target.sum((1, 2, 3)) - inter
    return ((inter + eps) / (union + eps)).mean().item()


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target, eps=1e-6):
        p = torch.sigmoid(logits)
        inter = (p * target).sum((1, 2, 3))
        dice = 1 - ((2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)).mean()
        return self.bce(logits, target) + dice
