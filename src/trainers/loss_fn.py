import torch
from einops import rearrange
import torch.nn.functional as F


class GeneLoss:
    @staticmethod
    def cross_entropy_loss(logist: torch.Tensor, target: torch.Tensor, weight=1.):
        if logist.ndim > 2:
            logist = rearrange(logist, 'b n c -> (b n) c')
        if target.ndim == 2:
            target = rearrange(target, 'b n -> (b n)')
        return F.cross_entropy(logist, target.long().to(logist.device)) * weight

    @staticmethod
    def focal_loss(logist, target, weight=1., gamma=2.):
        if logist.ndim > 2:
            logist = rearrange(logist, 'b n c -> (b n) c')
        if target.ndim == 2:
            target = rearrange(target, 'b n -> (b n)')

        ce_loss = F.cross_entropy(logist, target.long().to(logist.device))
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** gamma * ce_loss * weight
        return focal_loss
