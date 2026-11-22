"""Metoda de CI de la CP3: un U-Net pentru segmentarea alunecărilor pe patch-uri de 128 x 128.

Spre deosebire de RF, care clasifică fiecare pixel separat, U-Net-ul vede contextul spațial
al patch-ului (forma cicatricii, vecinătatea, versantul). Arhitectura e cea clasică
(Ronneberger et al., 2015), redusă ca să încapă pe un GPU de 4 GB:

    encoder: 4 niveluri de câte 2 convoluții 3x3 + BatchNorm + ReLU, cu max-pooling 2x2
    decoder: up-sampling prin convoluție transpusă + conexiuni skip din encoder
    ieșire:  un logit pe pixel (alunecare vs. restul)

La 128 x 128 și 4 niveluri, stratul cel mai adânc are 8 x 8 pixeli (~10 km de context).

Loss-ul: BCE + Dice, ambele doar pe pixelii valizi. BCE învață probabilități pe pixel, iar
Dice compensează dezechilibrul (~2% alunecare) optimizând direct suprapunerea, apropiată de F1.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, c_in: int, c_out: int):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, in_channels: int, base: int = 32, depth: int = 4):
        super().__init__()
        widths = [base * 2 ** i for i in range(depth + 1)]  # ex. 32, 64, 128, 256, 512
        self.down = nn.ModuleList([DoubleConv(in_channels, widths[0])])
        self.down.extend(DoubleConv(widths[i], widths[i + 1]) for i in range(depth))
        self.up = nn.ModuleList(nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2) for i in reversed(range(depth)))
        self.merge = nn.ModuleList(DoubleConv(2 * widths[i], widths[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, merge in zip(self.up, self.merge):
            x = merge(torch.cat([skips.pop(), up(x)], dim=1))
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def masked_bce_dice(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                    dice_weight: float = 1.0, eps: float = 1.0) -> torch.Tensor:
    """BCE medie pe pixelii valizi + (1 - Dice moale) pe batch; pixelii invalizi nu contează deloc."""
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
    prob = torch.sigmoid(logits) * valid
    t = target * valid
    dice = (2 * (prob * t).sum() + eps) / (prob.sum() + t.sum() + eps)
    return bce + dice_weight * (1 - dice)
