"""
losses.py
---------
A "loss function" is how we tell the model whether it did a good job or not.
The model tries to make this number as SMALL as possible.

We combine THREE ideas:
1. L1 loss       - "how different are the pixel values, on average?"
2. SSIM loss     - "does the overall structure/shape look similar?"
                    (this matches the exact metric KLA scores you on)
3. Edge loss     - "are the sharp edges/lines preserved?" (semiconductor
                    patterns are full of fine lines, so this matters a lot)

Using all three together stops the model from taking shortcuts, like
producing a blurry-but-average-correct image (which L1 alone might allow).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sobel_edges(img):
    """
    Sobel filters detect edges (sudden changes in brightness) in an image.
    We use this to compare 'edge maps' between prediction and ground truth,
    so the model is pushed to keep lines sharp instead of smoothing them away.
    """
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                            device=img.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                            device=img.device).view(1, 1, 3, 3)
    gx = F.conv2d(img, sobel_x, padding=1)
    gy = F.conv2d(img, sobel_y, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)


def edge_loss(pred, target):
    return F.l1_loss(sobel_edges(pred), sobel_edges(target))


class RestorationLoss(nn.Module):
    def __init__(self, ssim_weight=0.3, edge_weight=0.1):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        # pytorch_msssim is a small pip package that computes SSIM on GPU tensors.
        # If it's not installed, we just skip the SSIM term (still trains fine).
        try:
            from pytorch_msssim import ssim
            self.ssim_fn = ssim
        except ImportError:
            print("[warning] pytorch_msssim not installed - SSIM loss term disabled. "
                  "Run: pip install pytorch-msssim --break-system-packages")
            self.ssim_fn = None

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        loss = l1

        if self.ssim_fn is not None:
            ssim_val = self.ssim_fn(pred, target, data_range=1.0, size_average=True)
            loss = loss + self.ssim_weight * (1 - ssim_val)

        loss = loss + self.edge_weight * edge_loss(pred, target)
        return loss
