"""Optional GAN discriminator for RMD (experimental, ``--gan``).

Cleaned from the original ``utils/gan_model.py``: debug layers and the unused
2D branch were removed. The 3D Conv classifier is kept, with the same
classification / generator losses.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn


class GAN(nn.Module):
    def __init__(self, mode: str = "3d"):
        super().__init__()
        if mode not in ("3d",):
            raise ValueError(f"GAN only supports mode='3d', got {mode!r}")

        self.cls_pred_branch = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=1280, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(num_groups=1, num_channels=1280),
            nn.SiLU(),
            nn.Conv3d(in_channels=1280, out_channels=1280, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.GroupNorm(num_groups=1, num_channels=1280),
            nn.SiLU(),
            nn.Conv3d(in_channels=1280, out_channels=1280, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.GroupNorm(num_groups=1, num_channels=1280),
            nn.SiLU(),
            nn.Conv3d(in_channels=1280, out_channels=1280, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.GroupNorm(num_groups=1, num_channels=1280),
            nn.SiLU(),
            nn.Conv3d(in_channels=1280, out_channels=1280, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.GroupNorm(num_groups=1, num_channels=1280),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=1280, out_features=1),
        )
        self.cls_pred_branch.requires_grad_(True)

    def compute_guidance_clean_cls_loss(self, real_latent, fake_latent):
        pred_realism_on_real = self.compute_cls_logits(real_latent.detach())
        pred_realism_on_fake = self.compute_cls_logits(fake_latent.detach())
        classification_loss = F.softplus(pred_realism_on_fake).mean() + F.softplus(-pred_realism_on_real).mean()
        return classification_loss

    def compute_generator_clean_cls_loss(self, fake_latent):
        pred_realism_on_fake_with_grad = self.compute_cls_logits(fake_latent)
        gen_cls_loss = F.softplus(-pred_realism_on_fake_with_grad).mean()
        return gen_cls_loss

    def compute_cls_logits(self, latent):
        logits = self.cls_pred_branch(latent)
        return logits.squeeze(dim=1)
