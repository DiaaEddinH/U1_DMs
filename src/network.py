import torch
from utils import wrap
from torch import Tensor
from typing import Any
from math import log10, pi

from diffusion_models.models.models import ScoreModel
from diffusion_models.noise.noise_scheduler import Schedule

def wrapped_target(delta, std, K=3):
    ks = torch.arange(-K, K + 1, device=delta.device, dtype=delta.dtype)
    view = (1, len(ks)) + (1,) * (delta.dim() - 1)
    ks = ks.view(view)

    shifts = delta.unsqueeze(1) + 2 * torch.pi * ks
    var = std.unsqueeze(1) ** 2
    weights = torch.exp(-0.5 * shifts**2 / var)

    numerator = (weights * shifts).sum(dim=1)
    denominator = weights.sum(dim=1)
    return numerator / denominator

class WilsonScoreModel(ScoreModel):
    def __init__(
        self,
        network: torch.nn.Module,
        schedule: Schedule,
        device: str | torch.device | None = None,
        decay_rate: float = 0.999,
        **kwargs: Any,
    ):
        super().__init__(network, schedule, device, decay_rate, **kwargs)

    def plaquette(self, inputs: Tensor) -> Tensor:
        """Calculate θ_p(x) = θ_μ(x) + θ_ν(x+μ) - θ_μ(x+ν) - θ_ν(x)"""
        theta_0, theta_1 = inputs[:, 0], inputs[:, 1]
        plaq = (
            theta_0
            + torch.roll(theta_1, shifts=-1, dims=1)
            - torch.roll(theta_0, shifts=-1, dims=2)
            - theta_1
        )
        return wrap(plaq).unsqueeze(1)

    def plaquette_to_link_force(self, inputs: Tensor):
        F0 = inputs[:, 0] - torch.roll(inputs[:, 0], shifts=1, dims=2)
        F1 = torch.roll(inputs[:, 0], shifts=1, dims=1) - inputs[:, 0]
        return -torch.stack([F0, F1], dim=1)

    def forward(self, x, t, *labels):
        plaq = self.plaquette(x)
        vector = torch.cat([plaq.cos(), plaq.sin()], dim=1)
        plaq_score = super().forward(vector, t, *labels)
        return self.plaquette_to_link_force(plaq_score)

    def loss_fn(
        self, batch: Tensor, n_windings: int = 3, *labels: Tensor | tuple[Tensor, ...]
    ) -> Tensor:
        """
        Evaluate a denoising score-matching loss for angular data.

        :param batch: Clean data batch
        :type batch: Tensor
        :return: Scalar loss
        :rtype: Tensor
        """
        eps = self.schedule.eps
        z = torch.randn_like(batch)

        n = batch.shape[0]
        random_t = 10 ** torch.FloatTensor(n).uniform_(log10(eps), 0).to(self.device)
        mean, std = self.schedule.mean_stddev(batch, random_t)

        perturbed_x = wrap(mean + z * std)

        # Sum over winding numbers k to get the exact wrapped-normal score,
        # instead of only using the nearest periodic image.
        k = torch.arange(-n_windings, n_windings + 1, device=batch.device)
        k = k.reshape((2 * n_windings + 1,) + (1,) * batch.dim())

        diff = (
            perturbed_x.unsqueeze(0) - mean.unsqueeze(0) - 2 * pi * k
        )  # (2K+1, *batch.shape)
        std_ = std.unsqueeze(0)

        log_weights = -(diff**2) / (2 * std_**2)
        weights = torch.softmax(
            log_weights, dim=0
        )  # normalized mixture weights per winding

        # target = -std * score_true(perturbed_x | mean, std), wrapped-normal case
        target = torch.sum(weights * diff, dim=0) / std

        score = self.forward(perturbed_x, random_t, *labels)
        return 0.5 * torch.mean((std * score + target) ** 2)