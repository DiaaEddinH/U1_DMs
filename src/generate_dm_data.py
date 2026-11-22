import torch
import numpy as np

from torch import Tensor
from pathlib import Path
from network import WilsonScoreModel
from diffusion_models.utils import detach_to_numpy
from diffusion_models.networks.networks import CNet
from diffusion_models.config.config_cli import parse_config
from diffusion_models.noise.noise_scheduler import GeometricSchedule
from diffusion_models.sampling.samplers import (
    AngularEMSamplerWRescaling,
    AngularSHeunSamplerWRescaling,
    AngularMAALASamplerWRescaling,
    AngularPoorManHeunSamplerWRescaling,
    MAJDSampler
)
from diffusion_models.config.config import (
    NETWORK_REGISTRY,
    SAMPLER_REGISTRY,
    SCHEDULE_REGISTRY,
    build_sampler
)

NETWORK_REGISTRY.register("cnet")(CNet)
SCHEDULE_REGISTRY.register("geometric")(GeometricSchedule)

SAMPLER_REGISTRY.register("euler")(AngularEMSamplerWRescaling)
SAMPLER_REGISTRY.register("heun")(AngularSHeunSamplerWRescaling)
SAMPLER_REGISTRY.register("maala")(AngularMAALASamplerWRescaling)
SAMPLER_REGISTRY.register("poorman")(AngularPoorManHeunSamplerWRescaling)
SAMPLER_REGISTRY.register("majd")(MAJDSampler)


class U1LatticeAction:
    def __init__(self, beta: float) -> None:
        """Lattice U(1) gauge theory in 2D."""
        self.beta = beta
    
    def plaquette(self, theta: Tensor, mu=0, nu=1) -> Tensor:
        """Calculate θ_p(x) = θ_μ(x) + θ_ν(x+μ) - θ_μ(x+ν) - θ_ν(x) """
        return theta[:, mu] + torch.roll(theta[:, nu], shifts=-1, dims=mu+1) - torch.roll(theta[:, mu], shifts=-1, dims=nu+1) - theta[:, nu]

    def __call__(self, theta: Tensor) -> Tensor:
        """Compute the action S[U] = -β ReTrU = -β cosθ, -π ≤ θ ≤ π"""
        return -self.beta * torch.sum(torch.cos(self.plaquette(theta, 1, 0)), axis=(-2, -1))

if __name__ == "__main__":
    """
    Steps of generation code:

        1. Initialise model from config file and load weights
        2. Run sampler using the parameters from the config file.
        3. Save to a file path using config.trainer.file_path and extras.sample_dir
        4. Filename passed through extras.savefile, or --savefile
    """

    config = parse_config()
    extras = config.extra

    train_beta = extras["train_beta"]
    train_lattice = extras["train_lattice"]
    rescaling_factor = config.sampler.params["rescaling_factor"]

    target_beta = train_beta * rescaling_factor

    sample_dir = Path(extras["sample_dir"]) / config.trainer.file_path
    sample_dir.mkdir(parents=True, exist_ok=True)

    model = WilsonScoreModel.from_config(config, load_weights=True)

    if config.sampler.name in ["maala", "poorman", "majd"]:
        action = U1LatticeAction(beta=target_beta)
        cls = SAMPLER_REGISTRY.get(config.sampler.name)
        sampler = cls(model, action)
    else:
        sampler = build_sampler(model, config.sampler)
    
    if hasattr(sampler, "rescaling_factor"):
        sampler.rescaling_factor = rescaling_factor

    print(f"Built {sampler} @ {target_beta:.3f}")
    with torch.no_grad():
        with model.exponential_moving_average.average_parameters():
            samples = detach_to_numpy(sampler.sample(**config.sampler.params))

    savefile = sample_dir / extras["savefile"]
    print(config.sampler.params)
    print(f"Saving data of shape {samples.shape} to {savefile}.")
    np.save(savefile, arr=samples)
