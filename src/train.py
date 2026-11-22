import torch
import numpy as np

from pathlib import Path
from dataset import U1Dataset
from network import WilsonScoreModel

from diffusion_models.trainer import Trainer
from diffusion_models.networks.networks import CNet
from diffusion_models.utils import ddp_setup, destroy_ddp
from diffusion_models.config.config_cli import parse_config
from torch.utils.data import DataLoader, DistributedSampler
from diffusion_models.datasets.samplers import EpochShuffleSampler
from diffusion_models.noise.noise_scheduler import GeometricSchedule
from diffusion_models.datasets.datasets import MultiResolutionDataset
from diffusion_models.config.config import NETWORK_REGISTRY, SCHEDULE_REGISTRY

NETWORK_REGISTRY.register("cnet")(CNet)
SCHEDULE_REGISTRY.register("geometric")(GeometricSchedule)

weight_dir = Path("./data/weights")
weight_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    config = parse_config()
    extras = config.extra
    use_ddp = config.trainer.use_ddp
    train_beta = extras["train_beta"]
    train_lattice = extras["train_lattice"]
    sample_dir = Path(extras["sample_dir"])

    ddp_setup(use_ddp=use_ddp)

    datasets = [
        U1Dataset(
            filename=sample_dir / 
            config.trainer.file_path / 
            f"beta_{train_beta:.3f}_L{L:g}.npy"
        )
        for L in extras["train_lattice"]
    ]
    multi_dataset = MultiResolutionDataset(datasets=datasets)
    data_sampler = (
        DistributedSampler(dataset=datasets)
        if use_ddp
        else EpochShuffleSampler(multi_dataset)
    )

    loader = DataLoader(
        dataset=multi_dataset,
        batch_size=config.run.batch_size,
        sampler=data_sampler,
        shuffle=(data_sampler is None),
        drop_last=True,
    )

    model = WilsonScoreModel.from_config(config)

    trainer = Trainer.from_config(model, config)

    trainer.train(loader=loader)

    destroy_ddp(use_ddp=use_ddp)