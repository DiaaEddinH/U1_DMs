import numpy as np
from dataclasses import dataclass, field
from diffusion_models.datasets.datasets import BaseDataset

@dataclass
class U1Dataset(BaseDataset):
    filename: str = field(default=False)

    def __post_init__(self):
        data = np.load(self.filename)
        self.data = data.astype(np.float32)
        super().__post_init__()