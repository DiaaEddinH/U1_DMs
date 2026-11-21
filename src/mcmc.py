import yaml
import json
import numpy as np

from abc import ABC, abstractmethod

from utils import wrap
from pathlib import Path
from numpy.typing import NDArray
from collections.abc import Iterable
from dataclasses import dataclass, asdict

@dataclass
class ConfigMetadata:
    lattice_size: int = 8
    lattice_dimension: int = 3
    therm_sweeps: int = 1_000
    measure_sweeps: int = 1_000
    overrelax_sweeps: int = 5
    skip_sweeps: int = 5
    seed: int = None
    samples_dir: Path = Path("../data/samples")

    name: str = "run_xxx"
    betas: Iterable[float] | None = None


    def __post_init__(self):
        self.shape = self.lattice_dimension * (self.lattice_size,)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.betas = np.asarray(self.betas)


    @classmethod
    def from_yaml(cls, filepath: str | Path):
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        beta_cfg = data.pop("beta")
        betas = np.linspace(beta_cfg["start"], beta_cfg["stop"], beta_cfg["num"])

        if "samples_dir" in data:
            data["samples_dir"] = Path(data["samples_dir"])

        obj = cls(**data)
        obj.betas = betas
        return obj

    def init_experiment(self):
        exp_dir = self.samples_dir / self.name
        exp_dir.mkdir(parents=True, exist_ok=True)

        self.exp_dir = exp_dir
        meta_dict = asdict(self)
        if isinstance(self.betas, np.ndarray):
            meta_dict["betas"] = self.betas.tolist()
        meta_dict["samples_dir"] = str(meta_dict["samples_dir"])

        with open(exp_dir / "metadata.json", "w") as f:
            json.dump(meta_dict, f, indent=4)

        return exp_dir


@dataclass
class BaseState:
    shape: tuple[int]

    @classmethod
    def from_metadata(cls, meta):
        return cls(shape=meta.shape)

    def __post_init__(self):
        self.dim = len(self.shape)
        self.coords = np.indices(self.shape)
        self.parity = sum(self.coords) % 2
        self.get_initial_state()

    def nearest_neighbours(self, site: tuple[int]) -> Iterable[tuple[int]]:
        assert len(site) == self.dim
        for direction in range(self.dim):
            forward = list(site)
            backward = list(site)

            forward[direction] = (site[direction] + 1) % self.shape[direction]
            backward[direction] = (site[direction] - 1) % self.shape[direction]

            yield tuple(forward)
            yield tuple(backward)

    def get_initial_state(self) -> None:
        self.phi = np.zeros(self.shape)

    @abstractmethod
    def local_field(self):
        pass


class U1GaugeState(BaseState):
    def __post_init__(self):
        self.link_shape = (len(self.shape), *self.shape)
        super().__post_init__()

    def get_initial_state(self):
        self.phi = np.zeros(self.link_shape)

    def shift(self, field: NDArray, mu: int, sign: int = +1):
        return np.roll(field, -sign, axis=mu)

    def staple_sum(self, mu: int) -> tuple[NDArray, ...]:
        phi_mu = self.phi[mu]

        staple_real = np.zeros_like(phi_mu)
        staple_imag = np.zeros_like(phi_mu)

        for nu in range(self.dim):
            if nu == mu:
                continue

            phi_nu = self.phi[nu]

            phi_mu_pnu = self.shift(phi_mu, nu, +1)
            phi_nu_pmu = self.shift(phi_nu, mu, +1)

            phi_nu_mnu = self.shift(phi_nu, nu, -1)
            phi_mu_mnu = self.shift(phi_mu, nu, -1)

            phi_nu_pmu_mnu = self.shift(phi_nu_mnu, mu, +1)

            fwd_staple = phi_nu + phi_mu_pnu - phi_nu_pmu
            bwd_staple = -phi_nu_mnu + phi_mu_mnu + phi_nu_pmu_mnu

            staple_real += np.cos(fwd_staple) + np.cos(bwd_staple)
            staple_imag += np.sin(fwd_staple) + np.sin(bwd_staple)

        return staple_real, staple_imag


class UpdateScheme(ABC):
    """
    Abstract class for MCMC update schemes.
    """    
    def __init__(self, rng_seed: int = 4811):
        self.rng = np.random.default_rng(seed=rng_seed)

    @abstractmethod
    def __call__(self, state: BaseState):
        """
        Abstract method of updating the state of a lattice model

        :param state: Lattice model state
        :type state: BaseState
        """

class U1HeatbathUpdate(UpdateScheme):
    """
    Heatbath update for U(1) lattice gauge systems.
    """    
    def __init__(self, beta: float, rng_seed: int =4811):
        super().__init__(rng_seed)
        self.beta = beta

    def __call__(self, state: U1GaugeState):
        """
        Heatbath update of :param:`state` using a checkerboard setup.

        :param state: Lattice model state
        :type state: U1GaugeState
        """        
        for mu in range(state.dim):
            for parity in (0, 1):
                # Checkerboard mask
                mask = (state.parity % 2) == parity

                A_re, A_im = state.staple_sum(mu)
                A_mag = np.sqrt(A_re**2 + A_im**2)
                A_angle = np.arctan2(A_im, A_re)

                new_phi = self.rng.vonmises(A_angle, self.beta * A_mag)
                state.phi[mu] = np.where(mask, wrap(new_phi), state.phi[mu])


class U1Overrelaxation(UpdateScheme):
    """
    Overrelaxation microcanonial update for U(1) lattice gauge systems.
    """    
    def __init__(self, or_sweeps: int, rng_seed: int = None):
        super().__init__(rng_seed)
        self.or_sweeps = or_sweeps

    def __call__(self, state: U1GaugeState):
        """
        Overrelaxation update of :param:`state` using a checkerboard setup.

        :param state: Lattice model state
        :type state: U1GaugeState
        """       
        for _ in range(self.or_sweeps):
            for mu in range(state.dim):
                for colour in (0, 1):
                    # Checkerboard mask
                    mask = (state.parity % 2 == colour)

                    A_re, A_im = state.staple_sum(mu)
                    A_angle = np.arctan2(A_im, A_re)
                    reflected = wrap(2.0 * A_angle - state.phi[mu])
                    state.phi[mu][mask] = reflected[mask]