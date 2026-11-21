import argparse
import numpy as np
from mcmc import (
    U1GaugeState, 
    ConfigMetadata,
    U1HeatbathUpdate,
    U1Overrelaxation
)
from tqdm import tqdm
from pathlib import Path

def run_simulation(beta: float, state: U1GaugeState, meta: ConfigMetadata):
    """
    Sequentially run a list of Updates to generate MCMC data for U(1) lattice gauge theory.

    :param beta: Inverse coupling of theory
    :type beta: float
    :param state: Lattice model state
    :type state: U1GaugeState
    :param meta: Simulation metadata dataclass
    :type meta: ConfigMetadata
    """    
    updates = [
        U1HeatbathUpdate(beta=beta, rng_seed=meta.seed),
        U1Overrelaxation(or_sweeps=meta.overrelax_sweeps, rng_seed=meta.seed),
    ]

    configs = np.zeros((meta.measure_sweeps, *state.link_shape))

    # Thermalisation
    for _ in tqdm(range(meta.therm_sweeps), desc="Equilibrating...."):
        for update in updates:
            update(state)

    # Measurement
    for i in tqdm(range(meta.measure_sweeps), desc="Measuring..."):
        for _ in range(meta.skip_sweeps):
            for update in updates:
                update(state)

        configs[i] = state.phi

    return configs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    meta = ConfigMetadata.from_yaml(args.config)

    exp_dir = meta.init_experiment()

    for beta in meta.betas:
        print(f"Running U1 simulation for β = {beta:.3f} @ L = {meta.lattice_size:g}")
        state = U1GaugeState.from_metadata(meta)
        configs = run_simulation(beta, state, meta)

        np.save(exp_dir / f"beta_{beta:.3f}_L{meta.lattice_size:g}.npy", configs)