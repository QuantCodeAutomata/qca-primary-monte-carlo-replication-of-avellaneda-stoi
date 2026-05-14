"""
Synthetic Data Generation for Avellaneda-Stoikov Replication.

This experiment uses only synthetic simulated data (no external historical data).
This script generates and saves reference synthetic mid-price paths for
reproducibility and debugging purposes.

The paper requires:
  - Arithmetic Brownian motion: s_{t+dt} = s_t ± sigma*sqrt(dt) with equal probability
  - Parameters: s0=100, sigma=2, T=1, dt=0.005 (200 steps)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_abm_paths(
    s0: float = 100.0,
    sigma: float = 2.0,
    T: float = 1.0,
    dt: float = 0.005,
    n_paths: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate arithmetic Brownian motion paths using the paper's binomial approximation.

    At each step: s_{t+dt} = s_t ± sigma*sqrt(dt) with equal probability.

    Parameters
    ----------
    s0      : initial mid-price
    sigma   : volatility
    T       : horizon
    dt      : time step
    n_paths : number of paths
    seed    : random seed

    Returns
    -------
    pd.DataFrame of shape (n_steps+1, n_paths) with time index
    """
    rng = np.random.default_rng(seed)
    n_steps = round(T / dt)
    time_index = np.round(np.arange(n_steps + 1) * dt, 6)

    paths = np.zeros((n_steps + 1, n_paths))
    paths[0, :] = s0

    for i in range(n_steps):
        moves = np.where(rng.random(n_paths) < 0.5, 1, -1)
        paths[i + 1, :] = paths[i, :] + sigma * np.sqrt(dt) * moves

    df = pd.DataFrame(
        paths,
        index=time_index,
        columns=[f"path_{j}" for j in range(n_paths)],
    )
    df.index.name = "time"
    return df


def save_reference_paths(n_paths: int = 100) -> None:
    """
    Save a small set of reference paths for debugging.

    Parameters
    ----------
    n_paths : number of paths to save (default 100 for file size)
    """
    df = generate_abm_paths(n_paths=n_paths, seed=42)
    out_path = os.path.join(DATA_DIR, "reference_abm_paths.csv")
    df.to_csv(out_path)
    print(f"Saved {n_paths} reference ABM paths to {out_path}")
    return df


if __name__ == "__main__":
    print("Generating synthetic ABM reference paths...")
    df = save_reference_paths(n_paths=100)
    print(f"Shape: {df.shape}")
    print(f"First 5 rows:\n{df.iloc[:5, :3]}")
    print("Done.")
