"""Simulation parameters for the Avellaneda-Stoikov market-making replication."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class SimParams:
    """All simulation parameters from the paper."""
    # Market
    S0: float = 100.0          # Initial mid-price
    T: float = 1.0             # Time horizon
    sigma: float = 2.0         # Mid-price volatility
    dt: float = 0.005          # Discrete time step

    # Order-arrival intensity (exponential model)
    A: float = 140.0           # Baseline intensity
    k: float = 1.5             # Intensity decay per unit of distance

    # Initial state
    q0: int = 0                # Initial inventory
    X0: float = 0.0            # Initial cash

    # Monte Carlo
    n_paths: int = 1000        # Paths per scenario
    master_seed: int = 42      # Global random seed

    # Scenarios
    gammas: List[float] = field(default_factory=lambda: [0.01, 0.1, 0.5])

    @property
    def N(self) -> int:
        """Number of time steps."""
        return round(self.T / self.dt)

    def time_grid(self):
        """Return array of times [0, dt, 2dt, ..., T-dt]."""
        import numpy as np
        return np.arange(self.N) * self.dt


# Default parameter set used throughout all experiments
DEFAULT_PARAMS = SimParams()
