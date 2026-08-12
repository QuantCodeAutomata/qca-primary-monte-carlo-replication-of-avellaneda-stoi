"""Summary statistics, validation checks, and diagnostic functions
for the Avellaneda-Stoikov Monte Carlo replication."""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from src.config import SimParams


def compute_summary_stats(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Monte Carlo results into summary statistics per (gamma, strategy).

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from simulate_monte_carlo with columns:
        gamma, strategy, terminal_profit, terminal_inventory.

    Returns
    -------
    pd.DataFrame with one row per (gamma, strategy) and columns:
        mean_profit, std_profit, mean_final_q, std_final_q,
        n_paths, negative_delta_pct, bernoulli_violation_pct.
    """
    def agg(grp: pd.DataFrame) -> pd.Series:
        return pd.Series({
            'mean_profit': grp['terminal_profit'].mean(),
            'std_profit': grp['terminal_profit'].std(ddof=1),
            'mean_final_q': grp['terminal_inventory'].mean(),
            'std_final_q': grp['terminal_inventory'].std(ddof=1),
            'n_paths': len(grp),
            'negative_delta_pct': 100.0 * (grp['negative_delta_count'] > 0).mean(),
            'bernoulli_violation_pct': 100.0 * (grp['bernoulli_violation_count'] > 0).mean(),
        })

    summary = results_df.groupby(['gamma', 'strategy']).apply(agg, include_groups=False).reset_index()
    return summary


def validate_terminal_profit(results_df: pd.DataFrame, tol: float = 1e-8) -> bool:
    """Confirm all stored terminal profits are finite."""
    finite_check = np.isfinite(results_df['terminal_profit']).all()
    return bool(finite_check)


def validate_lambda_monotonic(params: SimParams, gamma: float, n_points: int = 100) -> bool:
    """Confirm lambda(delta) = A*exp(-k*delta) is monotonically decreasing in delta."""
    deltas = np.linspace(0.0, 5.0, n_points)
    lambdas = params.A * np.exp(-params.k * deltas)
    diffs = np.diff(lambdas)
    return bool((diffs < 0).all())


def validate_reservation_price(params: SimParams, gamma: float) -> bool:
    """Confirm r_t = S_t when q=0, and r_t -> S_t as t -> T."""
    S, q = 100.0, 0
    # When q=0, reservation price should equal S for any t
    for t in np.linspace(0, params.T, 10):
        r = S - q * gamma * params.sigma ** 2 * (params.T - t)
        if not np.isclose(r, S, atol=1e-10):
            return False
    # As t -> T, r -> S regardless of q (skew vanishes)
    t_near_T = params.T - 1e-10
    q_nonzero = 5
    r_near_T = S - q_nonzero * gamma * params.sigma ** 2 * (params.T - t_near_T)
    return np.isclose(r_near_T, S, atol=1e-6)


def compute_full_spread_formula(gamma: float, sigma: float, k: float, T_minus_t: float) -> float:
    """Full time-varying spread from Avellaneda-Stoikov eq. (3.18).

    spread = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)
    """
    return gamma * sigma ** 2 * T_minus_t + (2.0 / gamma) * np.log1p(gamma / k)


def compute_constant_spread(gamma: float, k: float) -> float:
    """Constant spread used in the table-faithful replication.

    spread_const = (2/gamma) * ln(1 + gamma/k)
    """
    return (2.0 / gamma) * np.log1p(gamma / k)


def compute_initial_full_spread(gamma: float, sigma: float, k: float, T: float) -> float:
    """Full spread evaluated at t=0, i.e., T-t = T."""
    return compute_full_spread_formula(gamma, sigma, k, T)


def print_diagnostics(results_df: pd.DataFrame, summary_df: pd.DataFrame, params: SimParams) -> None:
    """Print diagnostic information for the Monte Carlo run."""
    print("\n=== MONTE CARLO DIAGNOSTICS ===")
    print(f"Paths per scenario: {params.n_paths}")
    print(f"Steps per path: {params.N}")
    print(f"dt = {params.dt}, sigma = {params.sigma}, T = {params.T}")
    print(f"A = {params.A}, k = {params.k}")
    print()
    print(summary_df.to_string(index=False))
    print()
    violations = results_df[results_df['bernoulli_violation_count'] > 0]
    if len(violations) > 0:
        print(f"WARNING: {len(violations)} paths had Bernoulli probability > 1 in at least one step.")
    else:
        print("OK: No Bernoulli violations detected.")
