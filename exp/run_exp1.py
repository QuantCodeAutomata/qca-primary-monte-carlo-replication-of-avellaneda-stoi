"""Experiment 1 runner: Full time-varying spread Monte Carlo simulation.

Reproduces the main finite-horizon simulation from Avellaneda & Stoikov (2008).
Compares Strategy A (inventory-aware full spread) against Strategy B
(symmetric full spread) for gamma in {0.01, 0.1, 0.5}.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import SimParams
from src.simulator import simulate_monte_carlo
from src.metrics import compute_summary_stats, print_diagnostics, compute_initial_full_spread, validate_lambda_monotonic, validate_reservation_price
from exp.strategies_exp1 import inventory_full_spread, symmetric_full_spread

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)


def run_experiment_1(params: SimParams = None, verbose: bool = True) -> tuple:
    """Run the full Experiment 1 Monte Carlo simulation.

    Parameters
    ----------
    params : SimParams, optional
        Simulation parameters. Uses DEFAULT_PARAMS if None.
    verbose : bool
        Print progress and diagnostics if True.

    Returns
    -------
    all_results : pd.DataFrame
        All path-level results across all gammas and strategies.
    summary : pd.DataFrame
        Aggregated summary statistics.
    rep_paths : dict
        Dictionary mapping (gamma, strategy) to representative path DataFrame.
    """
    if params is None:
        params = SimParams()

    all_results = []
    rep_paths = {}

    strategies = [
        ('inventory_full', inventory_full_spread),
        ('symmetric_full', symmetric_full_spread),
    ]

    for gamma_idx, gamma in enumerate(params.gammas):
        if verbose:
            print(f'\n--- gamma = {gamma} ---')
            spread_t0 = compute_initial_full_spread(gamma, params.sigma, params.k, params.T)
            print(f'  Initial full spread (t=0): {spread_t0:.4f}')
            print(f'  Constant term (2/gamma)*ln(1+gamma/k): {(2/gamma)*np.log1p(gamma/params.k):.4f}')

        for strat_name, strat_fn in strategies:
            # Derive a deterministic seed per (gamma, strategy)
            seed = params.master_seed + gamma_idx * 100 + (0 if 'inventory' in strat_name else 1)

            if verbose:
                print(f'  Running {strat_name} (seed={seed}) ...', end=' ')

            results_df, rep_path_df = simulate_monte_carlo(
                strategy_fn=strat_fn,
                strategy_name=strat_name,
                gamma=gamma,
                params=params,
                seed=seed,
                rep_path_index=0,
            )

            all_results.append(results_df)
            rep_paths[(gamma, strat_name)] = rep_path_df

            if verbose:
                print(f'done. mean_profit={results_df["terminal_profit"].mean():.4f}, '
                      f'std_profit={results_df["terminal_profit"].std():.4f}')

    all_results_df = pd.concat(all_results, ignore_index=True)
    summary_df = compute_summary_stats(all_results_df)

    # Add spread columns to summary
    summary_df['initial_full_spread'] = summary_df['gamma'].apply(
        lambda g: compute_initial_full_spread(g, params.sigma, params.k, params.T)
    )
    summary_df['constant_term'] = summary_df['gamma'].apply(
        lambda g: (2.0 / g) * np.log1p(g / params.k)
    )

    if verbose:
        print('\n=== EXPERIMENT 1 SUMMARY ===')
        print_diagnostics(all_results_df, summary_df, params)

    # Validation checks
    for gamma in params.gammas:
        assert validate_lambda_monotonic(params, gamma), f'lambda not monotonic for gamma={gamma}'
        assert validate_reservation_price(params, gamma), f'reservation price check failed for gamma={gamma}'

    return all_results_df, summary_df, rep_paths


def save_results_exp1(all_results_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Save Experiment 1 results to CSV files."""
    all_results_df.to_csv(RESULTS_DIR / 'exp1_all_paths.csv', index=False)
    summary_df.to_csv(RESULTS_DIR / 'exp1_summary.csv', index=False)
    print(f'Saved exp1 results to {RESULTS_DIR}')


if __name__ == '__main__':
    all_results_df, summary_df, rep_paths = run_experiment_1(verbose=True)
    save_results_exp1(all_results_df, summary_df)
    print('\nSummary table:')
    print(summary_df.to_string(index=False))
