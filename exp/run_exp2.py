"""Experiment 2 runner: Constant spread Monte Carlo simulation.

Table-faithful replication of Avellaneda & Stoikov (2008) using
constant spread = (2/gamma)*ln(1 + gamma/k).

Compares Strategy A (inventory-centered constant spread) against
Strategy B (symmetric constant spread) for gamma in {0.01, 0.1, 0.5}.

This experiment tests whether the constant-spread interpretation better
matches the paper's reported table values.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import SimParams
from src.simulator import simulate_monte_carlo
from src.metrics import compute_summary_stats, print_diagnostics, compute_constant_spread, validate_lambda_monotonic
from exp.strategies_exp2 import inventory_const_spread, symmetric_const_spread, compute_half_spread

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

# Expected paper table spread values for verification reference
# (approximate values from Avellaneda-Stoikov reported tables)
PAPER_TABLE_SPREADS = {0.01: 1.33, 0.1: 1.33, 0.5: 1.15}


def run_experiment_2(params: SimParams = None, verbose: bool = True) -> tuple:
    """Run the full Experiment 2 Monte Carlo simulation.

    Parameters
    ----------
    params : SimParams, optional  Uses DEFAULT_PARAMS if None.
    verbose : bool  Print progress if True.

    Returns
    -------
    all_results : pd.DataFrame  Path-level results.
    summary : pd.DataFrame  Summary statistics.
    rep_paths : dict  Representative paths keyed by (gamma, strategy).
    """
    if params is None:
        params = SimParams()

    all_results = []
    rep_paths = {}

    strategies = [
        ('inventory_const', inventory_const_spread),
        ('symmetric_const', symmetric_const_spread),
    ]

    for gamma_idx, gamma in enumerate(params.gammas):
        const_spread = compute_constant_spread(gamma, params.k)
        h = compute_half_spread(gamma, params.k)

        if verbose:
            print(f'\n--- gamma = {gamma} ---')
            print(f'  Constant spread (2/gamma)*ln(1+gamma/k): {const_spread:.4f}')
            print(f'  Half-spread h: {h:.6f}')
            paper_val = PAPER_TABLE_SPREADS.get(gamma)
            if paper_val:
                print(f'  Paper table reference spread: ~{paper_val:.2f}')

        for strat_name, strat_fn in strategies:
            # Deterministic seed: offset by 200 to differ from Exp 1 seeds
            seed = params.master_seed + 200 + gamma_idx * 100 + (0 if 'inventory' in strat_name else 1)

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

    # Add constant spread column
    summary_df['constant_spread'] = summary_df['gamma'].apply(
        lambda g: compute_constant_spread(g, params.k)
    )
    summary_df['half_spread'] = summary_df['gamma'].apply(
        lambda g: compute_half_spread(g, params.k)
    )

    if verbose:
        print('\n=== EXPERIMENT 2 SUMMARY ===')
        print_diagnostics(all_results_df, summary_df, params)
        print('\nConstant spread comparison with paper table references:')
        for gamma in params.gammas:
            cs = compute_constant_spread(gamma, params.k)
            paper_val = PAPER_TABLE_SPREADS.get(gamma, 'N/A')
            print(f'  gamma={gamma}: computed={cs:.4f}, paper~{paper_val}')

    # Validation
    for gamma in params.gammas:
        assert validate_lambda_monotonic(params, gamma), f'lambda not monotonic for gamma={gamma}'

    return all_results_df, summary_df, rep_paths


def save_results_exp2(all_results_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Save Experiment 2 results to CSV files."""
    all_results_df.to_csv(RESULTS_DIR / 'exp2_all_paths.csv', index=False)
    summary_df.to_csv(RESULTS_DIR / 'exp2_summary.csv', index=False)
    print(f'Saved exp2 results to {RESULTS_DIR}')


if __name__ == '__main__':
    all_results_df, summary_df, rep_paths = run_experiment_2(verbose=True)
    save_results_exp2(all_results_df, summary_df)
    print('\nSummary table:')
    print(summary_df.to_string(index=False))
