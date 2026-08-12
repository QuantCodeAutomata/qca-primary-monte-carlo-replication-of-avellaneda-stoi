"""Main experiment runner for the Avellaneda-Stoikov Monte Carlo replication.

Runs both Experiment 1 (full time-varying spread) and
Experiment 2 (constant spread / table-faithful) and saves all results.

Usage:
    python -m exp.main
    # or
    python exp/main.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import SimParams
from exp.run_exp1 import run_experiment_1, save_results_exp1
from exp.run_exp2 import run_experiment_2, save_results_exp2
from exp.plotting import generate_all_plots

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)


def run_all_experiments(params: SimParams = None) -> None:
    """Run both experiments and save all results and plots."""
    if params is None:
        params = SimParams()

    print('=' * 60)
    print('AVELLANEDA-STOIKOV MONTE CARLO REPLICATION')
    print('=' * 60)
    print(f'Parameters: S0={params.S0}, T={params.T}, sigma={params.sigma}')
    print(f'            dt={params.dt}, N={params.N}')
    print(f'            A={params.A}, k={params.k}')
    print(f'            n_paths={params.n_paths}, seed={params.master_seed}')
    print(f'            gammas={params.gammas}')

    # ---- Experiment 1: Full time-varying spread ----
    print('\n' + '=' * 60)
    print('EXPERIMENT 1: Full Time-Varying Spread')
    print('=' * 60)
    all1, summary1, rep_paths1 = run_experiment_1(params=params, verbose=True)
    save_results_exp1(all1, summary1)
    generate_all_plots(all1, summary1, rep_paths1, exp_label='exp1', params=params)

    # ---- Experiment 2: Constant spread ----
    print('\n' + '=' * 60)
    print('EXPERIMENT 2: Constant Spread (Table-Faithful)')
    print('=' * 60)
    all2, summary2, rep_paths2 = run_experiment_2(params=params, verbose=True)
    save_results_exp2(all2, summary2)
    generate_all_plots(all2, summary2, rep_paths2, exp_label='exp2', params=params)

    # ---- Comparative analysis ----
    print('\n' + '=' * 60)
    print('COMPARATIVE ANALYSIS: Exp1 vs Exp2')
    print('=' * 60)
    compare_experiments(summary1, summary2, params)

    # ---- Save RESULTS.md ----
    write_results_md(summary1, summary2, params)
    print('\nAll experiments complete. Results in results/')


def compare_experiments(summary1: pd.DataFrame, summary2: pd.DataFrame, params: SimParams) -> None:
    """Compare Experiment 1 and Experiment 2 summary statistics."""
    print('\n--- Experiment 1 (Full Spread) ---')
    print(summary1[['gamma', 'strategy', 'initial_full_spread', 'constant_term',
                     'mean_profit', 'std_profit', 'std_final_q']].to_string(index=False))
    print('\n--- Experiment 2 (Constant Spread) ---')
    print(summary2[['gamma', 'strategy', 'constant_spread', 'half_spread',
                     'mean_profit', 'std_profit', 'std_final_q']].to_string(index=False))

    # Directional validation: inventory strategy should have lower std_profit
    for gamma in params.gammas:
        for exp_label, summary in [('Exp1', summary1), ('Exp2', summary2)]:
            inv_cols = summary[summary['strategy'].str.contains('inventory')]
            sym_cols = summary[summary['strategy'].str.contains('symmetric')]
            inv_row = inv_cols[inv_cols['gamma'] == gamma]
            sym_row = sym_cols[sym_cols['gamma'] == gamma]
            if len(inv_row) > 0 and len(sym_row) > 0:
                inv_std = inv_row['std_profit'].values[0]
                sym_std = sym_row['std_profit'].values[0]
                inv_std_q = inv_row['std_final_q'].values[0]
                sym_std_q = sym_row['std_final_q'].values[0]
                status_p = 'PASS' if inv_std < sym_std else 'MARGINAL'
                status_q = 'PASS' if inv_std_q < sym_std_q else 'MARGINAL'
                print(f'  [{exp_label}] gamma={gamma}: '
                      f'std(P&L): inv={inv_std:.4f} vs sym={sym_std:.4f} [{status_p}]; '
                      f'std(q): inv={inv_std_q:.4f} vs sym={sym_std_q:.4f} [{status_q}]')


def write_results_md(summary1: pd.DataFrame, summary2: pd.DataFrame, params: SimParams) -> None:
    """Write a RESULTS.md summary file."""
    lines = [
        '# Avellaneda-Stoikov Monte Carlo Replication — Results',
        '',
        '## Simulation Parameters',
        '',
        f'- S0 = {params.S0}, T = {params.T}, sigma = {params.sigma}',
        f'- dt = {params.dt}, N = {params.N}',
        f'- A = {params.A}, k = {params.k}',
        f'- n_paths = {params.n_paths}, master_seed = {params.master_seed}',
        f'- gammas = {params.gammas}',
        '',
        '## Experiment 1: Full Time-Varying Spread',
        '',
        '### Research Hypotheses Tested',
        '1. Inventory-aware quoting reduces std(Profit) vs symmetric benchmark',
        '2. Inventory-aware quoting reduces std(Final q) vs symmetric benchmark',
        '3. At low gamma (0.01), strategies converge behaviorally',
        '4. At high gamma (0.5), stronger profit-risk trade-off',
        '',
        '### Summary Table (Experiment 1)',
        '',
        summary1[['gamma', 'strategy', 'mean_profit', 'std_profit',
                   'mean_final_q', 'std_final_q']].to_markdown(index=False),
        '',
        '## Experiment 2: Constant Spread (Table-Faithful)',
        '',
        '### Summary Table (Experiment 2)',
        '',
        summary2[['gamma', 'strategy', 'constant_spread', 'mean_profit',
                   'std_profit', 'mean_final_q', 'std_final_q']].to_markdown(index=False),
        '',
        '## Key Findings',
        '',
    ]

    # Add directional findings
    for exp_label, summary in [('Experiment 1', summary1), ('Experiment 2', summary2)]:
        lines.append(f'### {exp_label}')
        for gamma in params.gammas:
            inv_row = summary[(summary['strategy'].str.contains('inventory')) & (summary['gamma'] == gamma)]
            sym_row = summary[(summary['strategy'].str.contains('symmetric')) & (summary['gamma'] == gamma)]
            if len(inv_row) > 0 and len(sym_row) > 0:
                inv_std = inv_row['std_profit'].values[0]
                sym_std = sym_row['std_profit'].values[0]
                inv_std_q = inv_row['std_final_q'].values[0]
                sym_std_q = sym_row['std_final_q'].values[0]
                direction = 'lower' if inv_std < sym_std else 'higher'
                direction_q = 'lower' if inv_std_q < sym_std_q else 'higher'
                lines.append(f'- gamma={gamma}: Inventory strategy std(Profit)={inv_std:.4f} '
                              f'({direction} than symmetric={sym_std:.4f}); '
                              f'std(q)={inv_std_q:.4f} ({direction_q} than {sym_std_q:.4f})')
        lines.append('')

    lines += [
        '## Figures Generated',
        '',
        '### Experiment 1',
        '- `exp1_path_gamma*_inventory_full.png`: Representative path plots',
        '- `exp1_path_gamma*_symmetric_full.png`: Representative path plots',
        '- `exp1_histograms_gamma*.png`: Terminal profit/inventory histograms',
        '- `exp1_summary_comparison.png`: Bar charts of summary statistics',
        '',
        '### Experiment 2',
        '- `exp2_path_gamma*_inventory_const.png`: Representative path plots',
        '- `exp2_path_gamma*_symmetric_const.png`: Representative path plots',
        '- `exp2_histograms_gamma*.png`: Terminal profit/inventory histograms',
        '- `exp2_summary_comparison.png`: Bar charts of summary statistics',
    ]

    results_md_path = RESULTS_DIR / 'RESULTS.md'
    with open(results_md_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Saved {results_md_path}')


if __name__ == '__main__':
    run_all_experiments()
