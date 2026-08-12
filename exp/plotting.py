"""Visualization functions for Avellaneda-Stoikov Monte Carlo replication.

All figures use matplotlib with seaborn styling and are saved as PNG.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file saving
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

sns.set_theme(style='whitegrid', palette='tab10')


def plot_representative_path(
    rep_path_df: pd.DataFrame,
    gamma: float,
    strategy_name: str,
    exp_label: str = 'exp1',
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot S_t, reservation price r_t, bid quote p_b, and ask quote p_a over time.

    Parameters
    ----------
    rep_path_df : pd.DataFrame
        Representative path data with columns: time, S, r, p_a, p_b.
    gamma : float  Risk-aversion coefficient.
    strategy_name : str  Strategy identifier for plot title.
    exp_label : str  Experiment label for filename.
    save_path : Path, optional  Override save location.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    t = rep_path_df['time'].values

    # Panel 1: Price paths
    ax = axes[0]
    ax.plot(t, rep_path_df['S'], label='Mid-price $S_t$', color='black', linewidth=1.2)
    if 'r' in rep_path_df.columns:
        ax.plot(t, rep_path_df['r'], label='Reservation price $r_t$',
                color='blue', linewidth=1.0, linestyle='--')
    ax.plot(t, rep_path_df['p_a'], label='Ask quote $p^a_t$',
            color='red', linewidth=0.8, alpha=0.7)
    ax.plot(t, rep_path_df['p_b'], label='Bid quote $p^b_t$',
            color='green', linewidth=0.8, alpha=0.7)
    ax.set_ylabel('Price')
    ax.set_title(f'Representative Path — {strategy_name} (gamma={gamma})')
    ax.legend(fontsize=8)

    # Panel 2: Inventory
    ax = axes[1]
    ax.step(t, rep_path_df['q'], where='post', color='purple', linewidth=1.0)
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
    ax.set_ylabel('Inventory $q_t$')
    ax.set_title('Inventory over Time')

    # Panel 3: Quote distances
    ax = axes[2]
    ax.plot(t, rep_path_df['delta_a'], label='$\\delta^a_t$', color='red', linewidth=0.8)
    ax.plot(t, rep_path_df['delta_b'], label='$\\delta^b_t$', color='green', linewidth=0.8)
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
    ax.set_ylabel('Quote Distance')
    ax.set_xlabel('Time $t$')
    ax.set_title('Quote Distances from Mid-Price')
    ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path is None:
        save_path = RESULTS_DIR / f'{exp_label}_path_gamma{gamma}_{strategy_name}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {save_path}')
    return fig


def plot_profit_histograms(
    all_results_df: pd.DataFrame,
    gamma: float,
    exp_label: str = 'exp1',
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot overlaid terminal profit histograms for two strategies at a given gamma.

    Parameters
    ----------
    all_results_df : pd.DataFrame  Path results with columns: gamma, strategy, terminal_profit.
    gamma : float  Which gamma scenario to plot.
    exp_label : str  Experiment label for filename.
    save_path : Path, optional  Override save location.
    """
    data = all_results_df[all_results_df['gamma'] == gamma]
    strategies = data['strategy'].unique()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['steelblue', 'darkorange']

    # Left: terminal profit histogram
    ax = axes[0]
    for strat, color in zip(strategies, colors):
        vals = data[data['strategy'] == strat]['terminal_profit'].values
        ax.hist(vals, bins=50, alpha=0.5, label=strat, color=color, density=True)
        ax.axvline(vals.mean(), color=color, linestyle='--', linewidth=1.5,
                   label=f'{strat} mean={vals.mean():.2f}')
    ax.set_xlabel('Terminal Profit $\\Pi_T$')
    ax.set_ylabel('Density')
    ax.set_title(f'Terminal Profit Distribution (gamma={gamma})')
    ax.legend(fontsize=8)

    # Right: terminal inventory histogram
    ax = axes[1]
    for strat, color in zip(strategies, colors):
        vals = data[data['strategy'] == strat]['terminal_inventory'].values
        ax.hist(vals, bins=range(int(vals.min())-1, int(vals.max())+2),
                alpha=0.5, label=strat, color=color, density=True)
    ax.set_xlabel('Terminal Inventory $q_T$')
    ax.set_ylabel('Density')
    ax.set_title(f'Terminal Inventory Distribution (gamma={gamma})')
    ax.legend(fontsize=8)

    plt.suptitle(f'Monte Carlo Results — gamma={gamma}', fontsize=12)
    plt.tight_layout()
    if save_path is None:
        save_path = RESULTS_DIR / f'{exp_label}_histograms_gamma{gamma}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {save_path}')
    return fig


def plot_summary_comparison(
    summary_df: pd.DataFrame,
    exp_label: str = 'exp1',
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot summary statistics comparison across gamma values for both strategies.

    Creates a 2x2 grid: mean profit, std profit, mean final q, std final q.
    """
    gammas = sorted(summary_df['gamma'].unique())
    strategies = summary_df['strategy'].unique()
    metrics = ['mean_profit', 'std_profit', 'mean_final_q', 'std_final_q']
    titles = [
        'Mean Terminal Profit',
        'Std Terminal Profit',
        'Mean Terminal Inventory',
        'Std Terminal Inventory',
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = ['steelblue', 'darkorange', 'green']
    x = np.arange(len(gammas))
    width = 0.35

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        for j, strat in enumerate(strategies):
            vals = [summary_df[(summary_df['gamma'] == g) & (summary_df['strategy'] == strat)][metric].values[0]
                    for g in gammas]
            offset = (j - len(strategies) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=strat, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'gamma={g}' for g in gammas])
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.legend(fontsize=8)

    plt.suptitle(f'Summary Statistics Comparison ({exp_label})', fontsize=13)
    plt.tight_layout()
    if save_path is None:
        save_path = RESULTS_DIR / f'{exp_label}_summary_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {save_path}')
    return fig


def generate_all_plots(
    all_results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    rep_paths: Dict[Tuple, pd.DataFrame],
    exp_label: str,
    params,
) -> None:
    """Generate all plots for an experiment."""
    print(f'Generating plots for {exp_label}...')
    gammas = params.gammas

    # Representative path plots
    for (gamma, strat_name), path_df in rep_paths.items():
        if path_df is not None:
            plot_representative_path(path_df, gamma, strat_name, exp_label=exp_label)

    # Profit histograms per gamma
    for gamma in gammas:
        plot_profit_histograms(all_results_df, gamma, exp_label=exp_label)

    # Summary comparison
    plot_summary_comparison(summary_df, exp_label=exp_label)
    print(f'All {exp_label} plots saved to results/')
