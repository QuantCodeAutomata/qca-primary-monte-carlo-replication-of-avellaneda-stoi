"""
Experiment 1: Primary Monte Carlo Replication of Avellaneda-Stoikov
Finite-Horizon Inventory-Based Market Making.

Reproduces the paper's main numerical simulations using:
  - Arithmetic Brownian Motion (binomial discretisation)
  - Finite-horizon closed-form quoting rules (Eq. 3.8 / 3.9)
  - Symmetric exponential execution intensities
  - Inventory-based vs. symmetric benchmark strategies

Reference: Avellaneda & Stoikov (2008), Quantitative Finance.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

from src.simulation import (
    SimParams,
    SimResults,
    run_monte_carlo,
    draw_shared_randoms,
    validate_parameters,
)

# Custom — Context7 found no library equivalent for AS finite-horizon quoting rules


GAMMAS = [0.1, 0.01, 0.5]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run_experiment_1(
    params: SimParams | None = None,
    gammas: List[float] | None = None,
    results_dir: str = RESULTS_DIR,
) -> Dict[str, Dict[str, SimResults]]:
    """
    Run Experiment 1: primary finite-horizon Monte Carlo replication.

    For each gamma, runs both inventory-based and symmetric strategies using
    common random numbers (shared price paths and execution uniforms) so that
    differences are attributable to strategy choice alone.

    Parameters
    ----------
    params      : SimParams (defaults to paper's baseline values)
    gammas      : list of risk-aversion values (defaults to {0.1, 0.01, 0.5})
    results_dir : directory to save plots and tables

    Returns
    -------
    Nested dict: results[gamma_str][strategy] = SimResults
    """
    if params is None:
        params = SimParams()
    if gammas is None:
        gammas = GAMMAS

    os.makedirs(results_dir, exist_ok=True)

    # Validate parameters before running
    for g in gammas:
        validate_parameters(params, g)

    all_results: Dict[str, Dict[str, SimResults]] = {}

    for gamma in gammas:
        g_key = f"gamma_{gamma}"
        all_results[g_key] = {}

        # Draw shared random numbers so both strategies see identical price paths
        # and execution uniforms — differences are purely due to quoting rules.
        price_moves, ask_uniforms, bid_uniforms = draw_shared_randoms(params)

        for strategy in ["inventory", "symmetric"]:
            res = run_monte_carlo(
                params,
                gamma,
                strategy,
                shared_price_moves=price_moves,
                shared_ask_uniforms=ask_uniforms,
                shared_bid_uniforms=bid_uniforms,
            )
            all_results[g_key][strategy] = res

    # Generate outputs
    _print_summary_table(all_results, gammas, label="Exp 1 (Finite-Horizon)")
    _save_summary_csv(all_results, gammas, results_dir, prefix="exp1")
    _plot_sample_path(all_results, gammas, results_dir, prefix="exp1")
    _plot_profit_histograms(all_results, gammas, results_dir, prefix="exp1")

    return all_results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_summary_table(
    results: Dict[str, Dict[str, SimResults]],
    gammas: List[float],
    label: str = "",
) -> None:
    """Print a formatted summary table to stdout."""
    header = f"\n{'='*70}\n{label} — Summary Statistics\n{'='*70}"
    print(header)
    fmt = "{:<10} {:<12} {:>12} {:>12} {:>12} {:>12}"
    print(fmt.format("gamma", "strategy", "E[Pi_T]", "Std[Pi_T]", "E[q_T]", "Std[q_T]"))
    print("-" * 70)
    for gamma in gammas:
        g_key = f"gamma_{gamma}"
        for strategy in ["inventory", "symmetric"]:
            r = results[g_key][strategy]
            print(
                fmt.format(
                    f"{gamma:.3f}",
                    strategy,
                    f"{r.mean_profit:.2f}",
                    f"{r.std_profit:.2f}",
                    f"{r.mean_final_q:.2f}",
                    f"{r.std_final_q:.2f}",
                )
            )
    print("=" * 70)


def _save_summary_csv(
    results: Dict[str, Dict[str, SimResults]],
    gammas: List[float],
    results_dir: str,
    prefix: str = "exp1",
) -> None:
    """Save summary statistics to a CSV file."""
    rows = []
    for gamma in gammas:
        g_key = f"gamma_{gamma}"
        for strategy in ["inventory", "symmetric"]:
            r = results[g_key][strategy]
            rows.append(
                {
                    "gamma": gamma,
                    "strategy": strategy,
                    "mean_profit": round(r.mean_profit, 4),
                    "std_profit": round(r.std_profit, 4),
                    "mean_final_q": round(r.mean_final_q, 4),
                    "std_final_q": round(r.std_final_q, 4),
                }
            )
    df = pd.DataFrame(rows)
    path = os.path.join(results_dir, f"{prefix}_summary.csv")
    df.to_csv(path, index=False)
    print(f"Saved summary CSV → {path}")


def _plot_sample_path(
    results: Dict[str, Dict[str, SimResults]],
    gammas: List[float],
    results_dir: str,
    prefix: str = "exp1",
) -> None:
    """
    Plot one representative path showing mid-price, reservation price,
    bid quote, and ask quote for the inventory strategy.
    """
    for gamma in gammas:
        g_key = f"gamma_{gamma}"
        path = results[g_key]["inventory"].sample_path
        if path is None:
            continue

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        t = path.time[:-1]  # quotes defined at steps 0..n-1

        # Panel 1: prices
        ax = axes[0]
        ax.plot(path.time, path.s, color="black", lw=1.2, label="Mid-price $S_t$")
        ax.plot(t, path.r[:-1], color="blue", lw=1.0, ls="--", label="Reservation price $r_t$")
        ax.plot(t, path.pa[:-1], color="green", lw=0.8, ls=":", label="Ask quote $p^a_t$")
        ax.plot(t, path.pb[:-1], color="red", lw=0.8, ls=":", label="Bid quote $p^b_t$")
        ax.set_ylabel("Price")
        ax.set_title(f"Sample Path — Inventory Strategy (γ={gamma})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel 2: inventory
        ax = axes[1]
        ax.step(path.time, path.q, color="purple", lw=1.0, where="post")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_ylabel("Inventory $q_t$")
        ax.grid(True, alpha=0.3)

        # Panel 3: cumulative cash
        ax = axes[2]
        ax.plot(path.time, path.x, color="darkorange", lw=1.0)
        ax.set_ylabel("Cash $X_t$")
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(results_dir, f"{prefix}_sample_path_gamma{gamma}.png")
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"Saved sample path plot → {fig_path}")


def _plot_profit_histograms(
    results: Dict[str, Dict[str, SimResults]],
    gammas: List[float],
    results_dir: str,
    prefix: str = "exp1",
) -> None:
    """
    Plot overlaid histograms of terminal profit for inventory and symmetric
    strategies for each gamma value.
    """
    fig, axes = plt.subplots(1, len(gammas), figsize=(5 * len(gammas), 5))
    if len(gammas) == 1:
        axes = [axes]

    for ax, gamma in zip(axes, gammas):
        g_key = f"gamma_{gamma}"
        inv_profits = results[g_key]["inventory"].profits
        sym_profits = results[g_key]["symmetric"].profits

        bins = np.linspace(
            min(inv_profits.min(), sym_profits.min()),
            max(inv_profits.max(), sym_profits.max()),
            50,
        )
        ax.hist(inv_profits, bins=bins, alpha=0.55, color="steelblue",
                label="Inventory", density=True)
        ax.hist(sym_profits, bins=bins, alpha=0.55, color="tomato",
                label="Symmetric", density=True)
        ax.set_title(f"Terminal Profit Distribution\nγ={gamma}")
        ax.set_xlabel("$\\Pi_T = X_T + q_T S_T$")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Annotate means
        ax.axvline(np.mean(inv_profits), color="steelblue", lw=1.5, ls="--")
        ax.axvline(np.mean(sym_profits), color="tomato", lw=1.5, ls="--")

    plt.suptitle("Exp 1: Terminal Profit Histograms (Finite-Horizon Formulas)", y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(results_dir, f"{prefix}_profit_histograms.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved profit histogram plot → {fig_path}")


def build_results_markdown(
    results: Dict[str, Dict[str, SimResults]],
    gammas: List[float],
    label: str = "Experiment 1",
) -> str:
    """Build a Markdown table string for the results."""
    lines = [
        f"## {label}",
        "",
        "| gamma | strategy | E[Pi_T] | Std[Pi_T] | E[q_T] | Std[q_T] |",
        "|-------|----------|---------|-----------|--------|----------|",
    ]
    for gamma in gammas:
        g_key = f"gamma_{gamma}"
        for strategy in ["inventory", "symmetric"]:
            r = results[g_key][strategy]
            lines.append(
                f"| {gamma} | {strategy} | {r.mean_profit:.2f} | "
                f"{r.std_profit:.2f} | {r.mean_final_q:.2f} | {r.std_final_q:.2f} |"
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    results = run_experiment_1()
