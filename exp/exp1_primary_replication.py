"""
Experiment 1: Primary Monte Carlo Replication of Avellaneda-Stoikov
Finite-Horizon Inventory-Based Market Making.

Reproduces the paper's main numerical simulations comparing:
  - Inventory-based quoting strategy
  - Symmetric mid-price-centered benchmark

for gamma in {0.1, 0.01, 0.5}.

Usage:
    python -m exp.exp1_primary_replication
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.avellaneda_stoikov import (
    ASParams,
    run_monte_carlo,
    validate_reservation_price_properties,
    validate_lambda_dt,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

GAMMA_VALUES = [0.1, 0.01, 0.5]


def run_exp1() -> pd.DataFrame:
    """
    Run the primary Monte Carlo replication for all gamma values.

    Returns
    -------
    pd.DataFrame : summary statistics table
    """
    rows = []

    for gamma in GAMMA_VALUES:
        params = ASParams(gamma=gamma, seed=42)

        # Pre-simulation validations
        validate_reservation_price_properties(params)
        validate_lambda_dt(params)

        print(f"\n[EXP1] gamma={gamma}")

        for strategy in ("inventory", "symmetric"):
            result = run_monte_carlo(params, strategy)
            print(
                f"  {strategy:12s} | "
                f"E[Pi]={result['mean_profit']:8.2f}  "
                f"Std[Pi]={result['std_profit']:8.2f}  "
                f"E[q_T]={result['mean_inventory']:6.2f}  "
                f"Std[q_T]={result['std_inventory']:6.2f}"
            )
            rows.append({
                "gamma": gamma,
                "strategy": strategy,
                "mean_profit": round(result["mean_profit"], 2),
                "std_profit": round(result["std_profit"], 2),
                "mean_inventory": round(result["mean_inventory"], 2),
                "std_inventory": round(result["std_inventory"], 2),
            })

    df = pd.DataFrame(rows)
    return df


def plot_sample_path(gamma: float = 0.1) -> None:
    """
    Plot a representative path showing mid-price, reservation price,
    bid quote, and ask quote for the inventory strategy.

    Parameters
    ----------
    gamma : risk-aversion coefficient for the sample path
    """
    params = ASParams(gamma=gamma, seed=42)
    result = run_monte_carlo(params, "inventory")
    path = result["sample_path"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Price panel
    ax1 = axes[0]
    ax1.plot(path["time"], path["s"], label="Mid-price $S_t$", color="black", lw=1.5)
    ax1.plot(path["time"], path["r"], label="Reservation price $r_t$",
             color="blue", lw=1.2, linestyle="--")
    ax1.plot(path["time"], path["p_ask"], label="Ask quote $p^a_t$",
             color="red", lw=0.8, alpha=0.7)
    ax1.plot(path["time"], path["p_bid"], label="Bid quote $p^b_t$",
             color="green", lw=0.8, alpha=0.7)
    ax1.set_ylabel("Price")
    ax1.set_title(
        f"Avellaneda-Stoikov: Sample Path (Inventory Strategy, γ={gamma})\n"
        "Finite-Horizon Approximation"
    )
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Inventory panel
    ax2 = axes[1]
    ax2.step(path["time"], path["q"], label="Inventory $q_t$",
             color="purple", lw=1.2, where="post")
    ax2.axhline(0, color="black", lw=0.5, linestyle=":")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Inventory")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"exp1_sample_path_gamma{gamma}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_profit_histograms(gamma: float = 0.1) -> None:
    """
    Plot overlaid histograms of terminal profit for inventory and symmetric strategies.

    Parameters
    ----------
    gamma : risk-aversion coefficient
    """
    params = ASParams(gamma=gamma, seed=42)

    inv_result = run_monte_carlo(params, "inventory")
    sym_result = run_monte_carlo(params, "symmetric")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        inv_result["profits"], bins=60, alpha=0.6,
        label=f"Inventory (μ={inv_result['mean_profit']:.2f}, σ={inv_result['std_profit']:.2f})",
        color="steelblue", density=True
    )
    ax.hist(
        sym_result["profits"], bins=60, alpha=0.6,
        label=f"Symmetric (μ={sym_result['mean_profit']:.2f}, σ={sym_result['std_profit']:.2f})",
        color="tomato", density=True
    )
    ax.set_xlabel("Terminal Profit $\\Pi_T = X_T + q_T S_T$")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Terminal Profit Distribution — γ={gamma}\n"
        "Inventory vs. Symmetric Strategy (1000 paths)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"exp1_profit_hist_gamma{gamma}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_all_gamma_comparison(df: pd.DataFrame) -> None:
    """
    Bar chart comparing std of profit and std of inventory across gamma values.

    Parameters
    ----------
    df : summary statistics DataFrame from run_exp1()
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    gammas = GAMMA_VALUES
    x = np.arange(len(gammas))
    width = 0.35

    for ax, metric, ylabel, title in [
        (axes[0], "std_profit", "Std of Terminal Profit", "Risk (Std Profit) by γ"),
        (axes[1], "std_inventory", "Std of Terminal Inventory", "Inventory Risk (Std q_T) by γ"),
    ]:
        inv_vals = [df[(df.gamma == g) & (df.strategy == "inventory")][metric].values[0]
                    for g in gammas]
        sym_vals = [df[(df.gamma == g) & (df.strategy == "symmetric")][metric].values[0]
                    for g in gammas]

        bars1 = ax.bar(x - width / 2, inv_vals, width, label="Inventory", color="steelblue")
        bars2 = ax.bar(x + width / 2, sym_vals, width, label="Symmetric", color="tomato")
        ax.set_xticks(x)
        ax.set_xticklabels([f"γ={g}" for g in gammas])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Avellaneda-Stoikov: Risk Comparison Across γ Values\n"
                 "Inventory vs. Symmetric Strategy", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "exp1_gamma_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def save_results_table(df: pd.DataFrame) -> None:
    """Save summary statistics to CSV."""
    out_path = os.path.join(RESULTS_DIR, "exp1_summary_statistics.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("EXP 1: Primary Monte Carlo Replication")
    print("Avellaneda-Stoikov Finite-Horizon Market Making")
    print("=" * 60)

    df = run_exp1()
    save_results_table(df)

    print("\n[EXP1] Generating plots...")
    for gamma in GAMMA_VALUES:
        plot_sample_path(gamma)
        plot_profit_histograms(gamma)

    plot_all_gamma_comparison(df)

    print("\n[EXP1] Summary Table:")
    print(df.to_string(index=False))
    print("\n[EXP1] Done.")
