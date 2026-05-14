"""
Experiment 2: Diagnostic Replication — Spread-Ambiguity Variant.

Tests whether the paper's reported table values are better matched by using
the constant spread term 2/gamma*ln(1+gamma/k) instead of the full
time-varying finite-horizon spread.

Usage:
    python -m exp.exp2_diagnostic_variant
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.avellaneda_stoikov import ASParams, run_monte_carlo, optimal_half_spread_base
from src.diagnostic_variant import run_monte_carlo_diagnostic

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

GAMMA_VALUES = [0.1, 0.01, 0.5]


def compute_reported_spread(gamma: float, k: float) -> float:
    """
    Compute the constant spread reported in the paper's tables.

    spread = 2/gamma * ln(1 + gamma/k)

    Parameters
    ----------
    gamma : risk-aversion coefficient
    k     : intensity decay parameter

    Returns
    -------
    float : constant spread
    """
    return 2.0 * optimal_half_spread_base(gamma, k)


def run_exp2() -> pd.DataFrame:
    """
    Run the diagnostic (constant-spread) replication for all gamma values.

    Returns
    -------
    pd.DataFrame : summary statistics table
    """
    rows = []

    for gamma in GAMMA_VALUES:
        params = ASParams(gamma=gamma, seed=42)
        reported_spread = compute_reported_spread(gamma, params.k)

        print(f"\n[EXP2] gamma={gamma}  |  Reported spread = {reported_spread:.4f}")

        for strategy in ("inventory", "symmetric"):
            result = run_monte_carlo_diagnostic(params, strategy)
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
                "reported_spread": round(reported_spread, 4),
                "mean_profit": round(result["mean_profit"], 2),
                "std_profit": round(result["std_profit"], 2),
                "mean_inventory": round(result["mean_inventory"], 2),
                "std_inventory": round(result["std_inventory"], 2),
            })

    df = pd.DataFrame(rows)
    return df


def compare_exp1_vs_exp2() -> pd.DataFrame:
    """
    Compare primary (finite-horizon) vs. diagnostic (constant-spread) results.

    Returns
    -------
    pd.DataFrame : comparison table
    """
    rows = []

    for gamma in GAMMA_VALUES:
        params = ASParams(gamma=gamma, seed=42)

        for strategy in ("inventory", "symmetric"):
            # Primary (finite-horizon)
            r1 = run_monte_carlo(params, strategy)
            # Diagnostic (constant-spread)
            r2 = run_monte_carlo_diagnostic(params, strategy)

            rows.append({
                "gamma": gamma,
                "strategy": strategy,
                "exp1_mean_profit": round(r1["mean_profit"], 2),
                "exp2_mean_profit": round(r2["mean_profit"], 2),
                "exp1_std_profit": round(r1["std_profit"], 2),
                "exp2_std_profit": round(r2["std_profit"], 2),
                "exp1_std_inventory": round(r1["std_inventory"], 2),
                "exp2_std_inventory": round(r2["std_inventory"], 2),
            })

    return pd.DataFrame(rows)


def plot_comparison(df_compare: pd.DataFrame) -> None:
    """
    Plot comparison of exp1 vs exp2 for std_profit and std_inventory.

    Parameters
    ----------
    df_compare : comparison DataFrame from compare_exp1_vs_exp2()
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    gammas = GAMMA_VALUES
    x = np.arange(len(gammas))
    width = 0.2

    for ax, metric_base, ylabel, title in [
        (axes[0], "std_profit", "Std of Terminal Profit",
         "Spread Ambiguity Effect on Profit Risk"),
        (axes[1], "std_inventory", "Std of Terminal Inventory",
         "Spread Ambiguity Effect on Inventory Risk"),
    ]:
        for i, (strategy, color_pair) in enumerate(
            [("inventory", ("steelblue", "cornflowerblue")),
             ("symmetric", ("tomato", "lightsalmon"))]
        ):
            sub = df_compare[df_compare.strategy == strategy]
            e1_vals = [sub[sub.gamma == g][f"exp1_{metric_base}"].values[0] for g in gammas]
            e2_vals = [sub[sub.gamma == g][f"exp2_{metric_base}"].values[0] for g in gammas]

            offset = (i - 0.75) * width
            ax.bar(x + offset, e1_vals, width,
                   label=f"{strategy} (finite-horizon)", color=color_pair[0], alpha=0.85)
            ax.bar(x + offset + width, e2_vals, width,
                   label=f"{strategy} (constant-spread)", color=color_pair[1], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([f"γ={g}" for g in gammas])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("EXP2: Finite-Horizon vs. Constant-Spread Interpretation\n"
                 "(Spread Ambiguity Diagnostic)", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "exp2_spread_ambiguity_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def save_results(df: pd.DataFrame, df_compare: pd.DataFrame) -> None:
    """Save results to CSV."""
    df.to_csv(os.path.join(RESULTS_DIR, "exp2_diagnostic_statistics.csv"), index=False)
    df_compare.to_csv(os.path.join(RESULTS_DIR, "exp2_comparison_exp1_vs_exp2.csv"), index=False)
    print(f"  Saved exp2 CSVs to {RESULTS_DIR}")


if __name__ == "__main__":
    print("=" * 60)
    print("EXP 2: Diagnostic Replication — Spread-Ambiguity Variant")
    print("=" * 60)

    df = run_exp2()
    df_compare = compare_exp1_vs_exp2()
    save_results(df, df_compare)

    print("\n[EXP2] Generating comparison plot...")
    plot_comparison(df_compare)

    print("\n[EXP2] Diagnostic Summary Table:")
    print(df.to_string(index=False))

    print("\n[EXP2] Exp1 vs Exp2 Comparison:")
    print(df_compare.to_string(index=False))

    print("\n[EXP2] Done.")
