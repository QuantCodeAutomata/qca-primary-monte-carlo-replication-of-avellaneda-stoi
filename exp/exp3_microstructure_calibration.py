"""
Experiment 3: Microstructure-to-Intensity Calibration Mapping.

Reproduces the paper's derivation linking market-order size distributions
and price-impact laws to reduced-form execution intensities.

Usage:
    python -m exp.exp3_microstructure_calibration
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

from src.microstructure_calibration import (
    exponential_intensity,
    power_law_intensity,
    interpret_exponential_params,
    exponential_intensity_sensitivity,
    power_law_intensity_sensitivity,
    tabulate_exponential_params,
    tabulate_power_law_exponents,
    ALPHA_US_STOCKS, ALPHA_NASDAQ, ALPHA_PARIS,
    BETA_SQUARE_ROOT, BETA_EMPIRICAL,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_exp3() -> None:
    """
    Run the microstructure calibration experiment:
    1. Print symbolic derivation summary
    2. Tabulate parameters for representative alpha/K/beta values
    3. Generate sensitivity plots
    """
    print("\n[EXP3] Microstructure Derivation Summary")
    print("-" * 50)
    print("CASE 1: Logarithmic Impact → Exponential Intensity")
    print("  Market-order size: P(Q > x) ∝ x^{-alpha}")
    print("  Impact: Δp = (1/K) ln(Q)  →  Q(δ) = exp(K·δ)")
    print("  P(Q > Q(δ)) ∝ exp(-alpha·K·δ)")
    print("  λ(δ) = A·exp(-k·δ)  where A = Λ/alpha, k = alpha·K")
    print()
    print("CASE 2: Power-Law Impact → Power-Law Intensity")
    print("  Market-order size: P(Q > x) ∝ x^{-alpha}")
    print("  Impact: Δp ∝ Q^beta  →  Q(δ) ∝ δ^{1/beta}")
    print("  P(Q > Q(δ)) ∝ δ^{-alpha/beta}")
    print("  λ(δ) = B·δ^{-alpha/beta}")

    # Tabulate exponential parameters
    print("\n[EXP3] Exponential Intensity Parameters (A, k) for representative values:")
    exp_table = tabulate_exponential_params(Lambda=1.0, K_values=[0.5, 1.0, 1.5, 2.0])
    df_exp = pd.DataFrame(exp_table)
    print(df_exp.to_string(index=False))
    df_exp.to_csv(os.path.join(RESULTS_DIR, "exp3_exponential_params.csv"), index=False)

    # Tabulate power-law exponents
    print("\n[EXP3] Power-Law Intensity Exponents (alpha/beta):")
    pl_table = tabulate_power_law_exponents()
    df_pl = pd.DataFrame(pl_table)
    print(df_pl.to_string(index=False))
    df_pl.to_csv(os.path.join(RESULTS_DIR, "exp3_powerlaw_exponents.csv"), index=False)

    # Verify main simulation parameters
    Lambda = 1.0
    alpha_ref = ALPHA_US_STOCKS
    K_ref = 1.5 / alpha_ref  # k=1.5 → K = k/alpha
    A_ref, k_ref = interpret_exponential_params(Lambda, alpha_ref, K_ref)
    print(f"\n[EXP3] Main simulation params interpretation:")
    print(f"  A=140, k=1.5 correspond to (with Lambda=140, alpha={alpha_ref}):")
    A_main, k_main = interpret_exponential_params(140.0, alpha_ref, K_ref)
    print(f"  A={A_main:.2f}, k={k_main:.4f}")

    return df_exp, df_pl


def plot_exponential_sensitivity() -> None:
    """
    Plot exponential intensity curves for different alpha and K values.
    """
    delta_grid = np.linspace(0.01, 3.0, 300)
    Lambda = 1.0
    alpha_values = [ALPHA_NASDAQ, ALPHA_PARIS, ALPHA_US_STOCKS]
    K_values = [0.5, 1.0, 1.5]

    results = exponential_intensity_sensitivity(delta_grid, Lambda, alpha_values, K_values)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Vary alpha, fix K=1.0
    ax = axes[0]
    K_fixed = 1.0
    colors = ["steelblue", "tomato", "green"]
    for alpha, color in zip(alpha_values, colors):
        A, k = interpret_exponential_params(Lambda, alpha, K_fixed)
        lam = results[(alpha, K_fixed)]
        ax.plot(delta_grid, lam, label=f"α={alpha}", color=color, lw=1.5)
    ax.set_xlabel("Quote distance δ")
    ax.set_ylabel("Execution intensity λ(δ)")
    ax.set_title(f"Exponential Intensity: Varying α (K={K_fixed})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Vary K, fix alpha=ALPHA_US_STOCKS
    ax = axes[1]
    alpha_fixed = ALPHA_US_STOCKS
    colors = ["steelblue", "tomato", "green"]
    for K, color in zip(K_values, colors):
        A, k = interpret_exponential_params(Lambda, alpha_fixed, K)
        lam = results[(alpha_fixed, K)]
        ax.plot(delta_grid, lam, label=f"K={K}", color=color, lw=1.5)
    ax.set_xlabel("Quote distance δ")
    ax.set_ylabel("Execution intensity λ(δ)")
    ax.set_title(f"Exponential Intensity: Varying K (α={alpha_fixed})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Microstructure Calibration: Exponential Intensity Sensitivity\n"
                 "λ(δ) = A·exp(-k·δ)  [Logarithmic Impact Case]", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "exp3_exponential_sensitivity.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_power_law_sensitivity() -> None:
    """
    Plot power-law intensity curves for different alpha and beta values.
    """
    delta_grid = np.linspace(0.05, 3.0, 300)  # avoid delta=0 (singular)
    B = 1.0
    alpha_values = [ALPHA_NASDAQ, ALPHA_PARIS, ALPHA_US_STOCKS]
    beta_values = [BETA_SQUARE_ROOT, BETA_EMPIRICAL]

    results = power_law_intensity_sensitivity(delta_grid, B, alpha_values, beta_values)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Vary alpha, fix beta=0.5
    ax = axes[0]
    beta_fixed = BETA_SQUARE_ROOT
    colors = ["steelblue", "tomato", "green"]
    for alpha, color in zip(alpha_values, colors):
        lam = results[(alpha, beta_fixed)]
        ax.plot(delta_grid, lam, label=f"α={alpha}", color=color, lw=1.5)
    ax.set_xlabel("Quote distance δ")
    ax.set_ylabel("Execution intensity λ(δ)")
    ax.set_title(f"Power-Law Intensity: Varying α (β={beta_fixed})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Vary beta, fix alpha=ALPHA_US_STOCKS
    ax = axes[1]
    alpha_fixed = ALPHA_US_STOCKS
    colors = ["steelblue", "tomato"]
    for beta, color in zip(beta_values, colors):
        lam = results[(alpha_fixed, beta)]
        ax.plot(delta_grid, lam, label=f"β={beta}", color=color, lw=1.5)
    ax.set_xlabel("Quote distance δ")
    ax.set_ylabel("Execution intensity λ(δ)")
    ax.set_title(f"Power-Law Intensity: Varying β (α={alpha_fixed})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Microstructure Calibration: Power-Law Intensity Sensitivity\n"
                 "λ(δ) = B·δ^{-α/β}  [Power-Law Impact Case]", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "exp3_powerlaw_sensitivity.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_model_comparison() -> None:
    """
    Compare exponential vs. power-law intensity on the same axes.
    """
    delta_grid_exp = np.linspace(0.01, 3.0, 300)
    delta_grid_pl = np.linspace(0.05, 3.0, 300)

    # Use representative parameters
    Lambda = 1.0
    alpha = ALPHA_US_STOCKS
    K = 1.0
    A, k = interpret_exponential_params(Lambda, alpha, K)
    B = 1.0
    beta = BETA_EMPIRICAL

    lam_exp = exponential_intensity(delta_grid_exp, A, k)
    lam_pl = power_law_intensity(delta_grid_pl, B, alpha, beta)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(delta_grid_exp, lam_exp, label=f"Exponential: A·exp(-k·δ)\n(α={alpha}, K={K})",
            color="steelblue", lw=2)
    ax.plot(delta_grid_pl, lam_pl, label=f"Power-law: B·δ^{{-α/β}}\n(α={alpha}, β={beta})",
            color="tomato", lw=2, linestyle="--")
    ax.set_xlabel("Quote distance δ")
    ax.set_ylabel("Execution intensity λ(δ)")
    ax.set_title("Exponential vs. Power-Law Execution Intensity\n"
                 "Microstructure Calibration Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, min(lam_exp.max(), lam_pl.max()) * 1.1)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "exp3_model_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("EXP 3: Microstructure-to-Intensity Calibration Mapping")
    print("=" * 60)

    df_exp, df_pl = run_exp3()

    print("\n[EXP3] Generating plots...")
    plot_exponential_sensitivity()
    plot_power_law_sensitivity()
    plot_model_comparison()

    print("\n[EXP3] Done.")
