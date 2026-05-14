"""
Experiment 4: Appendix Extension — Mean-Variance Market Making Under GBM.

Reproduces the paper's appendix model as a separate theoretical extension,
using geometric Brownian motion and a mean-variance objective.

Usage:
    python -m exp.exp4_appendix_gbm
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

from src.appendix_gbm import (
    value_function,
    reservation_ask_gbm,
    reservation_bid_gbm,
    reservation_midpoint_gbm,
    reservation_price_abm,
    validate_gbm_reservation_properties,
    compute_reservation_price_grid,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Baseline parameters
S0 = 100.0
SIGMA = 0.2   # GBM volatility (dimensionless, not the ABM sigma=2)
T = 1.0
GAMMA_VALUES = [0.1, 0.01, 0.5]
Q_GRID = np.arange(-10, 11)
T_GRID = np.linspace(0, T * 0.99, 50)  # avoid t=T exactly for numerical stability


def run_exp4() -> None:
    """
    Run the appendix GBM mean-variance experiment:
    1. Validate reservation price properties
    2. Compute and display reservation prices over (q, t) grid
    3. Compare GBM appendix vs. ABM main-model reservation prices
    """
    print("\n[EXP4] Validating GBM reservation price properties...")
    for gamma in GAMMA_VALUES:
        validate_gbm_reservation_properties(s=S0, sigma=SIGMA, T=T, gamma=gamma)
        print(f"  gamma={gamma}: All properties validated ✓")

    # Tabulate reservation prices at t=0 for different q and gamma
    rows = []
    for gamma in GAMMA_VALUES:
        for q in Q_GRID:
            R = reservation_midpoint_gbm(S0, q, 0.0, T, SIGMA, gamma)
            Ra = reservation_ask_gbm(S0, q, 0.0, T, SIGMA, gamma)
            Rb = reservation_bid_gbm(S0, q, 0.0, T, SIGMA, gamma)
            r_abm = reservation_price_abm(S0, q, 0.0, T, 2.0, gamma)  # ABM sigma=2
            rows.append({
                "gamma": gamma,
                "q": q,
                "R_gbm": round(R, 4),
                "Ra_gbm": round(Ra, 4),
                "Rb_gbm": round(Rb, 4),
                "r_abm": round(r_abm, 4),
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "exp4_reservation_prices.csv"), index=False)
    print(f"\n[EXP4] Saved reservation price table to results/")

    # Print sample for gamma=0.1
    print("\n[EXP4] Sample reservation prices (gamma=0.1, t=0):")
    sample = df[df.gamma == 0.1][["q", "R_gbm", "Ra_gbm", "Rb_gbm", "r_abm"]]
    print(sample.to_string(index=False))

    return df


def plot_reservation_price_vs_inventory(df: pd.DataFrame) -> None:
    """
    Plot reservation price vs. inventory for different gamma values.

    Parameters
    ----------
    df : DataFrame from run_exp4()
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax, gamma in zip(axes, GAMMA_VALUES):
        sub = df[df.gamma == gamma]
        ax.plot(sub["q"], sub["R_gbm"], label="GBM appendix R(s,q,0)",
                color="steelblue", lw=2, marker="o", markersize=4)
        ax.plot(sub["q"], sub["r_abm"], label="ABM main model r(s,q,0)",
                color="tomato", lw=2, linestyle="--", marker="s", markersize=4)
        ax.axhline(S0, color="black", lw=0.8, linestyle=":", label="Mid-price s=100")
        ax.set_xlabel("Inventory q")
        ax.set_ylabel("Reservation Price")
        ax.set_title(f"γ={gamma}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Appendix GBM vs. Main ABM: Reservation Price vs. Inventory\n"
                 "(t=0, s=100, T=1)", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "exp4_reservation_vs_inventory.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_reservation_price_vs_time() -> None:
    """
    Plot reservation price vs. time for different inventory levels.
    """
    gamma = 0.1
    q_values = [-5, 0, 5]
    t_values = np.linspace(0, T * 0.99, 100)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # GBM appendix
    ax = axes[0]
    colors = ["steelblue", "black", "tomato"]
    for q, color in zip(q_values, colors):
        R_vals = [reservation_midpoint_gbm(S0, q, t, T, SIGMA, gamma) for t in t_values]
        ax.plot(t_values, R_vals, label=f"q={q}", color=color, lw=1.5)
    ax.axhline(S0, color="gray", lw=0.8, linestyle=":", label="Mid-price")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Reservation Price R(s,q,t)")
    ax.set_title(f"GBM Appendix: R vs. Time (γ={gamma})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ABM main model
    ax = axes[1]
    for q, color in zip(q_values, colors):
        r_vals = [reservation_price_abm(S0, q, t, T, 2.0, gamma) for t in t_values]
        ax.plot(t_values, r_vals, label=f"q={q}", color=color, lw=1.5)
    ax.axhline(S0, color="gray", lw=0.8, linestyle=":", label="Mid-price")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Reservation Price r(s,q,t)")
    ax.set_title(f"ABM Main Model: r vs. Time (γ={gamma})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Reservation Price Dynamics: GBM Appendix vs. ABM Main Model\n"
                 "(s=100, T=1, γ=0.1)", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "exp4_reservation_vs_time.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_value_function_surface() -> None:
    """
    Plot the appendix value function V(x,s,q,t) as a function of q and t.
    """
    gamma = 0.1
    x = 0.0
    s = S0
    q_vals = np.linspace(-10, 10, 50)
    t_vals = np.linspace(0, T * 0.99, 50)
    Q, Tv = np.meshgrid(q_vals, t_vals)

    V = np.vectorize(lambda q, t: value_function(x, s, q, t, T, SIGMA, gamma))(Q, Tv)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(Q, Tv, V, cmap="viridis", alpha=0.85)
    ax.set_xlabel("Inventory q")
    ax.set_ylabel("Time t")
    ax.set_zlabel("V(x,s,q,t)")
    ax.set_title(f"Appendix Value Function V(x,s,q,t)\n"
                 f"(GBM, Mean-Variance, γ={gamma}, s={s}, x={x})")
    fig.colorbar(surf, shrink=0.5, aspect=10)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "exp4_value_function_surface.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("EXP 4: Appendix Extension — GBM + Mean-Variance")
    print("=" * 60)

    df = run_exp4()

    print("\n[EXP4] Generating plots...")
    plot_reservation_price_vs_inventory(df)
    plot_reservation_price_vs_time()
    plot_value_function_surface()

    print("\n[EXP4] Done.")
