"""
Experiment 4: Appendix Extension — Mean-Variance Market Making Under GBM.

Reproduces the paper's appendix model as a separate theoretical and numerical
extension, using geometric Brownian motion for prices and a mean-variance
objective rather than the main text's arithmetic BM and exponential utility.

Appendix value function:
  V(x, s, q, t) = x + q*s + (gamma*q^2*s^2/2) * (exp(sigma^2*(T-t)) - 1)

Reservation ask and bid prices:
  R^a(s, q, t) = s + ((1-2q)/2) * gamma * s^2 * (exp(sigma^2*(T-t)) - 1)
  R^b(s, q, t) = s + ((-1-2q)/2) * gamma * s^2 * (exp(sigma^2*(T-t)) - 1)

Reservation price midpoint:
  R(s, q, t) = (R^a + R^b) / 2 = s - q * gamma * s^2 * (exp(sigma^2*(T-t)) - 1)

Reference: Avellaneda & Stoikov (2008), Quantitative Finance — Appendix.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Custom — Context7 found no library equivalent for AS appendix GBM mean-variance formulas

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ---------------------------------------------------------------------------
# Appendix formulas
# ---------------------------------------------------------------------------

def appendix_value_function(
    x: float,
    s: float,
    q: float,
    t: float,
    gamma: float,
    sigma: float,
    T: float,
) -> float:
    """
    Appendix mean-variance value function.

    V(x, s, q, t) = x + q*s + (gamma*q^2*s^2/2) * (exp(sigma^2*(T-t)) - 1)

    Objective: E_t[(x + q*S_T)] - (gamma/2) * Var_t[q*S_T]
    Under GBM without drift: dS_u/S_u = sigma * dW_u

    Parameters
    ----------
    x     : current cash
    s     : current mid-price
    q     : current inventory
    t     : current time
    gamma : risk-aversion coefficient (mean-variance)
    sigma : volatility (GBM)
    T     : horizon
    """
    tau = T - t
    return x + q * s + (gamma * q ** 2 * s ** 2 / 2.0) * (np.exp(sigma ** 2 * tau) - 1.0)


def appendix_reservation_ask(
    s: float,
    q: float,
    t: float,
    gamma: float,
    sigma: float,
    T: float,
) -> float:
    """
    Appendix reservation ask price (indifference price for selling one unit).

    R^a(s, q, t) = s + ((1 - 2q) / 2) * gamma * s^2 * (exp(sigma^2*(T-t)) - 1)

    Parameters
    ----------
    s     : current mid-price
    q     : current inventory
    t     : current time
    gamma : risk-aversion coefficient
    sigma : volatility (GBM)
    T     : horizon
    """
    tau = T - t
    factor = (1.0 - 2.0 * q) / 2.0
    return s + factor * gamma * s ** 2 * (np.exp(sigma ** 2 * tau) - 1.0)


def appendix_reservation_bid(
    s: float,
    q: float,
    t: float,
    gamma: float,
    sigma: float,
    T: float,
) -> float:
    """
    Appendix reservation bid price (indifference price for buying one unit).

    R^b(s, q, t) = s + ((-1 - 2q) / 2) * gamma * s^2 * (exp(sigma^2*(T-t)) - 1)

    Parameters
    ----------
    s     : current mid-price
    q     : current inventory
    t     : current time
    gamma : risk-aversion coefficient
    sigma : volatility (GBM)
    T     : horizon
    """
    tau = T - t
    factor = (-1.0 - 2.0 * q) / 2.0
    return s + factor * gamma * s ** 2 * (np.exp(sigma ** 2 * tau) - 1.0)


def appendix_reservation_midpoint(
    s: float,
    q: float,
    t: float,
    gamma: float,
    sigma: float,
    T: float,
) -> float:
    """
    Appendix reservation price midpoint: R = (R^a + R^b) / 2.

    R(s, q, t) = s - q * gamma * s^2 * (exp(sigma^2*(T-t)) - 1)

    Parameters
    ----------
    s     : current mid-price
    q     : current inventory
    t     : current time
    gamma : risk-aversion coefficient
    sigma : volatility (GBM)
    T     : horizon
    """
    tau = T - t
    return s - q * gamma * s ** 2 * (np.exp(sigma ** 2 * tau) - 1.0)


def main_model_reservation_price(
    s: float,
    q: float,
    t: float,
    gamma: float,
    sigma: float,
    T: float,
) -> float:
    """
    Main model (arithmetic BM + CARA) reservation price for comparison.

    r(s, q, t) = s - q * gamma * sigma^2 * (T - t)

    Parameters
    ----------
    s     : current mid-price
    q     : current inventory
    t     : current time
    gamma : risk-aversion coefficient
    sigma : volatility (arithmetic BM)
    T     : horizon
    """
    tau = T - t
    return s - q * gamma * sigma ** 2 * tau


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_appendix_formulas(
    s: float = 100.0,
    gamma: float = 0.1,
    sigma: float = 0.2,
    T: float = 1.0,
) -> None:
    """
    Validate qualitative properties of the appendix reservation price.

    1. Positive inventory → R < s (personal valuation below mid-price)
    2. Negative inventory → R > s (personal valuation above mid-price)
    3. Adjustment vanishes as t → T
    4. R^a > R^b (ask above bid)
    """
    t_mid = T / 2.0

    # Check 1: positive inventory
    R_pos = appendix_reservation_midpoint(s, 5.0, t_mid, gamma, sigma, T)
    assert R_pos < s, f"Positive inventory should give R < s, got R={R_pos:.4f}"

    # Check 2: negative inventory
    R_neg = appendix_reservation_midpoint(s, -5.0, t_mid, gamma, sigma, T)
    assert R_neg > s, f"Negative inventory should give R > s, got R={R_neg:.4f}"

    # Check 3: adjustment vanishes at T
    R_at_T = appendix_reservation_midpoint(s, 10.0, T, gamma, sigma, T)
    assert abs(R_at_T - s) < 1e-12, f"R should equal s at T, got {R_at_T:.6f}"

    # Check 4: R^a > R^b
    Ra = appendix_reservation_ask(s, 0.0, t_mid, gamma, sigma, T)
    Rb = appendix_reservation_bid(s, 0.0, t_mid, gamma, sigma, T)
    assert Ra > Rb, f"R^a should exceed R^b, got R^a={Ra:.4f}, R^b={Rb:.4f}"

    print("Appendix formula validation passed.")


# ---------------------------------------------------------------------------
# Numerical illustrations
# ---------------------------------------------------------------------------

def run_experiment_4(
    results_dir: str = RESULTS_DIR,
    s0: float = 100.0,
    sigma: float = 0.2,
    T: float = 1.0,
    gammas: List[float] | None = None,
    q_grid: np.ndarray | None = None,
) -> Dict:
    """
    Run Experiment 4: appendix GBM mean-variance reservation price analysis.

    Produces:
      1. Validation of appendix formulas
      2. Reservation price vs inventory plots for different gamma and time
      3. Comparison with main model reservation price
      4. Value function surface illustration

    Parameters
    ----------
    results_dir : directory to save outputs
    s0          : current mid-price
    sigma       : GBM volatility
    T           : horizon
    gammas      : list of risk-aversion values
    q_grid      : inventory grid for plots

    Returns
    -------
    dict with computed reservation prices and comparison data
    """
    os.makedirs(results_dir, exist_ok=True)

    if gammas is None:
        gammas = [0.1, 0.01, 0.5]
    if q_grid is None:
        q_grid = np.arange(-10, 11, dtype=float)

    # Validate formulas
    validate_appendix_formulas(s=s0, gamma=0.1, sigma=sigma, T=T)

    # Time grid for time-evolution plots
    t_grid = np.linspace(0, T * 0.99, 100)

    results = {}

    # -----------------------------------------------------------------------
    # 1. Reservation price vs inventory for different gamma and time
    # -----------------------------------------------------------------------
    _plot_reservation_vs_inventory(s0, sigma, T, gammas, q_grid, results_dir)

    # -----------------------------------------------------------------------
    # 2. Reservation price vs time for different inventory levels
    # -----------------------------------------------------------------------
    _plot_reservation_vs_time(s0, sigma, T, gammas, t_grid, results_dir)

    # -----------------------------------------------------------------------
    # 3. Cross-model comparison: appendix vs main model
    # -----------------------------------------------------------------------
    comparison_data = _plot_model_comparison(s0, sigma, T, gammas, q_grid, t_grid, results_dir)
    results["comparison"] = comparison_data

    # -----------------------------------------------------------------------
    # 4. Print summary table
    # -----------------------------------------------------------------------
    _print_summary(s0, sigma, T, gammas, q_grid)

    return results


def _plot_reservation_vs_inventory(
    s0: float,
    sigma: float,
    T: float,
    gammas: List[float],
    q_grid: np.ndarray,
    results_dir: str,
) -> None:
    """Plot appendix reservation price midpoint vs inventory for different gamma."""
    t_values = [0.0, T / 2, T * 0.9]
    t_labels = ["t=0", "t=T/2", "t=0.9T"]

    fig, axes = plt.subplots(1, len(gammas), figsize=(5 * len(gammas), 5))
    if len(gammas) == 1:
        axes = [axes]

    for ax, gamma in zip(axes, gammas):
        for t, label in zip(t_values, t_labels):
            R = np.array([
                appendix_reservation_midpoint(s0, q, t, gamma, sigma, T)
                for q in q_grid
            ])
            ax.plot(q_grid, R, label=label, lw=1.5)
        ax.axhline(s0, color="black", lw=0.8, ls="--", label="Mid-price")
        ax.set_title(f"Appendix Reservation Price\nγ={gamma} (GBM, Mean-Variance)")
        ax.set_xlabel("Inventory q")
        ax.set_ylabel("R(s, q, t)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Exp 4: Appendix Reservation Price vs Inventory", y=1.02)
    plt.tight_layout()
    path = os.path.join(results_dir, "exp4_reservation_vs_inventory.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved reservation vs inventory plot → {path}")


def _plot_reservation_vs_time(
    s0: float,
    sigma: float,
    T: float,
    gammas: List[float],
    t_grid: np.ndarray,
    results_dir: str,
) -> None:
    """Plot appendix reservation price vs time for different inventory levels."""
    q_values = [-5, -2, 0, 2, 5]

    fig, axes = plt.subplots(1, len(gammas), figsize=(5 * len(gammas), 5))
    if len(gammas) == 1:
        axes = [axes]

    for ax, gamma in zip(axes, gammas):
        for q in q_values:
            R = np.array([
                appendix_reservation_midpoint(s0, q, t, gamma, sigma, T)
                for t in t_grid
            ])
            ax.plot(t_grid, R, label=f"q={q}", lw=1.5)
        ax.axhline(s0, color="black", lw=0.8, ls="--", label="Mid-price")
        ax.set_title(f"Appendix Reservation Price vs Time\nγ={gamma}")
        ax.set_xlabel("Time t")
        ax.set_ylabel("R(s, q, t)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Exp 4: Appendix Reservation Price vs Time", y=1.02)
    plt.tight_layout()
    path = os.path.join(results_dir, "exp4_reservation_vs_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved reservation vs time plot → {path}")


def _plot_model_comparison(
    s0: float,
    sigma: float,
    T: float,
    gammas: List[float],
    q_grid: np.ndarray,
    t_grid: np.ndarray,
    results_dir: str,
) -> pd.DataFrame:
    """
    Compare appendix (GBM, mean-variance) vs main model (ABM, CARA) reservation prices.
    """
    t_compare = T / 2.0
    rows = []

    fig, axes = plt.subplots(1, len(gammas), figsize=(5 * len(gammas), 5))
    if len(gammas) == 1:
        axes = [axes]

    for ax, gamma in zip(axes, gammas):
        R_appendix = np.array([
            appendix_reservation_midpoint(s0, q, t_compare, gamma, sigma, T)
            for q in q_grid
        ])
        R_main = np.array([
            main_model_reservation_price(s0, q, t_compare, gamma, sigma, T)
            for q in q_grid
        ])

        ax.plot(q_grid, R_appendix, color="steelblue", lw=2,
                label="Appendix (GBM, MV)")
        ax.plot(q_grid, R_main, color="tomato", lw=2, ls="--",
                label="Main model (ABM, CARA)")
        ax.axhline(s0, color="black", lw=0.8, ls=":", label="Mid-price")
        ax.set_title(f"Reservation Price Comparison\nγ={gamma}, t=T/2")
        ax.set_xlabel("Inventory q")
        ax.set_ylabel("Reservation price")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        for q, ra, rm in zip(q_grid, R_appendix, R_main):
            rows.append({
                "gamma": gamma,
                "q": q,
                "R_appendix": round(ra, 4),
                "R_main": round(rm, 4),
                "diff": round(ra - rm, 4),
            })

    plt.suptitle("Exp 4: Appendix vs Main Model Reservation Price (t=T/2)", y=1.02)
    plt.tight_layout()
    path = os.path.join(results_dir, "exp4_model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved model comparison plot → {path}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "exp4_comparison.csv"), index=False)
    return df


def _print_summary(
    s0: float,
    sigma: float,
    T: float,
    gammas: List[float],
    q_grid: np.ndarray,
) -> None:
    """Print a summary table of appendix reservation prices."""
    print(f"\n{'='*70}")
    print("Exp 4: Appendix Reservation Price Summary (t=T/2)")
    print(f"{'='*70}")
    t = T / 2.0
    fmt = "{:<8} {:<8} {:>14} {:>14} {:>14} {:>14}"
    print(fmt.format("gamma", "q", "R_appendix", "R_main", "R^a", "R^b"))
    print("-" * 70)
    for gamma in gammas:
        for q in [-5, -2, 0, 2, 5]:
            R = appendix_reservation_midpoint(s0, q, t, gamma, sigma, T)
            r = main_model_reservation_price(s0, q, t, gamma, sigma, T)
            Ra = appendix_reservation_ask(s0, q, t, gamma, sigma, T)
            Rb = appendix_reservation_bid(s0, q, t, gamma, sigma, T)
            print(fmt.format(f"{gamma:.3f}", f"{q}", f"{R:.4f}", f"{r:.4f}",
                             f"{Ra:.4f}", f"{Rb:.4f}"))
    print("=" * 70)


def build_results_markdown(results: Dict, gammas: List[float]) -> str:
    """Build a Markdown summary for Experiment 4."""
    lines = [
        "## Experiment 4: Appendix Extension (GBM + Mean-Variance)",
        "",
        "### Appendix Formulas",
        "",
        "**Value function:**",
        "V(x, s, q, t) = x + q·s + (γ·q²·s²/2) · (exp(σ²·(T-t)) - 1)",
        "",
        "**Reservation ask:**",
        "R^a(s, q, t) = s + ((1-2q)/2) · γ · s² · (exp(σ²·(T-t)) - 1)",
        "",
        "**Reservation bid:**",
        "R^b(s, q, t) = s + ((-1-2q)/2) · γ · s² · (exp(σ²·(T-t)) - 1)",
        "",
        "**Reservation midpoint:**",
        "R(s, q, t) = s - q · γ · s² · (exp(σ²·(T-t)) - 1)",
        "",
        "### Key Differences from Main Model",
        "",
        "| Feature | Main Model | Appendix |",
        "|---------|-----------|----------|",
        "| Price dynamics | Arithmetic BM | Geometric BM |",
        "| Objective | CARA (exponential utility) | Mean-Variance |",
        "| Reservation price adjustment | q·γ·σ²·(T-t) | q·γ·s²·(exp(σ²(T-t))-1) |",
        "| Scales with price level | No | Yes (s²) |",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    run_experiment_4()
