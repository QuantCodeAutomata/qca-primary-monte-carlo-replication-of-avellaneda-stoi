"""
Experiment 3: Microstructure-to-Intensity Calibration Mapping.

Reproduces the paper's derivation linking market-order size distributions and
price-impact laws to reduced-form execution intensities, showing when exponential
versus power-law arrival intensities arise.

Two cases are derived:
  1. Power-law order sizes + logarithmic impact → exponential intensity
  2. Power-law order sizes + power-law impact   → power-law intensity

Reference: Avellaneda & Stoikov (2008), Quantitative Finance.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Custom — Context7 found no library equivalent for AS microstructure calibration derivation

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Representative empirical alpha values cited in the paper
ALPHA_VALUES = {
    "US stocks": 1.53,
    "NASDAQ": 1.40,
    "Paris Bourse": 1.50,
}

# Representative beta values for power-law impact
BETA_VALUES = [0.5, 0.76]


# ---------------------------------------------------------------------------
# Analytical derivations
# ---------------------------------------------------------------------------

def exponential_intensity(
    delta: np.ndarray,
    Lambda: float,
    alpha: float,
    K: float,
) -> np.ndarray:
    """
    Execution intensity under power-law order sizes and logarithmic impact.

    Derivation:
      - Market orders arrive at rate Lambda.
      - Order size Q ~ power-law: P(Q > x) ∝ x^{-alpha}
      - Logarithmic impact: ΔP = (1/K) * ln(Q)  →  Q(delta) = exp(K * delta)
      - P(Q > Q(delta)) = P(Q > exp(K*delta)) ∝ exp(-alpha * K * delta)
      - lambda(delta) = Lambda * P(Q > Q(delta)) = A * exp(-k * delta)
        where A = Lambda / alpha  (paper's normalisation)
              k = alpha * K

    Parameters
    ----------
    delta  : array of quote distances (>= 0)
    Lambda : overall market-order arrival rate
    alpha  : power-law tail exponent
    K      : log-impact constant

    Returns
    -------
    lambda(delta) = A * exp(-k * delta)
    """
    A = Lambda / alpha
    k = alpha * K
    return A * np.exp(-k * delta)


def power_law_intensity(
    delta: np.ndarray,
    Lambda: float,
    alpha: float,
    beta: float,
    B_scale: float = 1.0,
) -> np.ndarray:
    """
    Execution intensity under power-law order sizes and power-law impact.

    Derivation:
      - Market orders arrive at rate Lambda.
      - Order size Q ~ power-law: P(Q > x) ∝ x^{-alpha}
      - Power-law impact: ΔP ∝ Q^beta  →  Q(delta) ∝ delta^{1/beta}
      - P(Q > Q(delta)) ∝ delta^{-alpha/beta}
      - lambda(delta) = B * delta^{-alpha/beta}

    Note: singular at delta = 0; use delta > 0.

    Parameters
    ----------
    delta   : array of quote distances (> 0, avoid zero)
    Lambda  : overall market-order arrival rate
    alpha   : power-law tail exponent
    beta    : power-law impact exponent
    B_scale : proportionality constant (normalisation)

    Returns
    -------
    lambda(delta) = B_scale * delta^{-alpha/beta}
    """
    assert np.all(delta > 0), "delta must be strictly positive for power-law intensity"
    exponent = alpha / beta
    return B_scale * delta ** (-exponent)


def derive_parameters_from_microstructure(
    Lambda: float,
    alpha: float,
    K: float,
) -> Dict[str, float]:
    """
    Compute the reduced-form exponential intensity parameters A and k from
    underlying microstructure primitives.

    Parameters
    ----------
    Lambda : overall market-order arrival rate
    alpha  : power-law tail exponent of order-size distribution
    K      : log-impact constant

    Returns
    -------
    dict with keys 'A', 'k', 'Lambda', 'alpha', 'K'
    """
    A = Lambda / alpha
    k = alpha * K
    return {"A": A, "k": k, "Lambda": Lambda, "alpha": alpha, "K": K}


# ---------------------------------------------------------------------------
# Numerical illustrations
# ---------------------------------------------------------------------------

def run_experiment_3(
    results_dir: str = RESULTS_DIR,
    delta_grid: np.ndarray | None = None,
    Lambda: float = 140.0,
    K_values: List[float] | None = None,
) -> Dict:
    """
    Run Experiment 3: microstructure calibration mapping.

    Produces:
      1. Symbolic derivation summary (printed)
      2. Parameter table for representative alpha values
      3. Plots of lambda(delta) for both model classes

    Parameters
    ----------
    results_dir  : directory to save outputs
    delta_grid   : array of quote distances for plotting
    Lambda       : overall market-order arrival rate
    K_values     : list of log-impact constants to illustrate

    Returns
    -------
    dict with derivation results and parameter tables
    """
    os.makedirs(results_dir, exist_ok=True)

    if delta_grid is None:
        delta_grid = np.linspace(0.01, 3.0, 300)

    if K_values is None:
        K_values = [0.5, 1.0, 1.5]

    # -----------------------------------------------------------------------
    # 1. Print symbolic derivation summary
    # -----------------------------------------------------------------------
    _print_derivation_summary()

    # -----------------------------------------------------------------------
    # 2. Parameter table for representative alpha values
    # -----------------------------------------------------------------------
    param_rows = []
    for market, alpha in ALPHA_VALUES.items():
        for K in K_values:
            p = derive_parameters_from_microstructure(Lambda, alpha, K)
            param_rows.append(
                {
                    "market": market,
                    "alpha": alpha,
                    "K": K,
                    "A (=Lambda/alpha)": round(p["A"], 4),
                    "k (=alpha*K)": round(p["k"], 4),
                }
            )
    param_df = pd.DataFrame(param_rows)
    print("\nParameter mapping table (exponential intensity):")
    print(param_df.to_string(index=False))
    param_df.to_csv(os.path.join(results_dir, "exp3_param_table.csv"), index=False)

    # -----------------------------------------------------------------------
    # 3. Plots
    # -----------------------------------------------------------------------
    _plot_exponential_intensities(delta_grid, Lambda, K_values, results_dir)
    _plot_power_law_intensities(delta_grid, Lambda, results_dir)
    _plot_comparison(delta_grid, Lambda, results_dir)

    return {
        "param_table": param_df,
        "alpha_values": ALPHA_VALUES,
        "beta_values": BETA_VALUES,
    }


def _print_derivation_summary() -> None:
    """Print the analytical derivation summary to stdout."""
    print("\n" + "=" * 70)
    print("Exp 3: Microstructure-to-Intensity Calibration Derivation")
    print("=" * 70)
    print("""
CASE 1: Power-law order sizes + Logarithmic impact → Exponential intensity
---------------------------------------------------------------------------
Setup:
  - Market orders arrive at overall rate Lambda
  - Order size Q ~ power-law: P(Q > x) ∝ x^{-alpha}
  - Logarithmic impact: ΔP = (1/K) * ln(Q)
    → Minimum size to reach quote distance delta: Q(delta) = exp(K * delta)

Derivation:
  P(Q > Q(delta)) = P(Q > exp(K*delta)) ∝ exp(-alpha * K * delta)
  lambda(delta) = Lambda * P(Q > Q(delta)) = A * exp(-k * delta)
  where:
    A = Lambda / alpha   (paper's normalisation)
    k = alpha * K

Interpretation:
  - Steeper decay (larger k) for larger alpha (heavier-tailed orders) or
    larger K (stronger impact)
  - A and k in the main simulation are reduced-form counterparts of
    Lambda, alpha, and K

CASE 2: Power-law order sizes + Power-law impact → Power-law intensity
-----------------------------------------------------------------------
Setup:
  - Market orders arrive at overall rate Lambda
  - Order size Q ~ power-law: P(Q > x) ∝ x^{-alpha}
  - Power-law impact: ΔP ∝ Q^beta
    → Minimum size to reach distance delta: Q(delta) ∝ delta^{1/beta}

Derivation:
  P(Q > Q(delta)) ∝ (delta^{1/beta})^{-alpha} = delta^{-alpha/beta}
  lambda(delta) = B * delta^{-alpha/beta}

Interpretation:
  - Steeper decay for larger alpha/beta ratio
  - Singular at delta = 0 (unlike exponential case)
  - Empirical beta values: 0.5 (square-root impact), 0.76 (Almgren et al.)
""")
    print("=" * 70)


def _plot_exponential_intensities(
    delta_grid: np.ndarray,
    Lambda: float,
    K_values: List[float],
    results_dir: str,
) -> None:
    """Plot exponential intensity curves for different alpha and K values."""
    fig, axes = plt.subplots(1, len(ALPHA_VALUES), figsize=(5 * len(ALPHA_VALUES), 5))

    for ax, (market, alpha) in zip(axes, ALPHA_VALUES.items()):
        for K in K_values:
            lam = exponential_intensity(delta_grid, Lambda, alpha, K)
            ax.plot(delta_grid, lam, label=f"K={K}")
        ax.set_title(f"Exponential Intensity\n{market} (α={alpha})")
        ax.set_xlabel("Quote distance δ")
        ax.set_ylabel("λ(δ)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, Lambda / alpha * 1.1)

    plt.suptitle("Exp 3: Exponential Intensity (Power-law sizes + Log impact)", y=1.02)
    plt.tight_layout()
    path = os.path.join(results_dir, "exp3_exponential_intensity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved exponential intensity plot → {path}")


def _plot_power_law_intensities(
    delta_grid: np.ndarray,
    Lambda: float,
    results_dir: str,
) -> None:
    """Plot power-law intensity curves for different alpha and beta values."""
    # Avoid delta=0 for power-law
    delta_pos = delta_grid[delta_grid > 0.05]

    fig, axes = plt.subplots(1, len(BETA_VALUES), figsize=(5 * len(BETA_VALUES), 5))
    if len(BETA_VALUES) == 1:
        axes = [axes]

    for ax, beta in zip(axes, BETA_VALUES):
        for market, alpha in ALPHA_VALUES.items():
            lam = power_law_intensity(delta_pos, Lambda, alpha, beta)
            ax.plot(delta_pos, lam, label=f"{market} (α={alpha})")
        ax.set_title(f"Power-law Intensity\nβ={beta}")
        ax.set_xlabel("Quote distance δ")
        ax.set_ylabel("λ(δ)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, min(ax.get_ylim()[1], 500))

    plt.suptitle("Exp 3: Power-law Intensity (Power-law sizes + Power-law impact)", y=1.02)
    plt.tight_layout()
    path = os.path.join(results_dir, "exp3_powerlaw_intensity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved power-law intensity plot → {path}")


def _plot_comparison(
    delta_grid: np.ndarray,
    Lambda: float,
    results_dir: str,
) -> None:
    """
    Plot side-by-side comparison of exponential vs power-law intensity
    for a representative parameter set.
    """
    alpha = ALPHA_VALUES["US stocks"]  # 1.53
    K = 1.0
    beta = 0.5
    delta_pos = delta_grid[delta_grid > 0.05]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Exponential
    lam_exp = exponential_intensity(delta_grid, Lambda, alpha, K)
    ax1.plot(delta_grid, lam_exp, color="steelblue", lw=2)
    ax1.set_title(f"Exponential Intensity\n(α={alpha}, K={K}, A={Lambda/alpha:.1f}, k={alpha*K:.2f})")
    ax1.set_xlabel("Quote distance δ")
    ax1.set_ylabel("λ(δ)")
    ax1.grid(True, alpha=0.3)
    ax1.text(0.6, 0.8, f"λ(δ) = {Lambda/alpha:.1f}·exp(-{alpha*K:.2f}·δ)",
             transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Power-law
    lam_pow = power_law_intensity(delta_pos, Lambda, alpha, beta)
    ax2.plot(delta_pos, lam_pow, color="tomato", lw=2)
    ax2.set_title(f"Power-law Intensity\n(α={alpha}, β={beta}, exponent={alpha/beta:.2f})")
    ax2.set_xlabel("Quote distance δ")
    ax2.set_ylabel("λ(δ)")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, min(lam_pow.max(), 500))
    ax2.text(0.4, 0.8, f"λ(δ) = B·δ^(-{alpha/beta:.2f})",
             transform=ax2.transAxes, fontsize=10,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle("Exp 3: Intensity Model Comparison (US stocks, α=1.53)", y=1.02)
    plt.tight_layout()
    path = os.path.join(results_dir, "exp3_intensity_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved intensity comparison plot → {path}")


def build_results_markdown(results: Dict) -> str:
    """Build a Markdown summary for Experiment 3."""
    lines = [
        "## Experiment 3: Microstructure-to-Intensity Calibration",
        "",
        "### Key Derivations",
        "",
        "**Case 1: Power-law sizes + Logarithmic impact → Exponential intensity**",
        "",
        "λ(δ) = A · exp(-k · δ)  where  A = Λ/α,  k = α·K",
        "",
        "**Case 2: Power-law sizes + Power-law impact → Power-law intensity**",
        "",
        "λ(δ) = B · δ^(-α/β)",
        "",
        "### Parameter Mapping Table",
        "",
        "| Market | α | K | A=Λ/α | k=α·K |",
        "|--------|---|---|-------|-------|",
    ]
    for _, row in results["param_table"].iterrows():
        lines.append(
            f"| {row['market']} | {row['alpha']} | {row['K']} | "
            f"{row['A (=Lambda/alpha)']:.4f} | {row['k (=alpha*K)']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    run_experiment_3()
