"""
Microstructure-to-Intensity Calibration Mapping (exp_3).

Reproduces the paper's derivation linking market-order size distributions and
price-impact laws to reduced-form execution intensities.

Shows when exponential vs. power-law arrival intensities arise from:
  - Power-law market-order size distribution
  - Logarithmic vs. power-law price impact

Custom implementation — Context7 found no library equivalent for this derivation.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Intensity models
# ---------------------------------------------------------------------------

def exponential_intensity(
    delta: np.ndarray,
    A: float,
    k: float,
) -> np.ndarray:
    """
    Exponential execution intensity (logarithmic-impact case).

    lambda(delta) = A * exp(-k * delta)

    Derived from:
      - Power-law market-order size: P(Q > x) ∝ x^{-alpha}
      - Logarithmic impact: Delta_p = (1/K) * ln(Q)  →  Q(delta) = exp(K*delta)
      - P(Q > Q(delta)) ∝ exp(-alpha*K*delta)
      - lambda(delta) = Lambda * P(Q > Q(delta)) = A * exp(-k*delta)
        where A = Lambda / alpha (paper's normalization), k = alpha * K

    Parameters
    ----------
    delta : array of quote distances (must be >= 0)
    A     : intensity scale (= Lambda / alpha)
    k     : intensity decay (= alpha * K)

    Returns
    -------
    np.ndarray : execution intensities
    """
    return A * np.exp(-k * delta)


def power_law_intensity(
    delta: np.ndarray,
    B: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    Power-law execution intensity (power-law impact case).

    lambda(delta) = B * delta^{-alpha/beta}

    Derived from:
      - Power-law market-order size: P(Q > x) ∝ x^{-alpha}
      - Power-law impact: Delta_p ∝ Q^beta  →  Q(delta) ∝ delta^{1/beta}
      - P(Q > Q(delta)) ∝ delta^{-alpha/beta}
      - lambda(delta) = Lambda * P(Q > Q(delta)) = B * delta^{-alpha/beta}

    Parameters
    ----------
    delta : array of quote distances (must be > 0; singular at delta=0)
    B     : intensity scale
    alpha : power-law tail exponent of market-order size distribution
    beta  : price-impact exponent

    Returns
    -------
    np.ndarray : execution intensities
    """
    delta = np.asarray(delta, dtype=float)
    assert np.all(delta > 0), "delta must be > 0 for power-law intensity (singular at 0)"
    return B * delta ** (-alpha / beta)


# ---------------------------------------------------------------------------
# Parameter interpretation
# ---------------------------------------------------------------------------

def interpret_exponential_params(
    Lambda: float,
    alpha: float,
    K: float,
) -> Tuple[float, float]:
    """
    Interpret exponential intensity parameters A and k in terms of
    microstructure primitives.

    A = Lambda / alpha   (paper's normalization)
    k = alpha * K

    Parameters
    ----------
    Lambda : overall market-order arrival frequency
    alpha  : power-law tail exponent of market-order size distribution
    K      : log-impact constant (Delta_p = (1/K) * ln(Q))

    Returns
    -------
    (A, k) : reduced-form intensity parameters
    """
    A = Lambda / alpha
    k = alpha * K
    return A, k


def interpret_power_law_params(
    alpha: float,
    beta: float,
) -> float:
    """
    Compute the power-law intensity exponent alpha/beta.

    Parameters
    ----------
    alpha : power-law tail exponent of market-order size distribution
    beta  : price-impact exponent

    Returns
    -------
    float : exponent of power-law intensity decay
    """
    return alpha / beta


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def exponential_intensity_sensitivity(
    delta_grid: np.ndarray,
    Lambda: float,
    alpha_values: list,
    K_values: list,
) -> dict:
    """
    Compute exponential intensity curves for different alpha and K values.

    Parameters
    ----------
    delta_grid   : array of quote distances
    Lambda       : overall market-order arrival frequency
    alpha_values : list of tail exponents to compare
    K_values     : list of log-impact constants to compare

    Returns
    -------
    dict mapping (alpha, K) → intensity array
    """
    results = {}
    for alpha in alpha_values:
        for K in K_values:
            A, k = interpret_exponential_params(Lambda, alpha, K)
            results[(alpha, K)] = exponential_intensity(delta_grid, A, k)
    return results


def power_law_intensity_sensitivity(
    delta_grid: np.ndarray,
    B: float,
    alpha_values: list,
    beta_values: list,
) -> dict:
    """
    Compute power-law intensity curves for different alpha and beta values.

    Parameters
    ----------
    delta_grid   : array of quote distances (must be > 0)
    B            : intensity scale
    alpha_values : list of tail exponents
    beta_values  : list of impact exponents

    Returns
    -------
    dict mapping (alpha, beta) → intensity array
    """
    results = {}
    for alpha in alpha_values:
        for beta in beta_values:
            results[(alpha, beta)] = power_law_intensity(delta_grid, B, alpha, beta)
    return results


# ---------------------------------------------------------------------------
# Tabulated examples
# ---------------------------------------------------------------------------

# Representative alpha values from the paper
ALPHA_US_STOCKS = 1.53
ALPHA_NASDAQ = 1.4
ALPHA_PARIS = 1.5

# Representative beta values from the paper
BETA_SQUARE_ROOT = 0.5
BETA_EMPIRICAL = 0.76


def tabulate_exponential_params(
    Lambda: float = 1.0,
    K_values: list = None,
) -> list:
    """
    Tabulate (A, k) for representative alpha values and K values.

    Parameters
    ----------
    Lambda   : overall market-order arrival frequency
    K_values : list of log-impact constants

    Returns
    -------
    list of dicts with alpha, K, A, k
    """
    if K_values is None:
        K_values = [0.5, 1.0, 1.5, 2.0]

    rows = []
    for alpha in [ALPHA_US_STOCKS, ALPHA_NASDAQ, ALPHA_PARIS]:
        for K in K_values:
            A, k = interpret_exponential_params(Lambda, alpha, K)
            rows.append({
                "alpha": alpha,
                "K": K,
                "A": round(A, 4),
                "k": round(k, 4),
            })
    return rows


def tabulate_power_law_exponents() -> list:
    """
    Tabulate alpha/beta for representative alpha and beta values.

    Returns
    -------
    list of dicts with alpha, beta, exponent
    """
    rows = []
    for alpha in [ALPHA_US_STOCKS, ALPHA_NASDAQ, ALPHA_PARIS]:
        for beta in [BETA_SQUARE_ROOT, BETA_EMPIRICAL]:
            exp = interpret_power_law_params(alpha, beta)
            rows.append({
                "alpha": alpha,
                "beta": beta,
                "exponent (alpha/beta)": round(exp, 4),
            })
    return rows
