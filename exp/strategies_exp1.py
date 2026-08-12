"""Experiment 1 strategies: Full time-varying spread (Avellaneda-Stoikov eq. 3.18).

Strategy A: inventory_full_spread
  - Centers quotes around reservation price r_t = S_t - q_t*gamma*sigma^2*(T-t)
  - Uses the paper's asymmetric individual quote distances:
      delta_a = 0.5*(1 - 2*q)*gamma*sigma^2*(T-t) + (1/gamma)*ln(1 + gamma/k)
      delta_b = 0.5*(1 + 2*q)*gamma*sigma^2*(T-t) + (1/gamma)*ln(1 + gamma/k)
  - Total spread = gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/k)

Strategy B: symmetric_full_spread
  - Centers quotes around mid-price S_t
  - Uses the same total spread as Strategy A at time t, split equally:
      half_spread = 0.5*(gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/k))
      delta_a = delta_b = half_spread
"""
from __future__ import annotations
import numpy as np
from src.config import SimParams
from typing import Tuple


def inventory_full_spread(
    S: float, q: int, t: float, gamma: float, params: SimParams
) -> Tuple[float, float]:
    """Inventory-aware quoting with full time-varying spread (Exp 1, Strategy A).

    Individual quote distances from Avellaneda-Stoikov methodology step 2:
        delta_a = 0.5*(1 - 2*q)*gamma*sigma^2*(T-t) + (1/gamma)*ln(1 + gamma/k)
        delta_b = 0.5*(1 + 2*q)*gamma*sigma^2*(T-t) + (1/gamma)*ln(1 + gamma/k)

    These are equivalent to centering around the reservation price:
        r = S - q*gamma*sigma^2*(T-t)
        delta_a = p_a - S = (r - S) + half_spread + half_inventory_term

    Parameters
    ----------
    S : float  Current mid-price
    q : int    Current inventory
    t : float  Current time
    gamma : float  Risk-aversion coefficient
    params : SimParams  Simulation parameters

    Returns
    -------
    (delta_a, delta_b) : tuple of float
        Ask and bid distances from mid-price.
    """
    sigma = params.sigma
    k = params.k
    T = params.T
    tau = T - t  # time to maturity

    # Logarithm term: (1/gamma)*ln(1 + gamma/k)
    # Use log1p for numerical stability: log(1 + gamma/k) = log1p(gamma/k)
    log_term = np.log1p(gamma / k) / gamma

    # Full time-varying individual distances (paper methodology step 2)
    delta_a = 0.5 * (1.0 - 2.0 * q) * gamma * sigma ** 2 * tau + log_term
    delta_b = 0.5 * (1.0 + 2.0 * q) * gamma * sigma ** 2 * tau + log_term

    return delta_a, delta_b


def symmetric_full_spread(
    S: float, q: int, t: float, gamma: float, params: SimParams
) -> Tuple[float, float]:
    """Symmetric benchmark with full time-varying spread (Exp 1, Strategy B).

    Centers quotes around the mid-price S_t with the same total spread as
    Strategy A, split equally:
        half_spread = 0.5 * (gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/k))
        delta_a = delta_b = half_spread

    Parameters
    ----------
    S : float  Current mid-price (unused for quote computation but kept for interface)
    q : int    Current inventory (ignored)
    t : float  Current time
    gamma : float  Risk-aversion coefficient
    params : SimParams  Simulation parameters

    Returns
    -------
    (delta_a, delta_b) : tuple of float, both equal to half_spread
    """
    sigma = params.sigma
    k = params.k
    T = params.T
    tau = T - t

    # Full time-varying total spread then halved
    full_spread = gamma * sigma ** 2 * tau + (2.0 / gamma) * np.log1p(gamma / k)
    half_spread = 0.5 * full_spread

    return half_spread, half_spread


def compute_reservation_price(S: float, q: int, t: float, gamma: float, params: SimParams) -> float:
    """Compute the reservation price r_t = S_t - q_t*gamma*sigma^2*(T-t)."""
    return S - q * gamma * params.sigma ** 2 * (params.T - t)
