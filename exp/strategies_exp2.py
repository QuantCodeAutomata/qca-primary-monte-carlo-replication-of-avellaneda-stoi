"""Experiment 2 strategies: Constant spread (table-faithful interpretation).

Alternative interpretation where the spread is constant, equal to the
limit as T-t -> 0 of the full-spread formula:
    spread_const = (2/gamma) * ln(1 + gamma/k)

Strategy A: inventory_const_spread
  - Reservation price center: r_t = S_t - q_t*gamma*sigma^2*(T-t)
  - Symmetric quotes around r_t with constant half-spread h:
      p_a = r_t + h,  p_b = r_t - h
  - Quote distances from mid:
      delta_a = -q*gamma*sigma^2*(T-t) + h
      delta_b =  q*gamma*sigma^2*(T-t) + h
  - Total spread = 2h = spread_const (constant, time-independent)

Strategy B: symmetric_const_spread
  - Center: mid-price S_t
  - delta_a = delta_b = h (constant, ignores inventory)
"""
from __future__ import annotations
import numpy as np
from src.config import SimParams
from typing import Tuple


def compute_half_spread(gamma: float, k: float) -> float:
    """Compute constant half-spread: h = (1/gamma) * ln(1 + gamma/k).

    This is the half of the constant spread that matches the paper's
    table-faithful interpretation.
    """
    return np.log1p(gamma / k) / gamma


def inventory_const_spread(
    S: float, q: int, t: float, gamma: float, params: SimParams
) -> Tuple[float, float]:
    """Inventory-based quoting with constant spread (Exp 2, Strategy A).

    Centers quotes around reservation price r_t with constant half-spread h.
    Quote distances from mid-price S_t:
        delta_a = p_a - S_t = (r_t + h) - S_t = -q*gamma*sigma^2*(T-t) + h
        delta_b = S_t - p_b = S_t - (r_t - h) =  q*gamma*sigma^2*(T-t) + h

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
    """
    sigma = params.sigma
    k = params.k
    T = params.T
    tau = T - t

    h = compute_half_spread(gamma, k)
    inventory_skew = q * gamma * sigma ** 2 * tau

    delta_a = -inventory_skew + h  # equivalent to (r_t + h) - S_t
    delta_b = inventory_skew + h   # equivalent to S_t - (r_t - h)

    return delta_a, delta_b


def symmetric_const_spread(
    S: float, q: int, t: float, gamma: float, params: SimParams
) -> Tuple[float, float]:
    """Symmetric benchmark with constant spread (Exp 2, Strategy B).

    Centers quotes around mid-price with constant half-spread h, ignoring inventory.
        delta_a = delta_b = h

    Parameters
    ----------
    S : float  Current mid-price (unused)
    q : int    Current inventory (ignored)
    t : float  Current time (unused)
    gamma : float  Risk-aversion coefficient
    params : SimParams  Simulation parameters

    Returns
    -------
    (delta_a, delta_b) : tuple of float, both equal to h
    """
    h = compute_half_spread(gamma, params.k)
    return h, h
