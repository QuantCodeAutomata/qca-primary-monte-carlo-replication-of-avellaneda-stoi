"""
Appendix Extension: Mean-Variance Market Making Under Geometric Brownian Motion (exp_4).

Reproduces the paper's appendix model using GBM price dynamics and a mean-variance
objective, as a separate theoretical extension from the main CARA/ABM model.

Reference:
    Avellaneda & Stoikov (2008), Appendix — Mean-Variance formulation under GBM.

Custom implementation — Context7 found no library equivalent for these appendix formulas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Value function
# ---------------------------------------------------------------------------

def value_function(
    x: float,
    s: float,
    q: float,
    t: float,
    T: float,
    sigma: float,
    gamma: float,
) -> float:
    """
    Appendix mean-variance value function under GBM.

    V(x, s, q, t) = x + q*s + (gamma*q^2*s^2 / 2) * (exp(sigma^2*(T-t)) - 1)

    Objective: E_t[(x + q*S_T)] - (gamma/2) * Var_t[q*S_T]

    Parameters
    ----------
    x     : current cash
    s     : current mid-price
    q     : current inventory
    t     : current time
    T     : horizon
    sigma : GBM volatility
    gamma : risk-aversion coefficient

    Returns
    -------
    float : value function
    """
    tau = T - t
    return x + q * s + (gamma * q ** 2 * s ** 2 / 2.0) * (np.exp(sigma ** 2 * tau) - 1.0)


# ---------------------------------------------------------------------------
# Reservation prices
# ---------------------------------------------------------------------------

def reservation_ask_gbm(
    s: float,
    q: float,
    t: float,
    T: float,
    sigma: float,
    gamma: float,
) -> float:
    """
    Appendix reservation ask price under GBM and mean-variance objective.

    R^a(s, q, t) = s + ((1 - 2q) / 2) * gamma * s^2 * (exp(sigma^2*(T-t)) - 1)

    Parameters
    ----------
    s     : current mid-price
    q     : current inventory
    t     : current time
    T     : horizon
    sigma : GBM volatility
    gamma : risk-aversion coefficient

    Returns
    -------
    float : reservation ask price
    """
    tau = T - t
    factor = (np.exp(sigma ** 2 * tau) - 1.0)
    return s + ((1.0 - 2.0 * q) / 2.0) * gamma * s ** 2 * factor


def reservation_bid_gbm(
    s: float,
    q: float,
    t: float,
    T: float,
    sigma: float,
    gamma: float,
) -> float:
    """
    Appendix reservation bid price under GBM and mean-variance objective.

    R^b(s, q, t) = s + ((-1 - 2q) / 2) * gamma * s^2 * (exp(sigma^2*(T-t)) - 1)

    Parameters
    ----------
    s     : current mid-price
    q     : current inventory
    t     : current time
    T     : horizon
    sigma : GBM volatility
    gamma : risk-aversion coefficient

    Returns
    -------
    float : reservation bid price
    """
    tau = T - t
    factor = (np.exp(sigma ** 2 * tau) - 1.0)
    return s + ((-1.0 - 2.0 * q) / 2.0) * gamma * s ** 2 * factor


def reservation_midpoint_gbm(
    s: float,
    q: float,
    t: float,
    T: float,
    sigma: float,
    gamma: float,
) -> float:
    """
    Appendix reservation price midpoint under GBM.

    R(s, q, t) = (R^a + R^b) / 2 = s - q * gamma * s^2 * (exp(sigma^2*(T-t)) - 1)

    Parameters
    ----------
    s     : current mid-price
    q     : current inventory
    t     : current time
    T     : horizon
    sigma : GBM volatility
    gamma : risk-aversion coefficient

    Returns
    -------
    float : reservation price midpoint
    """
    tau = T - t
    factor = (np.exp(sigma ** 2 * tau) - 1.0)
    return s - q * gamma * s ** 2 * factor


# ---------------------------------------------------------------------------
# Main-model reservation price (for cross-model comparison)
# ---------------------------------------------------------------------------

def reservation_price_abm(
    s: float,
    q: float,
    t: float,
    T: float,
    sigma: float,
    gamma: float,
) -> float:
    """
    Main-model (ABM + CARA) reservation price for cross-model comparison.

    r(s, q, t) = s - q * gamma * sigma^2 * (T - t)

    Parameters
    ----------
    s     : current mid-price
    q     : current inventory
    t     : current time
    T     : horizon
    sigma : ABM volatility
    gamma : risk-aversion coefficient

    Returns
    -------
    float : reservation price
    """
    tau = T - t
    return s - q * gamma * sigma ** 2 * tau


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_gbm_reservation_properties(
    s: float = 100.0,
    sigma: float = 0.2,
    T: float = 1.0,
    gamma: float = 0.1,
) -> None:
    """
    Validate qualitative properties of the GBM reservation price:
    1. Positive inventory → R < s
    2. Negative inventory → R > s
    3. Adjustment shrinks as t → T (tau → 0)
    4. R^a and R^b are symmetric around R
    """
    t_mid = T / 2.0

    # Property 1: positive inventory → R < s
    R_pos = reservation_midpoint_gbm(s, 5, t_mid, T, sigma, gamma)
    assert R_pos < s, f"Positive inventory should give R < s, got R={R_pos}, s={s}"

    # Property 2: negative inventory → R > s
    R_neg = reservation_midpoint_gbm(s, -5, t_mid, T, sigma, gamma)
    assert R_neg > s, f"Negative inventory should give R > s, got R={R_neg}, s={s}"

    # Property 3: adjustment shrinks as tau → 0
    R_terminal = reservation_midpoint_gbm(s, 10, T - 1e-10, T, sigma, gamma)
    assert abs(R_terminal - s) < 1e-6, \
        f"At T, reservation price should approach s, got {R_terminal}"

    # Property 4: R^a and R^b symmetric around R
    Ra = reservation_ask_gbm(s, 3, t_mid, T, sigma, gamma)
    Rb = reservation_bid_gbm(s, 3, t_mid, T, sigma, gamma)
    R = reservation_midpoint_gbm(s, 3, t_mid, T, sigma, gamma)
    assert abs((Ra + Rb) / 2.0 - R) < 1e-12, \
        f"(R^a + R^b)/2 should equal R, got {(Ra+Rb)/2}, R={R}"


# ---------------------------------------------------------------------------
# Numerical illustration grids
# ---------------------------------------------------------------------------

def compute_reservation_price_grid(
    s: float,
    sigma: float,
    T: float,
    gamma: float,
    q_values: list,
    t_values: list,
) -> pd.DataFrame:
    """
    Compute reservation prices over a grid of (q, t) values for both GBM and ABM models.

    Parameters
    ----------
    s        : current mid-price
    sigma    : volatility (GBM dimensionless; ABM uses sigma=2 internally for comparison)
    T        : horizon
    gamma    : risk-aversion coefficient
    q_values : list of inventory values
    t_values : list of time values

    Returns
    -------
    pd.DataFrame with columns: q, t, R_gbm, Ra_gbm, Rb_gbm, r_abm
    """
    rows = []
    for q in q_values:
        for t in t_values:
            R = reservation_midpoint_gbm(s, q, t, T, sigma, gamma)
            Ra = reservation_ask_gbm(s, q, t, T, sigma, gamma)
            Rb = reservation_bid_gbm(s, q, t, T, sigma, gamma)
            r = reservation_price_abm(s, q, t, T, sigma, gamma)
            rows.append({
                "q": q,
                "t": t,
                "R_gbm": R,
                "Ra_gbm": Ra,
                "Rb_gbm": Rb,
                "r_abm": r,
            })
    return pd.DataFrame(rows)
