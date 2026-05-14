"""
Tests for the appendix GBM mean-variance model (src/appendix_gbm.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.appendix_gbm import (
    value_function,
    reservation_ask_gbm,
    reservation_bid_gbm,
    reservation_midpoint_gbm,
    reservation_price_abm,
    validate_gbm_reservation_properties,
    compute_reservation_price_grid,
)


class TestValueFunction:
    def test_zero_inventory_equals_x_plus_qs(self):
        """At q=0, V = x + q*s = x."""
        V = value_function(x=0.0, s=100.0, q=0, t=0.0, T=1.0, sigma=0.2, gamma=0.1)
        assert abs(V - 0.0) < 1e-12

    def test_formula_exact(self):
        """Verify exact formula: V = x + q*s + (gamma*q^2*s^2/2)*(exp(sigma^2*(T-t))-1)."""
        x, s, q, t, T, sigma, gamma = 0.0, 100.0, 3, 0.0, 1.0, 0.2, 0.1
        expected = x + q * s + (gamma * q ** 2 * s ** 2 / 2) * (np.exp(sigma ** 2 * (T - t)) - 1)
        result = value_function(x, s, q, t, T, sigma, gamma)
        assert abs(result - expected) < 1e-10

    def test_at_terminal_time(self):
        """At t=T, V = x + q*s (no variance term)."""
        x, s, q, T, sigma, gamma = 0.0, 100.0, 5, 1.0, 0.2, 0.1
        V = value_function(x, s, q, T, T, sigma, gamma)
        expected = x + q * s
        assert abs(V - expected) < 1e-10

    def test_increases_with_x(self):
        """V increases linearly with x."""
        V1 = value_function(0.0, 100.0, 3, 0.0, 1.0, 0.2, 0.1)
        V2 = value_function(10.0, 100.0, 3, 0.0, 1.0, 0.2, 0.1)
        assert abs(V2 - V1 - 10.0) < 1e-10


class TestReservationAskGBM:
    def test_formula_exact(self):
        """Verify exact formula: R^a = s + ((1-2q)/2)*gamma*s^2*(exp(sigma^2*(T-t))-1)."""
        s, q, t, T, sigma, gamma = 100.0, 3, 0.0, 1.0, 0.2, 0.1
        expected = s + ((1 - 2 * q) / 2) * gamma * s ** 2 * (np.exp(sigma ** 2 * (T - t)) - 1)
        result = reservation_ask_gbm(s, q, t, T, sigma, gamma)
        assert abs(result - expected) < 1e-10

    def test_at_terminal_time_equals_s(self):
        """At t=T, R^a = s."""
        Ra = reservation_ask_gbm(100.0, 5, 1.0, 1.0, 0.2, 0.1)
        assert abs(Ra - 100.0) < 1e-10

    def test_zero_inventory_above_mid(self):
        """At q=0, R^a > s (ask above mid)."""
        Ra = reservation_ask_gbm(100.0, 0, 0.0, 1.0, 0.2, 0.1)
        assert Ra > 100.0


class TestReservationBidGBM:
    def test_formula_exact(self):
        """Verify exact formula: R^b = s + ((-1-2q)/2)*gamma*s^2*(exp(sigma^2*(T-t))-1)."""
        s, q, t, T, sigma, gamma = 100.0, 3, 0.0, 1.0, 0.2, 0.1
        expected = s + ((-1 - 2 * q) / 2) * gamma * s ** 2 * (np.exp(sigma ** 2 * (T - t)) - 1)
        result = reservation_bid_gbm(s, q, t, T, sigma, gamma)
        assert abs(result - expected) < 1e-10

    def test_at_terminal_time_equals_s(self):
        """At t=T, R^b = s."""
        Rb = reservation_bid_gbm(100.0, 5, 1.0, 1.0, 0.2, 0.1)
        assert abs(Rb - 100.0) < 1e-10

    def test_zero_inventory_below_mid(self):
        """At q=0, R^b < s (bid below mid)."""
        Rb = reservation_bid_gbm(100.0, 0, 0.0, 1.0, 0.2, 0.1)
        assert Rb < 100.0

    def test_ask_above_bid(self):
        """R^a > R^b always."""
        for q in [-5, 0, 5]:
            Ra = reservation_ask_gbm(100.0, q, 0.0, 1.0, 0.2, 0.1)
            Rb = reservation_bid_gbm(100.0, q, 0.0, 1.0, 0.2, 0.1)
            assert Ra > Rb, f"q={q}: R^a={Ra:.4f} should be > R^b={Rb:.4f}"


class TestReservationMidpointGBM:
    def test_midpoint_of_ask_and_bid(self):
        """R = (R^a + R^b) / 2."""
        s, q, t, T, sigma, gamma = 100.0, 3, 0.0, 1.0, 0.2, 0.1
        Ra = reservation_ask_gbm(s, q, t, T, sigma, gamma)
        Rb = reservation_bid_gbm(s, q, t, T, sigma, gamma)
        R = reservation_midpoint_gbm(s, q, t, T, sigma, gamma)
        assert abs(R - (Ra + Rb) / 2) < 1e-10

    def test_formula_exact(self):
        """Verify exact formula: R = s - q*gamma*s^2*(exp(sigma^2*(T-t))-1)."""
        s, q, t, T, sigma, gamma = 100.0, 3, 0.0, 1.0, 0.2, 0.1
        expected = s - q * gamma * s ** 2 * (np.exp(sigma ** 2 * (T - t)) - 1)
        result = reservation_midpoint_gbm(s, q, t, T, sigma, gamma)
        assert abs(result - expected) < 1e-10

    def test_zero_inventory_equals_s(self):
        """At q=0, R = s."""
        R = reservation_midpoint_gbm(100.0, 0, 0.0, 1.0, 0.2, 0.1)
        assert abs(R - 100.0) < 1e-10

    def test_positive_inventory_below_s(self):
        """Positive inventory → R < s."""
        R = reservation_midpoint_gbm(100.0, 5, 0.0, 1.0, 0.2, 0.1)
        assert R < 100.0

    def test_negative_inventory_above_s(self):
        """Negative inventory → R > s."""
        R = reservation_midpoint_gbm(100.0, -5, 0.0, 1.0, 0.2, 0.1)
        assert R > 100.0

    def test_vanishes_at_terminal_time(self):
        """At t=T, R = s."""
        R = reservation_midpoint_gbm(100.0, 10, 1.0, 1.0, 0.2, 0.1)
        assert abs(R - 100.0) < 1e-10

    def test_scales_with_s_squared(self):
        """Adjustment scales with s^2 (GBM property)."""
        q, t, T, sigma, gamma = 3, 0.0, 1.0, 0.2, 0.1
        factor = np.exp(sigma ** 2 * T) - 1
        R1 = reservation_midpoint_gbm(100.0, q, t, T, sigma, gamma)
        R2 = reservation_midpoint_gbm(200.0, q, t, T, sigma, gamma)
        adj1 = 100.0 - R1
        adj2 = 200.0 - R2
        # adj2 / adj1 should equal (200/100)^2 = 4
        assert abs(adj2 / adj1 - 4.0) < 1e-8


class TestReservationPriceABM:
    def test_formula_exact(self):
        """Verify exact formula: r = s - q*gamma*sigma^2*(T-t)."""
        s, q, t, T, sigma, gamma = 100.0, 3, 0.0, 1.0, 2.0, 0.1
        expected = s - q * gamma * sigma ** 2 * (T - t)
        result = reservation_price_abm(s, q, t, T, sigma, gamma)
        assert abs(result - expected) < 1e-12

    def test_zero_inventory_equals_s(self):
        """At q=0, r = s."""
        r = reservation_price_abm(100.0, 0, 0.0, 1.0, 2.0, 0.1)
        assert abs(r - 100.0) < 1e-12


class TestValidateGBMProperties:
    def test_passes_for_valid_params(self):
        """Validation passes for standard parameters."""
        validate_gbm_reservation_properties(s=100.0, sigma=0.2, T=1.0, gamma=0.1)

    def test_passes_for_all_gamma_values(self):
        """Validation passes for all paper gamma values."""
        for gamma in [0.01, 0.1, 0.5]:
            validate_gbm_reservation_properties(s=100.0, sigma=0.2, T=1.0, gamma=gamma)


class TestComputeReservationPriceGrid:
    def test_returns_dataframe(self):
        """Returns a DataFrame."""
        import pandas as pd
        df = compute_reservation_price_grid(
            s=100.0, sigma=0.2, T=1.0, gamma=0.1,
            q_values=[-2, 0, 2], t_values=[0.0, 0.5]
        )
        assert isinstance(df, pd.DataFrame)

    def test_correct_shape(self):
        """DataFrame has correct number of rows."""
        df = compute_reservation_price_grid(
            s=100.0, sigma=0.2, T=1.0, gamma=0.1,
            q_values=[-2, 0, 2], t_values=[0.0, 0.5]
        )
        assert len(df) == 6  # 3 q values × 2 t values

    def test_zero_inventory_R_equals_s(self):
        """At q=0, R_gbm = s."""
        df = compute_reservation_price_grid(
            s=100.0, sigma=0.2, T=1.0, gamma=0.1,
            q_values=[0], t_values=[0.0, 0.5]
        )
        for _, row in df.iterrows():
            assert abs(row["R_gbm"] - 100.0) < 1e-10
