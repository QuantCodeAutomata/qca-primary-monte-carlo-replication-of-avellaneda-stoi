"""
Tests for the diagnostic spread-ambiguity variant (src/diagnostic_variant.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.avellaneda_stoikov import ASParams, optimal_half_spread_base
from src.diagnostic_variant import (
    constant_spread_quote_distances,
    run_monte_carlo_diagnostic,
)


class TestConstantSpreadQuoteDistances:
    def test_zero_inventory_symmetric(self):
        """At q=0, delta_a = delta_b = c_gamma."""
        gamma, k = 0.1, 1.5
        c = optimal_half_spread_base(gamma, k)
        da, db = constant_spread_quote_distances(0, gamma=gamma, sigma=2.0, k=k, tau=0.5)
        assert abs(da - c) < 1e-12
        assert abs(db - c) < 1e-12

    def test_total_spread_constant(self):
        """Total spread = 2*c_gamma regardless of inventory or time."""
        gamma, k, sigma = 0.1, 1.5, 2.0
        c = optimal_half_spread_base(gamma, k)
        for q in [-5, 0, 5]:
            for tau in [0.1, 0.5, 1.0]:
                da, db = constant_spread_quote_distances(q, gamma, sigma, k, tau)
                total = da + db
                assert abs(total - 2 * c) < 1e-12, \
                    f"q={q}, tau={tau}: total spread={total:.6f}, expected={2*c:.6f}"

    def test_positive_inventory_ask_closer(self):
        """
        Positive inventory → ask closer to mid-price (delta_a < delta_b).
        Formula: delta_a = c - q*adj, delta_b = c + q*adj → delta_a < delta_b for q>0.
        """
        da, db = constant_spread_quote_distances(5, gamma=0.1, sigma=2.0, k=1.5, tau=0.5)
        assert da < db

    def test_negative_inventory_bid_closer(self):
        """
        Negative inventory → bid closer to mid-price (delta_b < delta_a).
        Formula: delta_b = c + q*adj → smaller for q<0.
        """
        da, db = constant_spread_quote_distances(-5, gamma=0.1, sigma=2.0, k=1.5, tau=0.5)
        assert db < da

    def test_formula_exact(self):
        """Verify exact formula: da = c - q*gamma*sigma^2*tau, db = c + q*gamma*sigma^2*tau."""
        q, gamma, sigma, k, tau = 3, 0.1, 2.0, 1.5, 0.5
        c = optimal_half_spread_base(gamma, k)
        da, db = constant_spread_quote_distances(q, gamma, sigma, k, tau)
        assert abs(da - (c - q * gamma * sigma ** 2 * tau)) < 1e-12
        assert abs(db - (c + q * gamma * sigma ** 2 * tau)) < 1e-12


class TestRunMonteCarloDiagnostic:
    def test_returns_correct_keys(self):
        """Diagnostic MC result contains all expected keys."""
        params = ASParams(gamma=0.1, seed=42)
        result = run_monte_carlo_diagnostic(params, "inventory")
        for key in ["profits", "inventories", "mean_profit", "std_profit",
                    "mean_inventory", "std_inventory"]:
            assert key in result

    def test_profits_array_length(self):
        """Profits array has length n_paths."""
        params = ASParams(gamma=0.1, seed=42)
        result = run_monte_carlo_diagnostic(params, "inventory")
        assert len(result["profits"]) == params.n_paths

    def test_reproducibility(self):
        """Same seed produces identical results."""
        params = ASParams(gamma=0.1, seed=42)
        r1 = run_monte_carlo_diagnostic(params, "inventory")
        r2 = run_monte_carlo_diagnostic(params, "inventory")
        np.testing.assert_array_equal(r1["profits"], r2["profits"])

    def test_constant_spread_matches_reported(self):
        """
        Diagnostic spread should equal 2/gamma*ln(1+gamma/k) (paper's table value).
        """
        gamma, k = 0.1, 1.5
        params = ASParams(gamma=gamma, k=k, seed=42)
        c = optimal_half_spread_base(gamma, k)
        reported_spread = 2 * c
        # Verify the formula matches expected value
        expected = (2.0 / gamma) * np.log(1.0 + gamma / k)
        assert abs(reported_spread - expected) < 1e-12

    def test_inventory_strategy_lower_inventory_std(self):
        """Inventory strategy should have lower inventory std than symmetric."""
        params = ASParams(gamma=0.1, seed=42)
        inv = run_monte_carlo_diagnostic(params, "inventory")
        sym = run_monte_carlo_diagnostic(params, "symmetric")
        assert inv["std_inventory"] < sym["std_inventory"]

    def test_profits_finite(self):
        """All profits should be finite."""
        params = ASParams(gamma=0.1, seed=42)
        result = run_monte_carlo_diagnostic(params, "inventory")
        assert np.all(np.isfinite(result["profits"]))
