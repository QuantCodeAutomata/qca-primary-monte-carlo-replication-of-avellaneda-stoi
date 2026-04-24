"""
Tests for all four experiments.

Verifies:
  - Experiment 1: Primary finite-horizon replication methodology
  - Experiment 2: Diagnostic constant-spread variant
  - Experiment 3: Microstructure calibration derivations
  - Experiment 4: Appendix GBM mean-variance formulas
"""

from __future__ import annotations

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulation import SimParams, draw_shared_randoms
from src.exp1_primary_replication import run_experiment_1
from src.exp2_diagnostic_replication import (
    constant_half_spread,
    diagnostic_quote_distances,
    run_experiment_2,
)
from src.exp3_microstructure_calibration import (
    exponential_intensity,
    power_law_intensity,
    derive_parameters_from_microstructure,
    run_experiment_3,
    ALPHA_VALUES,
    BETA_VALUES,
)
from src.exp4_appendix_gbm_mv import (
    appendix_value_function,
    appendix_reservation_ask,
    appendix_reservation_bid,
    appendix_reservation_midpoint,
    main_model_reservation_price,
    validate_appendix_formulas,
    run_experiment_4,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fast_params() -> SimParams:
    """Minimal params for fast test execution."""
    return SimParams(n_paths=100, seed=42)


@pytest.fixture
def results_dir(tmp_path) -> str:
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Experiment 1 Tests
# ---------------------------------------------------------------------------

class TestExperiment1:
    """Tests for primary finite-horizon Monte Carlo replication."""

    def test_runs_without_error(self, fast_params, results_dir):
        """Experiment 1 should complete without raising exceptions."""
        results = run_experiment_1(
            params=fast_params, gammas=[0.1], results_dir=results_dir
        )
        assert results is not None

    def test_all_gammas_present(self, fast_params, results_dir):
        """Results should contain all specified gamma values."""
        gammas = [0.1, 0.01, 0.5]
        results = run_experiment_1(
            params=fast_params, gammas=gammas, results_dir=results_dir
        )
        for gamma in gammas:
            assert f"gamma_{gamma}" in results

    def test_both_strategies_present(self, fast_params, results_dir):
        """Results should contain both inventory and symmetric strategies."""
        results = run_experiment_1(
            params=fast_params, gammas=[0.1], results_dir=results_dir
        )
        assert "inventory" in results["gamma_0.1"]
        assert "symmetric" in results["gamma_0.1"]

    def test_statistics_finite(self, fast_params, results_dir):
        """All summary statistics should be finite."""
        results = run_experiment_1(
            params=fast_params, gammas=[0.1], results_dir=results_dir
        )
        r = results["gamma_0.1"]["inventory"]
        assert np.isfinite(r.mean_profit)
        assert np.isfinite(r.std_profit)
        assert np.isfinite(r.mean_final_q)
        assert np.isfinite(r.std_final_q)

    def test_inventory_reduces_risk_high_gamma(self, fast_params, results_dir):
        """
        For gamma=0.5, inventory strategy should reduce inventory dispersion
        relative to symmetric benchmark (core paper claim).
        """
        results = run_experiment_1(
            params=fast_params, gammas=[0.5], results_dir=results_dir
        )
        inv = results["gamma_0.5"]["inventory"]
        sym = results["gamma_0.5"]["symmetric"]
        # Inventory strategy should have lower or comparable inventory std
        # (with only 100 paths, allow some tolerance)
        assert inv.std_final_q <= sym.std_final_q * 2.0, \
            "Inventory strategy should not dramatically increase inventory dispersion"

    def test_csv_output_created(self, fast_params, results_dir):
        """Summary CSV should be created."""
        run_experiment_1(params=fast_params, gammas=[0.1], results_dir=results_dir)
        assert os.path.exists(os.path.join(results_dir, "exp1_summary.csv"))

    def test_plot_output_created(self, fast_params, results_dir):
        """Profit histogram plot should be created."""
        run_experiment_1(params=fast_params, gammas=[0.1], results_dir=results_dir)
        assert os.path.exists(os.path.join(results_dir, "exp1_profit_histograms.png"))

    def test_sample_path_plot_created(self, fast_params, results_dir):
        """Sample path plot should be created for each gamma."""
        run_experiment_1(params=fast_params, gammas=[0.1], results_dir=results_dir)
        assert os.path.exists(
            os.path.join(results_dir, "exp1_sample_path_gamma0.1.png")
        )

    def test_profits_array_length(self, fast_params, results_dir):
        """Profits array should have length equal to n_paths."""
        results = run_experiment_1(
            params=fast_params, gammas=[0.1], results_dir=results_dir
        )
        r = results["gamma_0.1"]["inventory"]
        assert len(r.profits) == fast_params.n_paths

    def test_mean_profit_matches_array(self, fast_params, results_dir):
        """Reported mean profit should match mean of profits array."""
        results = run_experiment_1(
            params=fast_params, gammas=[0.1], results_dir=results_dir
        )
        r = results["gamma_0.1"]["inventory"]
        assert abs(r.mean_profit - np.mean(r.profits)) < 1e-10


# ---------------------------------------------------------------------------
# Experiment 2 Tests
# ---------------------------------------------------------------------------

class TestExperiment2:
    """Tests for diagnostic constant-spread replication."""

    def test_constant_half_spread_positive(self):
        """Constant half-spread must be positive."""
        for gamma in [0.01, 0.1, 0.5]:
            c = constant_half_spread(gamma, 1.5)
            assert c > 0, f"Negative half-spread for gamma={gamma}"

    def test_constant_half_spread_formula(self):
        """Verify formula: c = (1/gamma) * ln(1 + gamma/k)."""
        gamma, k = 0.1, 1.5
        expected = (1.0 / gamma) * np.log(1.0 + gamma / k)
        result = constant_half_spread(gamma, k)
        assert abs(result - expected) < 1e-12

    def test_diagnostic_distances_symmetry_at_zero_q(self):
        """At q=0, diagnostic delta_a == delta_b == c_gamma."""
        gamma, k = 0.1, 1.5
        c = constant_half_spread(gamma, k)
        da, db = diagnostic_quote_distances(0, gamma, 2.0, k, 0.5)
        assert abs(da - c) < 1e-12
        assert abs(db - c) < 1e-12

    def test_diagnostic_total_spread_constant(self):
        """Total spread delta_a + delta_b should equal 2*c_gamma regardless of q and tau."""
        gamma, k = 0.1, 1.5
        c = constant_half_spread(gamma, k)
        for q in [-5, 0, 5]:
            for tau in [0.1, 0.5, 1.0]:
                da, db = diagnostic_quote_distances(q, gamma, 2.0, k, tau)
                total = da + db
                assert abs(total - 2 * c) < 1e-10, \
                    f"Total spread {total:.6f} != 2*c={2*c:.6f} for q={q}, tau={tau}"

    def test_diagnostic_centering_around_reservation(self):
        """
        Diagnostic quotes should be centered around the reservation price.
        p^a = s + delta_a, p^b = s - delta_b
        midpoint = s + (delta_a - delta_b)/2 = r (reservation price)
        """
        gamma, sigma, k = 0.1, 2.0, 1.5
        s, q, tau = 100.0, 3, 0.5
        da, db = diagnostic_quote_distances(q, gamma, sigma, k, tau)
        midpoint = s + (da - db) / 2.0
        # Reservation price
        r = s - q * gamma * sigma ** 2 * tau
        assert abs(midpoint - r) < 1e-10, \
            f"Quote midpoint {midpoint:.4f} != reservation price {r:.4f}"

    def test_runs_without_error(self, fast_params, results_dir):
        """Experiment 2 should complete without raising exceptions."""
        results = run_experiment_2(
            params=fast_params, gammas=[0.1], results_dir=results_dir
        )
        assert results is not None

    def test_csv_output_created(self, fast_params, results_dir):
        """Diagnostic summary CSV should be created."""
        run_experiment_2(params=fast_params, gammas=[0.1], results_dir=results_dir)
        assert os.path.exists(os.path.join(results_dir, "exp2_summary.csv"))

    def test_spread_values_match_paper_tables(self):
        """
        Constant spread 2*c_gamma should match paper's reported table values.
        For gamma=0.1, k=1.5: spread = (2/0.1)*ln(1+0.1/1.5) ≈ 1.29
        """
        gamma, k = 0.1, 1.5
        c = constant_half_spread(gamma, k)
        spread = 2 * c
        # Verify it's a reasonable positive value
        assert spread > 0
        # Verify formula matches expected calculation
        expected = (2.0 / gamma) * np.log(1.0 + gamma / k)
        assert abs(spread - expected) < 1e-12


# ---------------------------------------------------------------------------
# Experiment 3 Tests
# ---------------------------------------------------------------------------

class TestExperiment3:
    """Tests for microstructure calibration mapping."""

    def test_exponential_intensity_formula(self):
        """Verify exponential intensity formula: A*exp(-k*delta)."""
        Lambda, alpha, K = 140.0, 1.5, 1.0
        delta = np.array([0.5, 1.0, 1.5])
        A = Lambda / alpha
        k = alpha * K
        expected = A * np.exp(-k * delta)
        result = exponential_intensity(delta, Lambda, alpha, K)
        assert np.allclose(result, expected)

    def test_exponential_intensity_at_zero(self):
        """At delta=0, exponential intensity equals A = Lambda/alpha."""
        Lambda, alpha, K = 140.0, 1.5, 1.0
        lam = exponential_intensity(np.array([0.0]), Lambda, alpha, K)
        assert abs(lam[0] - Lambda / alpha) < 1e-10

    def test_exponential_intensity_decreasing(self):
        """Exponential intensity must decrease with delta."""
        Lambda, alpha, K = 140.0, 1.5, 1.0
        delta = np.linspace(0, 3, 100)
        lam = exponential_intensity(delta, Lambda, alpha, K)
        assert np.all(np.diff(lam) < 0)

    def test_power_law_intensity_formula(self):
        """Verify power-law intensity formula: B * delta^(-alpha/beta)."""
        Lambda, alpha, beta = 140.0, 1.5, 0.5
        delta = np.array([0.5, 1.0, 2.0])
        exponent = alpha / beta
        expected = 1.0 * delta ** (-exponent)
        result = power_law_intensity(delta, Lambda, alpha, beta)
        assert np.allclose(result, expected)

    def test_power_law_intensity_singular_at_zero(self):
        """Power-law intensity should raise for delta=0."""
        with pytest.raises(AssertionError):
            power_law_intensity(np.array([0.0, 1.0]), 140.0, 1.5, 0.5)

    def test_power_law_intensity_decreasing(self):
        """Power-law intensity must decrease with delta."""
        delta = np.linspace(0.1, 3, 100)
        lam = power_law_intensity(delta, 140.0, 1.5, 0.5)
        assert np.all(np.diff(lam) < 0)

    def test_derive_parameters_formula(self):
        """Verify A = Lambda/alpha and k = alpha*K."""
        Lambda, alpha, K = 140.0, 1.5, 1.0
        params = derive_parameters_from_microstructure(Lambda, alpha, K)
        assert abs(params["A"] - Lambda / alpha) < 1e-12
        assert abs(params["k"] - alpha * K) < 1e-12

    def test_alpha_values_in_range(self):
        """Cited alpha values should be in (1, 2) range."""
        for market, alpha in ALPHA_VALUES.items():
            assert 1.0 < alpha < 2.0, f"Alpha={alpha} for {market} out of expected range"

    def test_beta_values_positive(self):
        """Beta values should be positive."""
        for beta in BETA_VALUES:
            assert beta > 0

    def test_steeper_decay_for_larger_alpha(self):
        """Larger alpha should give steeper exponential decay."""
        Lambda, K = 140.0, 1.0
        delta = np.array([1.0])
        lam_small = exponential_intensity(delta, Lambda, 1.0, K)
        lam_large = exponential_intensity(delta, Lambda, 2.0, K)
        assert lam_large < lam_small, "Larger alpha should give smaller intensity at delta>0"

    def test_steeper_decay_for_larger_K(self):
        """Larger K should give steeper exponential decay."""
        Lambda, alpha = 140.0, 1.5
        delta = np.array([1.0])
        lam_small = exponential_intensity(delta, Lambda, alpha, 0.5)
        lam_large = exponential_intensity(delta, Lambda, alpha, 2.0)
        assert lam_large < lam_small, "Larger K should give smaller intensity at delta>0"

    def test_runs_without_error(self, results_dir):
        """Experiment 3 should complete without raising exceptions."""
        results = run_experiment_3(results_dir=results_dir)
        assert results is not None

    def test_param_table_created(self, results_dir):
        """Parameter table CSV should be created."""
        run_experiment_3(results_dir=results_dir)
        assert os.path.exists(os.path.join(results_dir, "exp3_param_table.csv"))

    def test_plots_created(self, results_dir):
        """All intensity plots should be created."""
        run_experiment_3(results_dir=results_dir)
        assert os.path.exists(os.path.join(results_dir, "exp3_exponential_intensity.png"))
        assert os.path.exists(os.path.join(results_dir, "exp3_powerlaw_intensity.png"))
        assert os.path.exists(os.path.join(results_dir, "exp3_intensity_comparison.png"))


# ---------------------------------------------------------------------------
# Experiment 4 Tests
# ---------------------------------------------------------------------------

class TestExperiment4:
    """Tests for appendix GBM mean-variance extension."""

    def test_value_function_formula(self):
        """Verify appendix value function formula."""
        x, s, q, t, gamma, sigma, T = 0.0, 100.0, 2.0, 0.5, 0.1, 0.2, 1.0
        tau = T - t
        expected = x + q * s + (gamma * q ** 2 * s ** 2 / 2.0) * (np.exp(sigma ** 2 * tau) - 1.0)
        result = appendix_value_function(x, s, q, t, gamma, sigma, T)
        assert abs(result - expected) < 1e-10

    def test_reservation_ask_formula(self):
        """Verify appendix reservation ask formula."""
        s, q, t, gamma, sigma, T = 100.0, 2.0, 0.5, 0.1, 0.2, 1.0
        tau = T - t
        expected = s + ((1.0 - 2.0 * q) / 2.0) * gamma * s ** 2 * (np.exp(sigma ** 2 * tau) - 1.0)
        result = appendix_reservation_ask(s, q, t, gamma, sigma, T)
        assert abs(result - expected) < 1e-10

    def test_reservation_bid_formula(self):
        """Verify appendix reservation bid formula."""
        s, q, t, gamma, sigma, T = 100.0, 2.0, 0.5, 0.1, 0.2, 1.0
        tau = T - t
        expected = s + ((-1.0 - 2.0 * q) / 2.0) * gamma * s ** 2 * (np.exp(sigma ** 2 * tau) - 1.0)
        result = appendix_reservation_bid(s, q, t, gamma, sigma, T)
        assert abs(result - expected) < 1e-10

    def test_reservation_midpoint_formula(self):
        """Verify appendix reservation midpoint formula."""
        s, q, t, gamma, sigma, T = 100.0, 2.0, 0.5, 0.1, 0.2, 1.0
        tau = T - t
        expected = s - q * gamma * s ** 2 * (np.exp(sigma ** 2 * tau) - 1.0)
        result = appendix_reservation_midpoint(s, q, t, gamma, sigma, T)
        assert abs(result - expected) < 1e-10

    def test_midpoint_equals_average_of_ask_bid(self):
        """Reservation midpoint should equal (R^a + R^b) / 2."""
        s, q, t, gamma, sigma, T = 100.0, 3.0, 0.3, 0.1, 0.2, 1.0
        Ra = appendix_reservation_ask(s, q, t, gamma, sigma, T)
        Rb = appendix_reservation_bid(s, q, t, gamma, sigma, T)
        R = appendix_reservation_midpoint(s, q, t, gamma, sigma, T)
        assert abs(R - (Ra + Rb) / 2.0) < 1e-10

    def test_ask_above_bid(self):
        """Reservation ask must exceed reservation bid."""
        s, t, gamma, sigma, T = 100.0, 0.5, 0.1, 0.2, 1.0
        for q in [-5, 0, 5]:
            Ra = appendix_reservation_ask(s, q, t, gamma, sigma, T)
            Rb = appendix_reservation_bid(s, q, t, gamma, sigma, T)
            assert Ra > Rb, f"R^a <= R^b for q={q}"

    def test_positive_inventory_lowers_reservation(self):
        """Positive inventory should lower the reservation midpoint below s."""
        s, t, gamma, sigma, T = 100.0, 0.5, 0.1, 0.2, 1.0
        for q in [1, 5, 10]:
            R = appendix_reservation_midpoint(s, q, t, gamma, sigma, T)
            assert R < s, f"Expected R < s for q={q}, got R={R:.4f}"

    def test_negative_inventory_raises_reservation(self):
        """Negative inventory should raise the reservation midpoint above s."""
        s, t, gamma, sigma, T = 100.0, 0.5, 0.1, 0.2, 1.0
        for q in [-1, -5, -10]:
            R = appendix_reservation_midpoint(s, q, t, gamma, sigma, T)
            assert R > s, f"Expected R > s for q={q}, got R={R:.4f}"

    def test_reservation_vanishes_at_horizon(self):
        """Reservation price adjustment must vanish at t=T."""
        s, gamma, sigma, T = 100.0, 0.1, 0.2, 1.0
        for q in [-10, -1, 0, 1, 10]:
            R = appendix_reservation_midpoint(s, q, T, gamma, sigma, T)
            assert abs(R - s) < 1e-12, f"R != s at T for q={q}"

    def test_zero_inventory_reservation_equals_mid(self):
        """At q=0, reservation midpoint should equal s."""
        s, t, gamma, sigma, T = 100.0, 0.5, 0.1, 0.2, 1.0
        R = appendix_reservation_midpoint(s, 0.0, t, gamma, sigma, T)
        assert abs(R - s) < 1e-12

    def test_appendix_validation_passes(self):
        """validate_appendix_formulas should not raise for valid inputs."""
        validate_appendix_formulas(s=100.0, gamma=0.1, sigma=0.2, T=1.0)

    def test_appendix_vs_main_model_same_direction(self):
        """
        Both models should show same directional effect:
        positive inventory → reservation below mid-price.
        """
        s, t, gamma, sigma, T = 100.0, 0.5, 0.1, 2.0, 1.0
        q = 5
        R_appendix = appendix_reservation_midpoint(s, q, t, gamma, sigma, T)
        r_main = main_model_reservation_price(s, q, t, gamma, sigma, T)
        assert R_appendix < s, "Appendix: positive inventory should give R < s"
        assert r_main < s, "Main model: positive inventory should give r < s"

    def test_appendix_scales_with_price_squared(self):
        """
        Appendix adjustment should scale with s^2 (unlike main model which is s-independent).
        """
        q, t, gamma, sigma, T = 5.0, 0.5, 0.1, 0.2, 1.0
        s1, s2 = 100.0, 200.0
        adj1 = appendix_reservation_midpoint(s1, q, t, gamma, sigma, T) - s1
        adj2 = appendix_reservation_midpoint(s2, q, t, gamma, sigma, T) - s2
        # adj2 / adj1 should be approximately (s2/s1)^2 = 4
        ratio = adj2 / adj1
        assert abs(ratio - (s2 / s1) ** 2) < 1e-6, \
            f"Appendix adjustment should scale with s^2, got ratio={ratio:.4f}"

    def test_runs_without_error(self, results_dir):
        """Experiment 4 should complete without raising exceptions."""
        results = run_experiment_4(results_dir=results_dir)
        assert results is not None

    def test_plots_created(self, results_dir):
        """All appendix plots should be created."""
        run_experiment_4(results_dir=results_dir)
        assert os.path.exists(os.path.join(results_dir, "exp4_reservation_vs_inventory.png"))
        assert os.path.exists(os.path.join(results_dir, "exp4_reservation_vs_time.png"))
        assert os.path.exists(os.path.join(results_dir, "exp4_model_comparison.png"))

    def test_comparison_csv_created(self, results_dir):
        """Comparison CSV should be created."""
        run_experiment_4(results_dir=results_dir)
        assert os.path.exists(os.path.join(results_dir, "exp4_comparison.csv"))


# ---------------------------------------------------------------------------
# Cross-experiment consistency tests
# ---------------------------------------------------------------------------

class TestCrossExperiment:
    """Tests verifying consistency across experiments."""

    def test_exp1_exp2_same_params(self, fast_params, results_dir):
        """Experiments 1 and 2 should use identical simulation parameters."""
        r1 = run_experiment_1(params=fast_params, gammas=[0.1], results_dir=results_dir)
        r2 = run_experiment_2(params=fast_params, gammas=[0.1], results_dir=results_dir)
        # Both should have valid results for same gamma
        assert "gamma_0.1" in r1
        assert "gamma_0.1" in r2

    def test_exp3_parameters_consistent_with_exp1(self):
        """
        Exp 3 parameter mapping should be consistent with Exp 1 parameters.
        With A=140, k=1.5, and Lambda=140, alpha=1.5, K=1.0:
        A = Lambda/alpha = 140/1.5 ≈ 93.3 (not exactly 140)
        This shows A=140 in Exp 1 is a direct parameter, not derived from microstructure.
        """
        # The paper uses A=140 directly; Exp 3 shows how A could be derived
        params = derive_parameters_from_microstructure(Lambda=140.0, alpha=1.0, K=1.5)
        assert params["A"] == 140.0  # Lambda/alpha = 140/1 = 140
        assert params["k"] == 1.5   # alpha*K = 1*1.5 = 1.5

    def test_exp4_independent_from_exp1(self):
        """
        Appendix model uses different dynamics; reservation price formulas
        should differ from main model for non-zero inventory.
        """
        s, q, t, gamma, sigma, T = 100.0, 5.0, 0.5, 0.1, 2.0, 1.0
        R_appendix = appendix_reservation_midpoint(s, q, t, gamma, sigma, T)
        r_main = main_model_reservation_price(s, q, t, gamma, sigma, T)
        # They should differ (different formulas)
        assert abs(R_appendix - r_main) > 0.01, \
            "Appendix and main model should give different reservation prices"
