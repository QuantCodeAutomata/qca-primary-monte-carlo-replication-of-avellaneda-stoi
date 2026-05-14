"""
Tests for the microstructure calibration module (src/microstructure_calibration.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.microstructure_calibration import (
    exponential_intensity,
    power_law_intensity,
    interpret_exponential_params,
    exponential_intensity_sensitivity,
    power_law_intensity_sensitivity,
    tabulate_exponential_params,
    tabulate_power_law_exponents,
    ALPHA_US_STOCKS, ALPHA_NASDAQ, ALPHA_PARIS,
    BETA_SQUARE_ROOT, BETA_EMPIRICAL,
)


class TestExponentialIntensity:
    def test_positive(self):
        """Exponential intensity is positive."""
        lam = exponential_intensity(0.5, A=140.0, k=1.5)
        assert lam > 0

    def test_formula_exact(self):
        """Verify exact formula: A*exp(-k*delta)."""
        delta, A, k = 0.5, 140.0, 1.5
        expected = A * np.exp(-k * delta)
        assert abs(exponential_intensity(delta, A, k) - expected) < 1e-12

    def test_decreasing(self):
        """Intensity decreases with quote distance."""
        deltas = np.array([0.1, 0.5, 1.0, 2.0])
        lams = exponential_intensity(deltas, A=140.0, k=1.5)
        assert np.all(np.diff(lams) < 0)

    def test_at_zero_equals_A(self):
        """At delta=0, intensity equals A."""
        lam = exponential_intensity(0.0, A=140.0, k=1.5)
        assert abs(lam - 140.0) < 1e-12

    def test_array_input(self):
        """Works with array input."""
        deltas = np.linspace(0.1, 2.0, 50)
        lams = exponential_intensity(deltas, A=140.0, k=1.5)
        assert lams.shape == (50,)
        assert np.all(lams > 0)


class TestPowerLawIntensity:
    def test_positive(self):
        """Power-law intensity is positive for positive delta."""
        lam = power_law_intensity(0.5, B=1.0, alpha=1.5, beta=0.5)
        assert lam > 0

    def test_formula_exact(self):
        """Verify exact formula: B*delta^{-alpha/beta}."""
        delta, B, alpha, beta = 0.5, 1.0, 1.5, 0.5
        expected = B * delta ** (-alpha / beta)
        assert abs(power_law_intensity(delta, B, alpha, beta) - expected) < 1e-12

    def test_decreasing(self):
        """Intensity decreases with quote distance."""
        deltas = np.array([0.1, 0.5, 1.0, 2.0])
        lams = power_law_intensity(deltas, B=1.0, alpha=1.5, beta=0.5)
        assert np.all(np.diff(lams) < 0)

    def test_singular_at_zero(self):
        """Power-law intensity is singular at delta=0."""
        lam = power_law_intensity(1e-10, B=1.0, alpha=1.5, beta=0.5)
        assert lam > 1e6  # very large

    def test_array_input(self):
        """Works with array input."""
        deltas = np.linspace(0.05, 2.0, 50)
        lams = power_law_intensity(deltas, B=1.0, alpha=1.5, beta=0.5)
        assert lams.shape == (50,)
        assert np.all(lams > 0)


class TestInterpretExponentialParams:
    def test_k_equals_alpha_times_K(self):
        """k = alpha * K."""
        Lambda, alpha, K = 1.0, 1.5, 1.0
        A, k = interpret_exponential_params(Lambda, alpha, K)
        assert abs(k - alpha * K) < 1e-12

    def test_A_equals_Lambda_over_alpha(self):
        """A = Lambda / alpha (paper's normalization)."""
        Lambda, alpha, K = 1.0, 1.5, 1.0
        A, k = interpret_exponential_params(Lambda, alpha, K)
        assert abs(A - Lambda / alpha) < 1e-12

    def test_positive_outputs(self):
        """A and k should be positive."""
        A, k = interpret_exponential_params(Lambda=140.0, alpha=1.5, K=1.0)
        assert A > 0
        assert k > 0


class TestRepresentativeAlphaValues:
    def test_alpha_us_stocks(self):
        """US stocks alpha = 1.53."""
        assert abs(ALPHA_US_STOCKS - 1.53) < 1e-10

    def test_alpha_nasdaq(self):
        """NASDAQ alpha = 1.4."""
        assert abs(ALPHA_NASDAQ - 1.4) < 1e-10

    def test_alpha_paris(self):
        """Paris Bourse alpha = 1.5."""
        assert abs(ALPHA_PARIS - 1.5) < 1e-10

    def test_beta_square_root(self):
        """Square-root impact beta = 0.5."""
        assert abs(BETA_SQUARE_ROOT - 0.5) < 1e-10

    def test_beta_empirical(self):
        """Empirical beta = 0.76."""
        assert abs(BETA_EMPIRICAL - 0.76) < 1e-10


class TestSensitivityFunctions:
    def test_exponential_sensitivity_returns_dict(self):
        """Sensitivity function returns dict with correct keys."""
        delta_grid = np.linspace(0.01, 2.0, 50)
        results = exponential_intensity_sensitivity(
            delta_grid, Lambda=1.0,
            alpha_values=[1.4, 1.5], K_values=[0.5, 1.0]
        )
        assert (1.4, 0.5) in results
        assert (1.5, 1.0) in results

    def test_power_law_sensitivity_returns_dict(self):
        """Power-law sensitivity function returns dict with correct keys."""
        delta_grid = np.linspace(0.05, 2.0, 50)
        results = power_law_intensity_sensitivity(
            delta_grid, B=1.0,
            alpha_values=[1.4, 1.5], beta_values=[0.5, 0.76]
        )
        assert (1.4, 0.5) in results
        assert (1.5, 0.76) in results

    def test_larger_alpha_steeper_decay(self):
        """Larger alpha → steeper exponential decay."""
        delta_grid = np.linspace(0.1, 2.0, 50)
        results = exponential_intensity_sensitivity(
            delta_grid, Lambda=1.0,
            alpha_values=[1.0, 2.0], K_values=[1.0]
        )
        lam_low = results[(1.0, 1.0)]
        lam_high = results[(2.0, 1.0)]
        # At large delta, higher alpha should give lower intensity
        assert lam_high[-1] < lam_low[-1]


class TestTabulateFunctions:
    def test_tabulate_exponential_params_returns_list(self):
        """Tabulate function returns non-empty list."""
        result = tabulate_exponential_params(Lambda=1.0, K_values=[0.5, 1.0])
        assert len(result) > 0

    def test_tabulate_power_law_exponents_returns_list(self):
        """Tabulate power-law exponents returns non-empty list."""
        result = tabulate_power_law_exponents()
        assert len(result) > 0

    def test_tabulate_exponential_has_required_columns(self):
        """Tabulated exponential params have required columns."""
        result = tabulate_exponential_params(Lambda=1.0, K_values=[1.0])
        assert "alpha" in result[0]
        assert "K" in result[0]
        assert "A" in result[0]
        assert "k" in result[0]
