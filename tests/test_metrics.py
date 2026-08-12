"""Tests for summary statistics and validation functions."""
import pytest
import numpy as np
import pandas as pd
from src.config import SimParams
from src.metrics import (
    compute_summary_stats,
    validate_terminal_profit,
    validate_lambda_monotonic,
    validate_reservation_price,
    compute_full_spread_formula,
    compute_constant_spread,
    compute_initial_full_spread,
)


@pytest.fixture
def params():
    return SimParams()


def make_mock_results(n_paths: int = 100) -> pd.DataFrame:
    """Create a minimal mock results DataFrame for testing."""
    rng = np.random.default_rng(0)
    rows = []
    for gamma in [0.01, 0.1, 0.5]:
        for strategy in ['inventory', 'symmetric']:
            for i in range(n_paths):
                rows.append({
                    'path_id': i,
                    'gamma': gamma,
                    'strategy': strategy,
                    'terminal_profit': rng.normal(0, 5),
                    'terminal_inventory': int(rng.integers(-5, 6)),
                    'negative_delta_count': 0,
                    'bernoulli_violation_count': 0,
                })
    return pd.DataFrame(rows)


class TestComputeSummaryStats:
    """Tests for summary statistics aggregation."""

    def test_output_shape(self):
        df = make_mock_results()
        summary = compute_summary_stats(df)
        assert len(summary) == 6  # 3 gammas * 2 strategies

    def test_required_columns(self):
        df = make_mock_results()
        summary = compute_summary_stats(df)
        for col in ['gamma', 'strategy', 'mean_profit', 'std_profit', 'mean_final_q', 'std_final_q']:
            assert col in summary.columns

    def test_std_non_negative(self):
        df = make_mock_results()
        summary = compute_summary_stats(df)
        assert (summary['std_profit'] >= 0).all()
        assert (summary['std_final_q'] >= 0).all()

    def test_n_paths_correct(self):
        df = make_mock_results(100)
        summary = compute_summary_stats(df)
        assert (summary['n_paths'] == 100).all()


class TestValidations:
    """Tests for validation functions."""

    def test_validate_terminal_profit_all_finite(self):
        df = make_mock_results()
        assert validate_terminal_profit(df) is True

    def test_validate_terminal_profit_with_nan(self):
        df = make_mock_results()
        df.loc[0, 'terminal_profit'] = np.nan
        assert validate_terminal_profit(df) is False

    def test_lambda_monotonic_all_gammas(self, params):
        for gamma in params.gammas:
            assert validate_lambda_monotonic(params, gamma)

    def test_reservation_price_all_gammas(self, params):
        for gamma in params.gammas:
            assert validate_reservation_price(params, gamma)


class TestSpreadFormulas:
    """Tests for spread formula functions."""

    def test_full_spread_at_maturity_equals_constant(self, params):
        """Full spread at T-t=0 equals the constant term."""
        gamma = 0.1
        full_at_T = compute_full_spread_formula(gamma, params.sigma, params.k, 0.0)
        const = compute_constant_spread(gamma, params.k)
        assert np.isclose(full_at_T, const, rtol=1e-10)

    def test_full_spread_positive(self, params):
        for gamma in params.gammas:
            spread = compute_initial_full_spread(gamma, params.sigma, params.k, params.T)
            assert spread > 0

    def test_constant_spread_matches_formula(self, params):
        """Constant spread = (2/gamma)*ln(1+gamma/k) must be positive."""
        for gamma in params.gammas:
            cs = compute_constant_spread(gamma, params.k)
            assert cs > 0
            expected = (2.0 / gamma) * np.log1p(gamma / params.k)
            assert np.isclose(cs, expected)

    def test_spread_values_for_known_gammas(self, params):
        """Verify spread values for known gamma values match hand-computed values."""
        k = params.k  # 1.5
        # gamma=0.1: (2/0.1)*ln(1+0.1/1.5) = 20*ln(1.0667) = 20*0.06454 = 1.291
        cs_01 = compute_constant_spread(0.1, k)
        assert 1.0 < cs_01 < 2.0  # Reasonable range
        # gamma=0.5: (2/0.5)*ln(1+0.5/1.5) = 4*ln(1.3333) = 4*0.2877 = 1.151
        cs_05 = compute_constant_spread(0.5, k)
        assert 0.5 < cs_05 < 2.0
