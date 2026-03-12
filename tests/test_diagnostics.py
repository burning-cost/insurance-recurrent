"""
Tests for the diagnostics module.

Diagnostics return DataFrames — we test shape, column presence,
and basic numerical properties rather than visual output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_recurrent import (
    RecurrentEventSimulator,
    SharedFrailtyModel,
    frailty_qq_data,
    cox_snell_residuals,
    event_rate_by_frailty_decile,
    frailty_summary_stats,
)


@pytest.fixture(scope="module")
def fitted_model_and_data():
    sim = RecurrentEventSimulator(n_policies=150, theta=0.5, baseline_rate=0.4, seed=55)
    data = sim.simulate()
    m = SharedFrailtyModel(max_iter=20)
    m.fit(data)
    return m, data


class TestFrailtyQQData:
    def test_returns_dataframe(self, fitted_model_and_data):
        m, data = fitted_model_and_data
        result = frailty_qq_data(m, data)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self, fitted_model_and_data):
        m, data = fitted_model_and_data
        result = frailty_qq_data(m, data)
        assert "theoretical" in result.columns
        assert "empirical" in result.columns
        assert "policy_id" in result.columns

    def test_correct_length(self, fitted_model_and_data):
        m, data = fitted_model_and_data
        result = frailty_qq_data(m, data)
        assert len(result) == data.n_policies

    def test_theoretical_positive(self, fitted_model_and_data):
        m, data = fitted_model_and_data
        result = frailty_qq_data(m, data)
        assert (result["theoretical"] > 0).all()

    def test_empirical_positive(self, fitted_model_and_data):
        m, data = fitted_model_and_data
        result = frailty_qq_data(m, data)
        assert (result["empirical"] > 0).all()

    def test_both_sorted_ascending(self, fitted_model_and_data):
        m, data = fitted_model_and_data
        result = frailty_qq_data(m, data)
        assert (result["theoretical"].diff().dropna() >= 0).all()
        assert (result["empirical"].diff().dropna() >= 0).all()


class TestCoxSnellResiduals:
    def test_returns_dataframe(self, fitted_model_and_data):
        m, data = fitted_model_and_data
        result = cox_snell_residuals(m, data)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self, fitted_model_and_data):
        m, data = fitted_model_and_data
        result = cox_snell_residuals(m, data)
        required = {"policy_id", "tstart", "tstop", "event", "residual", "cumulative_residual"}
        assert required.issubset(set(result.columns))

    def test_residuals_non_negative(self, fitted_model_and_data):
        m, data = fitted_model_and_data
        result = cox_snell_residuals(m, data)
        assert (result["residual"] >= 0).all()

    def test_cumulative_residuals_non_negative(self, fitted_model_and_data):
        m, data = fitted_model_and_data
        result = cox_snell_residuals(m, data)
        assert (result["cumulative_residual"] >= 0).all()

    def test_correct_row_count(self, fitted_model_and_data):
        m, data = fitted_model_and_data
        result = cox_snell_residuals(m, data)
        assert len(result) == data.n_rows


class TestEventRateByFrailtyDecile:
    def test_returns_dataframe(self, fitted_model_and_data):
        m, data = fitted_model_and_data
        result = event_rate_by_frailty_decile(m, data)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self, fitted_model_and_data):
        m, data = fitted_model_and_data
        result = event_rate_by_frailty_decile(m, data)
        assert "decile" in result.columns
        assert "observed_rate" in result.columns
        assert "n_policies" in result.columns

    def test_n_deciles(self, fitted_model_and_data):
        m, data = fitted_model_and_data
        result = event_rate_by_frailty_decile(m, data, n_deciles=10)
        # May have fewer deciles if not enough policies
        assert len(result) <= 10
        assert len(result) >= 1

    def test_custom_n_deciles(self, fitted_model_and_data):
        m, data = fitted_model_and_data
        result = event_rate_by_frailty_decile(m, data, n_deciles=5)
        assert len(result) <= 5

    def test_observed_rate_non_negative(self, fitted_model_and_data):
        m, data = fitted_model_and_data
        result = event_rate_by_frailty_decile(m, data)
        assert (result["observed_rate"] >= 0).all()

    def test_total_policies_match(self, fitted_model_and_data):
        m, data = fitted_model_and_data
        result = event_rate_by_frailty_decile(m, data)
        assert result["n_policies"].sum() == data.n_policies


class TestFrailtyummaryStats:
    def test_returns_dataframe(self, fitted_model_and_data):
        m, data = fitted_model_and_data
        preds = m.predict_frailty(data)
        result = frailty_summary_stats(preds, m.theta_)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self, fitted_model_and_data):
        m, data = fitted_model_and_data
        preds = m.predict_frailty(data)
        result = frailty_summary_stats(preds, m.theta_)
        assert "statistic" in result.columns
        assert "value" in result.columns

    def test_n_policies_stat_correct(self, fitted_model_and_data):
        m, data = fitted_model_and_data
        preds = m.predict_frailty(data)
        result = frailty_summary_stats(preds, m.theta_)
        n_row = result[result["statistic"] == "n_policies"]
        assert len(n_row) == 1
        assert int(n_row["value"].iloc[0]) == data.n_policies

    def test_frailty_mean_near_one(self, fitted_model_and_data):
        """Population average posterior frailty should be near 1."""
        m, data = fitted_model_and_data
        preds = m.predict_frailty(data)
        result = frailty_summary_stats(preds, m.theta_)
        mean_row = result[result["statistic"] == "frailty_mean"]
        frailty_mean = float(mean_row["value"].iloc[0])
        # Should be close to 1.0 (prior mean)
        assert 0.5 < frailty_mean < 2.0

    def test_all_stats_finite(self, fitted_model_and_data):
        m, data = fitted_model_and_data
        preds = m.predict_frailty(data)
        result = frailty_summary_stats(preds, m.theta_)
        assert result["value"].apply(np.isfinite).all()
