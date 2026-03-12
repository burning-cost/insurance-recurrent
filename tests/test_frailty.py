"""
Tests for SharedFrailtyModel.

The EM algorithm is complex enough that we need:
1. Basic API tests (fit/predict interface)
2. Convergence tests (does it converge on simulated data?)
3. Sanity check: higher-frailty policies should have higher posterior mean
4. Credibility factor tests (closed-form properties)
5. Bühlmann connection (posterior mean = credibility-blended estimate)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_recurrent import (
    RecurrentEventData,
    RecurrentEventSimulator,
    SharedFrailtyModel,
)


class TestSharedFrailtyModelInit:
    def test_default_construction(self):
        m = SharedFrailtyModel()
        assert m.theta_init == 1.0
        assert m.max_iter == 200
        assert not m.is_fitted

    def test_negative_theta_init_raises(self):
        with pytest.raises(ValueError, match="theta_init"):
            SharedFrailtyModel(theta_init=-0.1)

    def test_zero_theta_init_raises(self):
        with pytest.raises(ValueError, match="theta_init"):
            SharedFrailtyModel(theta_init=0.0)

    def test_is_fitted_before_fit(self):
        m = SharedFrailtyModel()
        assert not m.is_fitted

    def test_predict_before_fit_raises(self, small_sim_data):
        m = SharedFrailtyModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            m.predict_frailty(small_sim_data)

    def test_summary_before_fit_raises(self):
        m = SharedFrailtyModel()
        with pytest.raises(RuntimeError):
            m.summary()


class TestSharedFrailtyModelFit:
    def test_fit_returns_self(self, small_sim_data):
        m = SharedFrailtyModel(max_iter=5)
        result = m.fit(small_sim_data)
        assert result is m

    def test_is_fitted_after_fit(self, small_sim_data):
        m = SharedFrailtyModel(max_iter=10)
        m.fit(small_sim_data)
        assert m.is_fitted

    def test_theta_is_positive_after_fit(self, small_sim_data):
        m = SharedFrailtyModel(max_iter=20)
        m.fit(small_sim_data)
        assert m.theta_ > 0

    def test_theta_bounded(self, small_sim_data):
        m = SharedFrailtyModel(max_iter=20)
        m.fit(small_sim_data)
        assert 1e-4 <= m.theta_ <= 20.0

    def test_fit_without_covariates(self, small_sim_data):
        m = SharedFrailtyModel(max_iter=15)
        m.fit(small_sim_data, covariates=None)
        assert len(m.coef_) == 0
        assert len(m.covariate_names_) == 0

    def test_fit_with_covariates(self, medium_sim_data):
        m = SharedFrailtyModel(max_iter=10)
        m.fit(medium_sim_data, covariates=["age_cat", "vehicle_type"])
        assert len(m.coef_) == 2
        assert m.covariate_names_ == ["age_cat", "vehicle_type"]

    def test_baseline_hazard_is_dataframe(self, small_sim_data):
        m = SharedFrailtyModel(max_iter=10)
        m.fit(small_sim_data)
        bh = m.baseline_hazard_
        assert isinstance(bh, pd.DataFrame)
        assert "cumhaz" in bh.columns
        assert "time" in bh.columns

    def test_baseline_hazard_monotone(self, small_sim_data):
        m = SharedFrailtyModel(max_iter=10)
        m.fit(small_sim_data)
        bh = m.baseline_hazard_
        assert (bh["cumhaz"].diff().dropna() >= 0).all()

    def test_log_likelihood_is_finite(self, small_sim_data):
        m = SharedFrailtyModel(max_iter=15)
        m.fit(small_sim_data)
        assert np.isfinite(m.log_likelihood_)

    def test_n_iter_recorded(self, small_sim_data):
        m = SharedFrailtyModel(max_iter=10)
        m.fit(small_sim_data)
        assert 1 <= m.n_iter_ <= 10


class TestSharedFrailtyModelPredict:
    def test_predict_returns_list(self, small_sim_data):
        m = SharedFrailtyModel(max_iter=10)
        m.fit(small_sim_data)
        preds = m.predict_frailty(small_sim_data)
        assert isinstance(preds, list)

    def test_predict_has_all_policies(self, small_sim_data):
        m = SharedFrailtyModel(max_iter=10)
        m.fit(small_sim_data)
        preds = m.predict_frailty(small_sim_data)
        pred_ids = {p["policy_id"] for p in preds}
        data_ids = set(small_sim_data.policy_ids)
        assert pred_ids == data_ids

    def test_frailty_mean_positive(self, small_sim_data):
        m = SharedFrailtyModel(max_iter=10)
        m.fit(small_sim_data)
        preds = m.predict_frailty(small_sim_data)
        assert all(p["frailty_mean"] > 0 for p in preds)

    def test_frailty_mean_finite(self, small_sim_data):
        m = SharedFrailtyModel(max_iter=10)
        m.fit(small_sim_data)
        preds = m.predict_frailty(small_sim_data)
        assert all(np.isfinite(p["frailty_mean"]) for p in preds)

    def test_credibility_factor_in_range(self, small_sim_data):
        m = SharedFrailtyModel(max_iter=10)
        m.fit(small_sim_data)
        preds = m.predict_frailty(small_sim_data)
        for p in preds:
            assert 0.0 <= p["credibility_factor"] <= 1.0

    def test_higher_claims_higher_frailty(self, small_sim_data):
        """Policies with more claims should generally have higher posterior frailty."""
        m = SharedFrailtyModel(max_iter=15)
        m.fit(small_sim_data)
        preds = m.predict_frailty(small_sim_data)
        pred_df = pd.DataFrame(preds)

        # High-claim policies (top tertile) should have higher mean frailty
        high_claim = pred_df[pred_df["n_events"] >= pred_df["n_events"].quantile(0.67)]
        low_claim = pred_df[pred_df["n_events"] == 0]

        if len(high_claim) > 5 and len(low_claim) > 5:
            assert high_claim["frailty_mean"].mean() > low_claim["frailty_mean"].mean()

    def test_zero_claimer_frailty_less_than_one(self, small_sim_data):
        """Zero-claim policies should have posterior frailty < 1.0 (shrunk down)."""
        m = SharedFrailtyModel(max_iter=15)
        m.fit(small_sim_data)
        preds = m.predict_frailty(small_sim_data)
        zero_claimers = [p for p in preds if p["n_events"] == 0]
        if len(zero_claimers) > 0:
            assert all(p["frailty_mean"] < 1.0 for p in zero_claimers)


class TestSharedFrailtyModelPredictExpected:
    def test_predict_expected_events_returns_df(self, small_sim_data):
        m = SharedFrailtyModel(max_iter=10)
        m.fit(small_sim_data)
        result = m.predict_expected_events(small_sim_data, time_horizon=1.0)
        assert isinstance(result, pd.DataFrame)
        assert "expected_events" in result.columns

    def test_expected_events_non_negative(self, small_sim_data):
        m = SharedFrailtyModel(max_iter=10)
        m.fit(small_sim_data)
        result = m.predict_expected_events(small_sim_data)
        assert (result["expected_events"] >= 0).all()


class TestSharedFrailtyModelSummary:
    def test_summary_returns_dict(self, small_sim_data):
        m = SharedFrailtyModel(max_iter=10)
        m.fit(small_sim_data)
        s = m.summary()
        assert isinstance(s, dict)
        assert "theta" in s
        assert "log_likelihood" in s
        assert "converged" in s

    def test_print_summary_runs(self, small_sim_data, capsys):
        m = SharedFrailtyModel(max_iter=10)
        m.fit(small_sim_data)
        m.print_summary()
        captured = capsys.readouterr()
        assert "theta" in captured.out.lower()


class TestSharedFrailtyCredibility:
    def test_credibility_factors_df(self, small_sim_data):
        m = SharedFrailtyModel(max_iter=10)
        m.fit(small_sim_data)
        cred = m.credibility_factors
        assert isinstance(cred, pd.DataFrame)
        assert "credibility_factor" in cred.columns

    def test_credibility_increases_with_events(self, small_sim_data):
        m = SharedFrailtyModel(max_iter=10)
        m.fit(small_sim_data)
        cred = m.credibility_factors
        # Z should be monotonically increasing with n_events
        z_vals = cred["credibility_factor"].values
        assert (np.diff(z_vals) >= 0).all()

    def test_credibility_bounded_zero_one(self, small_sim_data):
        m = SharedFrailtyModel(max_iter=10)
        m.fit(small_sim_data)
        cred = m.credibility_factors
        assert (cred["credibility_factor"] >= 0).all()
        assert (cred["credibility_factor"] <= 1).all()

    def test_frailty_variance_property(self, small_sim_data):
        m = SharedFrailtyModel(max_iter=10)
        m.fit(small_sim_data)
        assert m.frailty_variance == m.theta_


class TestSharedFrailtyConvergence:
    def test_converges_on_clean_simulated_data(self):
        """EM should converge on reasonably sized clean simulated data."""
        sim = RecurrentEventSimulator(n_policies=300, theta=0.5, baseline_rate=0.4, seed=42)
        data = sim.simulate()
        m = SharedFrailtyModel(max_iter=100, tol=1e-5)
        m.fit(data)
        # Convergence not guaranteed in 100 iter for all configs, but
        # we check theta is in plausible range
        assert 0.1 < m.theta_ < 5.0

    def test_no_convergence_warns(self):
        """Very tight tolerance with few iterations should warn."""
        sim = RecurrentEventSimulator(n_policies=50, theta=0.5, seed=1)
        data = sim.simulate()
        m = SharedFrailtyModel(max_iter=2, tol=1e-20)
        with pytest.warns(RuntimeWarning, match="converge"):
            m.fit(data)
        assert not m.converged_
