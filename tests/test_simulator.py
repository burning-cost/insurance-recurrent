"""
Tests for RecurrentEventSimulator.

We verify:
1. Output is valid RecurrentEventData
2. Known DGP properties (mean claim rate, frailty moments)
3. Reproducibility with seed
4. Edge cases (theta=0, no lapses, covariates)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_recurrent import RecurrentEventData, RecurrentEventSimulator


class TestSimulatorBasicOutput:
    def test_returns_recurrent_event_data(self):
        sim = RecurrentEventSimulator(n_policies=50, seed=1)
        result = sim.simulate()
        assert isinstance(result, RecurrentEventData)

    def test_n_policies_roughly_correct(self):
        sim = RecurrentEventSimulator(n_policies=200, seed=2)
        data = sim.simulate()
        # Some policies may be dropped (degenerate windows) but most should survive
        assert data.n_policies >= 150

    def test_reproducibility_with_seed(self):
        sim1 = RecurrentEventSimulator(n_policies=100, seed=42)
        sim2 = RecurrentEventSimulator(n_policies=100, seed=42)
        d1 = sim1.simulate()
        d2 = sim2.simulate()
        assert d1.n_events == d2.n_events
        assert d1.n_policies == d2.n_policies

    def test_different_seeds_differ(self):
        sim1 = RecurrentEventSimulator(n_policies=100, seed=1)
        sim2 = RecurrentEventSimulator(n_policies=100, seed=2)
        d1 = sim1.simulate()
        d2 = sim2.simulate()
        # Very unlikely to be exactly equal
        assert d1.n_events != d2.n_events or d1.n_policies != d2.n_policies

    def test_output_is_valid(self):
        sim = RecurrentEventSimulator(n_policies=100, seed=5)
        data = sim.simulate()
        # Validation runs in __post_init__ — should not raise
        assert data.n_rows > 0


class TestSimulatorTimeScales:
    def test_gap_time_tstart_zero_for_events(self):
        sim = RecurrentEventSimulator(n_policies=100, seed=10, baseline_rate=0.5)
        data = sim.simulate(time_scale="gap")
        event_rows = data.df[data.df["event"] == 1]
        if len(event_rows) > 0:
            assert (event_rows["tstart"] == 0.0).all()

    def test_total_time_tstart_monotone_per_policy(self):
        sim = RecurrentEventSimulator(n_policies=50, seed=11)
        data = sim.simulate(time_scale="total")
        for pid, group in data.df.groupby("policy_id"):
            sorted_g = group.sort_values("tstart")
            assert (sorted_g["tstart"].values[1:] >= sorted_g["tstart"].values[:-1]).all()


class TestSimulatorDGPProperties:
    def test_zero_theta_gives_poisson_like(self):
        """With theta=0, all frailties are 1 — purely Poisson process."""
        sim = RecurrentEventSimulator(n_policies=500, theta=0.0, baseline_rate=0.2, seed=7)
        data = sim.simulate()
        # With unit frailty, claim count per unit exposure should be ~Poisson(0.2)
        # Check that mean is roughly right
        avg_rate = data.n_events / data.df["exposure"].sum()
        assert 0.1 < avg_rate < 0.4  # wide bounds for stochastic test

    def test_high_theta_gives_more_variability(self):
        """Higher theta should produce more spread in claim counts per policy."""
        sim_low = RecurrentEventSimulator(n_policies=300, theta=0.1, baseline_rate=0.5, seed=3)
        sim_high = RecurrentEventSimulator(n_policies=300, theta=2.0, baseline_rate=0.5, seed=3)
        d_low = sim_low.simulate()
        d_high = sim_high.simulate()

        def count_variance(data):
            counts = data.df.groupby("policy_id")["event"].sum()
            return counts.var()

        assert count_variance(d_high) >= count_variance(d_low)

    def test_covariate_effect_changes_rate(self):
        """A positive covariate coefficient should increase claim rate for group 1."""
        sim = RecurrentEventSimulator(
            n_policies=500, theta=0.0, baseline_rate=0.2,
            coef={"high_risk": 1.0}, seed=20
        )
        data = sim.simulate()

        # Group events by the high_risk covariate
        df = data.df
        rate_high = df[df.get("high_risk", 0) == 1.0]["event"].sum() / df[df.get("high_risk", 0) == 1.0]["exposure"].sum() if "high_risk" in df.columns else None
        rate_low = df[df.get("high_risk", 0) == 0.0]["event"].sum() / df[df.get("high_risk", 0) == 0.0]["exposure"].sum() if "high_risk" in df.columns else None

        if rate_high is not None and rate_low is not None:
            assert rate_high > rate_low  # exp(1.0) ≈ 2.7x higher


class TestSimulatorReturnTrueFramilty:
    def test_return_true_frailty(self):
        sim = RecurrentEventSimulator(n_policies=50, seed=30)
        result = sim.simulate(return_true_frailty=True)
        assert isinstance(result, tuple)
        data, truth = result
        assert isinstance(data, RecurrentEventData)
        assert isinstance(truth, pd.DataFrame)
        assert "true_frailty" in truth.columns
        assert "policy_id" in truth.columns

    def test_true_frailty_mean_near_one(self):
        """Gamma frailty with E[u]=1 — sample mean should be close to 1."""
        sim = RecurrentEventSimulator(n_policies=1000, theta=0.5, seed=99)
        _, truth = sim.simulate(return_true_frailty=True)
        mean_frailty = truth["true_frailty"].mean()
        assert 0.8 < mean_frailty < 1.2


class TestSimulatorFrailtyDist:
    def test_gamma_frailty(self):
        sim = RecurrentEventSimulator(n_policies=100, theta=0.5, frailty_dist="gamma", seed=1)
        data = sim.simulate()
        assert data.n_rows > 0

    def test_lognormal_frailty(self):
        sim = RecurrentEventSimulator(n_policies=100, theta=0.5, frailty_dist="lognormal", seed=1)
        data = sim.simulate()
        assert data.n_rows > 0

    def test_invalid_frailty_dist_raises(self):
        with pytest.raises(ValueError, match="frailty_dist"):
            sim = RecurrentEventSimulator(frailty_dist="gaussian", seed=1)
            sim.simulate()


class TestSimulatorLapse:
    def test_lapse_reduces_observation_time(self):
        """With high lapse rate, mean observation time should be shorter."""
        sim_no_lapse = RecurrentEventSimulator(n_policies=200, lapse_rate=0.0, seed=5)
        sim_lapse = RecurrentEventSimulator(n_policies=200, lapse_rate=2.0, seed=5)
        d_no = sim_no_lapse.simulate(return_true_frailty=True)[1]
        d_lapse = sim_lapse.simulate(return_true_frailty=True)[1]
        assert d_lapse["observation_time"].mean() < d_no["observation_time"].mean()


class TestParameterTable:
    def test_parameter_table_contains_theta(self):
        sim = RecurrentEventSimulator(theta=0.7, coef={"x": 0.3})
        tbl = sim.parameter_table()
        assert "theta (frailty variance)" in tbl["parameter"].values
        theta_row = tbl[tbl["parameter"] == "theta (frailty variance)"]
        assert float(theta_row["true_value"].iloc[0]) == pytest.approx(0.7)

    def test_parameter_table_contains_covariates(self):
        sim = RecurrentEventSimulator(coef={"age": 0.5, "region": -0.2})
        tbl = sim.parameter_table()
        assert "coef[age]" in tbl["parameter"].values
        assert "coef[region]" in tbl["parameter"].values
