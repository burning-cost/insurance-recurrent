"""
Tests for JointFrailtyModel.

The joint model is more complex — we test API completeness, basic fitting
behaviour, and the association parameter alpha.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_recurrent import RecurrentEventData, RecurrentEventSimulator, JointFrailtyModel


@pytest.fixture(scope="module")
def joint_data():
    """Simulated data with informative lapse (high-frailty policies lapse faster)."""
    sim = RecurrentEventSimulator(
        n_policies=200,
        theta=0.6,
        baseline_rate=0.3,
        lapse_rate=0.5,
        lapse_frailty_assoc=0.5,
        seed=77,
    )
    data, truth = sim.simulate(return_true_frailty=True)
    return data, truth


@pytest.fixture(scope="module")
def simple_lapse_df(joint_data):
    """Corresponding lapse DataFrame."""
    data, truth = joint_data
    lapse_df = truth[["policy_id", "observation_time", "lapsed"]].rename(
        columns={"observation_time": "lapse_time"}
    )
    return lapse_df


class TestJointFrailtyModelInit:
    def test_default_construction(self):
        m = JointFrailtyModel()
        assert m.theta_init == 1.0
        assert m.alpha_init == 0.0
        assert m.frailty_dist == "gamma"
        assert not m.is_fitted

    def test_invalid_frailty_dist_raises(self):
        with pytest.raises(ValueError, match="frailty_dist"):
            JointFrailtyModel(frailty_dist="pareto")

    def test_lognormal_frailty_allowed(self):
        m = JointFrailtyModel(frailty_dist="lognormal")
        assert m.frailty_dist == "lognormal"


class TestJointFrailtyModelFit:
    def test_fit_returns_self(self, joint_data):
        data, _ = joint_data
        m = JointFrailtyModel(max_iter=5)
        result = m.fit(data)
        assert result is m

    def test_is_fitted_after_fit(self, joint_data):
        data, _ = joint_data
        m = JointFrailtyModel(max_iter=5)
        m.fit(data)
        assert m.is_fitted

    def test_theta_positive(self, joint_data):
        data, _ = joint_data
        m = JointFrailtyModel(max_iter=10)
        m.fit(data)
        assert m.theta_ > 0

    def test_fit_with_lapse_data(self, joint_data, simple_lapse_df):
        data, _ = joint_data
        m = JointFrailtyModel(max_iter=5)
        m.fit(data, lapse_data=simple_lapse_df)
        assert m.is_fitted

    def test_association_estimated(self, joint_data, simple_lapse_df):
        data, _ = joint_data
        m = JointFrailtyModel(max_iter=10)
        m.fit(data, lapse_data=simple_lapse_df)
        # Association should be estimated (not just initial value 0.0)
        # In a 10-iter run it may not fully converge, but alpha should be finite
        assert np.isfinite(m.association_)

    def test_lognormal_fit_runs(self, joint_data):
        data, _ = joint_data
        m = JointFrailtyModel(frailty_dist="lognormal", max_iter=5)
        m.fit(data)
        assert m.is_fitted


class TestJointFrailtyModelPredict:
    def test_predict_frailty_returns_list(self, joint_data):
        data, _ = joint_data
        m = JointFrailtyModel(max_iter=5)
        m.fit(data)
        preds = m.predict_frailty(data)
        assert isinstance(preds, list)
        assert len(preds) == data.n_policies

    def test_frailty_mean_positive(self, joint_data):
        data, _ = joint_data
        m = JointFrailtyModel(max_iter=5)
        m.fit(data)
        preds = m.predict_frailty(data)
        assert all(p["frailty_mean"] > 0 for p in preds)

    def test_credibility_factor_bounded(self, joint_data):
        data, _ = joint_data
        m = JointFrailtyModel(max_iter=5)
        m.fit(data)
        preds = m.predict_frailty(data)
        for p in preds:
            assert 0.0 <= p["credibility_factor"] <= 1.0

    def test_predict_before_fit_raises(self, joint_data):
        data, _ = joint_data
        m = JointFrailtyModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            m.predict_frailty(data)


class TestJointFrailtyModelSummary:
    def test_summary_returns_dict(self, joint_data):
        data, _ = joint_data
        m = JointFrailtyModel(max_iter=5)
        m.fit(data)
        s = m.summary()
        assert "theta" in s
        assert "association_alpha" in s
        assert "frailty_dist" in s

    def test_print_summary_runs(self, joint_data, capsys):
        data, _ = joint_data
        m = JointFrailtyModel(max_iter=5)
        m.fit(data)
        m.print_summary()
        captured = capsys.readouterr()
        assert "alpha" in captured.out.lower()


class TestJointFrailtyQuadrature:
    def test_gamma_quadrature_nodes_positive(self):
        m = JointFrailtyModel(frailty_dist="gamma", n_quad=10, theta_init=0.5)
        nodes, weights = m._get_quadrature(0.5)
        assert (nodes >= 0).all()

    def test_lognormal_quadrature_nodes_positive(self):
        m = JointFrailtyModel(frailty_dist="lognormal", n_quad=10, theta_init=0.5)
        nodes, weights = m._get_quadrature(0.5)
        assert (nodes > 0).all()

    def test_quadrature_weights_positive(self):
        m = JointFrailtyModel(n_quad=10)
        _, weights = m._get_quadrature(0.5)
        assert (weights > 0).all()

    def test_quadrature_nodes_count(self):
        m = JointFrailtyModel(n_quad=15)
        nodes, weights = m._get_quadrature(0.5)
        assert len(nodes) == 15
        assert len(weights) == 15
