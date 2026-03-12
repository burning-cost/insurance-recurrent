"""
Tests for RecurrentEventData.

We test validation, factory constructors, properties, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_recurrent import RecurrentEventData


class TestRecurrentEventDataValidation:
    def test_valid_construction(self, minimal_counting_process_df):
        data = RecurrentEventData(df=minimal_counting_process_df)
        assert data.n_policies == 3

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"policy_id": ["A"], "tstart": [0.0], "tstop": [1.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            RecurrentEventData(df=df)

    def test_tstop_must_exceed_tstart(self):
        df = pd.DataFrame({
            "policy_id": ["A"], "tstart": [1.0], "tstop": [0.5],
            "event": [0], "exposure": [0.5],
        })
        with pytest.raises(ValueError, match="tstop > tstart"):
            RecurrentEventData(df=df)

    def test_equal_times_raises(self):
        df = pd.DataFrame({
            "policy_id": ["A"], "tstart": [0.5], "tstop": [0.5],
            "event": [0], "exposure": [0.5],
        })
        with pytest.raises(ValueError, match="tstop > tstart"):
            RecurrentEventData(df=df)

    def test_event_must_be_binary(self):
        df = pd.DataFrame({
            "policy_id": ["A"], "tstart": [0.0], "tstop": [1.0],
            "event": [2], "exposure": [1.0],
        })
        with pytest.raises(ValueError, match="binary"):
            RecurrentEventData(df=df)

    def test_exposure_must_be_positive(self):
        df = pd.DataFrame({
            "policy_id": ["A"], "tstart": [0.0], "tstop": [1.0],
            "event": [0], "exposure": [0.0],
        })
        with pytest.raises(ValueError, match="exposure"):
            RecurrentEventData(df=df)

    def test_exposure_cannot_exceed_one(self):
        df = pd.DataFrame({
            "policy_id": ["A"], "tstart": [0.0], "tstop": [1.0],
            "event": [0], "exposure": [1.5],
        })
        with pytest.raises(ValueError, match="exposure"):
            RecurrentEventData(df=df)

    def test_invalid_time_scale_raises(self):
        df = pd.DataFrame({
            "policy_id": ["A"], "tstart": [0.0], "tstop": [1.0],
            "event": [0], "exposure": [1.0],
        })
        with pytest.raises(ValueError, match="time_scale"):
            RecurrentEventData(df=df, time_scale="calendar")


class TestRecurrentEventDataProperties:
    def test_n_policies(self, minimal_data):
        assert minimal_data.n_policies == 3

    def test_n_events(self, minimal_counting_process_df):
        data = RecurrentEventData(df=minimal_counting_process_df)
        # A: 0, B: 2, C: 4 => 6
        assert data.n_events == 6

    def test_n_rows(self, minimal_counting_process_df):
        data = RecurrentEventData(df=minimal_counting_process_df)
        assert data.n_rows == len(minimal_counting_process_df)

    def test_policy_ids_returns_list(self, minimal_data):
        ids = minimal_data.policy_ids
        assert isinstance(ids, list)
        assert set(ids) == {"A", "B", "C"}

    def test_covariate_cols_excludes_reserved(self):
        df = pd.DataFrame({
            "policy_id": ["A"], "tstart": [0.0], "tstop": [1.0],
            "event": [0], "exposure": [1.0],
            "age_band": ["30-40"], "vehicle_class": [1],
        })
        data = RecurrentEventData(df=df)
        assert "age_band" in data.covariate_cols
        assert "vehicle_class" in data.covariate_cols
        assert "tstart" not in data.covariate_cols

    def test_repr_contains_key_info(self, minimal_data):
        r = repr(minimal_data)
        assert "n_policies=3" in r
        assert "gap" in r

    def test_summary_runs(self, minimal_data):
        s = minimal_data.summary()
        assert "Policies" in s
        assert "Events" in s


class TestRecurrentEventDataFromRecords:
    def test_from_records_basic(self):
        records = [
            {"policy_id": "P1", "tstart": 0.0, "tstop": 0.5, "event": 1, "exposure": 0.5},
            {"policy_id": "P1", "tstart": 0.0, "tstop": 0.5, "event": 0, "exposure": 0.5},
        ]
        data = RecurrentEventData.from_records(records)
        assert data.n_policies == 1
        assert data.n_events == 1


class TestRecurrentEventDataFromPolicyClaims:
    def test_from_policy_claims_basic(self, policies_df, claims_df):
        data = RecurrentEventData.from_policy_claims(
            policies=policies_df,
            claims=claims_df,
        )
        assert data.n_policies == 4  # all 4 policies, including zero-claimer
        assert data.n_events >= 6   # at least the 6 claims

    def test_zero_claimer_appears(self, policies_df, claims_df):
        data = RecurrentEventData.from_policy_claims(
            policies=policies_df,
            claims=claims_df,
        )
        # FL003 should still appear (as a single censored interval)
        ids = data.policy_ids
        assert "FL003" in ids

    def test_covariates_preserved(self, policies_df, claims_df):
        data = RecurrentEventData.from_policy_claims(
            policies=policies_df,
            claims=claims_df,
            covariate_cols=["fleet_size"],
        )
        assert "fleet_size" in data.covariate_cols

    def test_gap_vs_total_time(self, policies_df, claims_df):
        gap = RecurrentEventData.from_policy_claims(
            policies=policies_df, claims=claims_df, time_scale="gap"
        )
        total = RecurrentEventData.from_policy_claims(
            policies=policies_df, claims=claims_df, time_scale="total"
        )
        assert gap.time_scale == "gap"
        assert total.time_scale == "total"
        # Both should have same number of events
        assert gap.n_events == total.n_events

    def test_gap_time_resets_after_claim(self, policies_df, claims_df):
        data = RecurrentEventData.from_policy_claims(
            policies=policies_df, claims=claims_df, time_scale="gap"
        )
        # In gap time, tstart for event intervals should always be 0
        event_rows = data.df[data.df["event"] == 1]
        assert (event_rows["tstart"] == 0.0).all()


class TestGetPolicyArrays:
    def test_returns_all_policies(self, minimal_data):
        arrays = minimal_data.get_policy_arrays()
        assert set(arrays.keys()) == {"A", "B", "C"}

    def test_arrays_correct_shape(self, minimal_data):
        arrays = minimal_data.get_policy_arrays()
        tstart, tstop, event, exposure, X = arrays["B"]
        assert len(tstart) == len(tstop) == len(event) == len(exposure)
        assert X is None  # no covariates requested

    def test_covariate_array_shape(self):
        df = pd.DataFrame({
            "policy_id": ["A", "A"],
            "tstart": [0.0, 0.0],
            "tstop": [0.5, 0.5],
            "event": [1, 0],
            "exposure": [0.5, 0.5],
            "age": [35.0, 35.0],
            "vehicle": [1.0, 1.0],
        })
        data = RecurrentEventData(df=df)
        arrays = data.get_policy_arrays(covariates=["age", "vehicle"])
        _, _, _, _, X = arrays["A"]
        assert X is not None
        assert X.shape == (2, 2)
