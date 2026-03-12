"""
Shared test fixtures for insurance-recurrent.

Using realistic insurance-flavoured data throughout — not toy arrays.
Fleet insurance is the canonical use case: trucks with multiple at-fault
incidents per year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_recurrent import RecurrentEventData, RecurrentEventSimulator


@pytest.fixture(scope="session")
def small_sim_data():
    """Small simulated dataset with known frailty structure. 100 policies."""
    sim = RecurrentEventSimulator(
        n_policies=100,
        theta=0.6,
        baseline_rate=0.4,
        coef={"age_cat": 0.3},
        seed=42,
    )
    return sim.simulate(time_scale="gap")


@pytest.fixture(scope="session")
def medium_sim_data():
    """Medium simulated dataset. 500 policies, suitable for fitting tests."""
    sim = RecurrentEventSimulator(
        n_policies=500,
        theta=0.5,
        baseline_rate=0.3,
        coef={"age_cat": 0.4, "vehicle_type": -0.2},
        seed=123,
    )
    return sim.simulate(time_scale="gap")


@pytest.fixture(scope="session")
def sim_with_frailty(medium_sim_data):
    """Dataset with true frailty values for validation."""
    sim = RecurrentEventSimulator(
        n_policies=300,
        theta=0.8,
        baseline_rate=0.3,
        seed=99,
    )
    return sim.simulate(return_true_frailty=True)


@pytest.fixture(scope="session")
def minimal_counting_process_df():
    """
    A minimal hand-crafted counting process DataFrame.
    Three policies: one zero-claimer, one with 2 claims, one with 4 claims.
    """
    rows = [
        # Policy A: no claims — just one censoring interval
        {"policy_id": "A", "tstart": 0.0, "tstop": 1.0, "event": 0, "exposure": 1.0},
        # Policy B: 2 claims in gap time
        {"policy_id": "B", "tstart": 0.0, "tstop": 0.4, "event": 1, "exposure": 0.4},
        {"policy_id": "B", "tstart": 0.0, "tstop": 0.3, "event": 1, "exposure": 0.3},
        {"policy_id": "B", "tstart": 0.0, "tstop": 0.3, "event": 0, "exposure": 0.3},
        # Policy C: 4 claims
        {"policy_id": "C", "tstart": 0.0, "tstop": 0.2, "event": 1, "exposure": 0.2},
        {"policy_id": "C", "tstart": 0.0, "tstop": 0.2, "event": 1, "exposure": 0.2},
        {"policy_id": "C", "tstart": 0.0, "tstop": 0.2, "event": 1, "exposure": 0.2},
        {"policy_id": "C", "tstart": 0.0, "tstop": 0.2, "event": 1, "exposure": 0.2},
        {"policy_id": "C", "tstart": 0.0, "tstop": 0.2, "event": 0, "exposure": 0.2},
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def minimal_data(minimal_counting_process_df):
    return RecurrentEventData(df=minimal_counting_process_df)


@pytest.fixture(scope="session")
def policies_df():
    """Policy table for from_policy_claims tests."""
    return pd.DataFrame([
        {"policy_id": "FL001", "inception_date": "2022-01-01", "expiry_date": "2023-01-01", "fleet_size": "small"},
        {"policy_id": "FL002", "inception_date": "2022-03-01", "expiry_date": "2023-03-01", "fleet_size": "large"},
        {"policy_id": "FL003", "inception_date": "2022-06-01", "expiry_date": "2023-06-01", "fleet_size": "small"},
        {"policy_id": "FL004", "inception_date": "2022-01-01", "expiry_date": "2024-01-01", "fleet_size": "medium"},
    ])


@pytest.fixture(scope="session")
def claims_df():
    """Claims table for from_policy_claims tests."""
    return pd.DataFrame([
        {"policy_id": "FL001", "claim_date": "2022-04-15"},
        {"policy_id": "FL001", "claim_date": "2022-09-20"},
        {"policy_id": "FL002", "claim_date": "2022-07-01"},
        {"policy_id": "FL004", "claim_date": "2022-05-01"},
        {"policy_id": "FL004", "claim_date": "2022-11-15"},
        {"policy_id": "FL004", "claim_date": "2023-06-20"},
        # FL003: no claims (zero-claimer)
    ])
