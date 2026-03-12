"""
Diagnostics for frailty models.

Three core diagnostics:
1. Frailty distribution QQ plot — compare empirical posterior means to
   the fitted gamma distribution. Deviations suggest model misspecification.

2. Marginal Cox-Snell residuals — for recurrent events, the cumulative
   intensity at each event time should be ~Exponential(1) if the model
   is correct.

3. Event rate by frailty decile — the most interpretable diagnostic for
   actuarial audiences. Plots observed claim rate vs model-predicted rate
   for each decile of the posterior frailty distribution.

These return data structures (DataFrames and dicts) rather than matplotlib
figures. This keeps a hard dependency on matplotlib out of the base package
(it's an optional extra) and makes the diagnostics testable.
"""

from __future__ import annotations

from typing import Optional
import warnings

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist

from .data import RecurrentEventData
from .frailty import SharedFrailtyModel
from ._types import FrailtyPrediction


def frailty_qq_data(
    model: SharedFrailtyModel,
    data: RecurrentEventData,
) -> pd.DataFrame:
    """
    Compute data for a frailty distribution QQ plot.

    Compares the empirical distribution of posterior frailty means to the
    theoretical gamma distribution with the fitted theta. Perfect model fit
    would show a straight line through the origin with slope 1.

    Parameters
    ----------
    model : SharedFrailtyModel
        Fitted model.
    data : RecurrentEventData
        Dataset (typically the training data).

    Returns
    -------
    pd.DataFrame with columns:
        theoretical : float, theoretical quantiles from Gamma(1/theta, 1/theta)
        empirical : float, empirical frailty quantiles (sorted ascending)
        policy_id : str, policy identifier at each quantile
    """
    predictions = model.predict_frailty(data)
    frailty_means = np.array([p["frailty_mean"] for p in predictions])
    pids = [p["policy_id"] for p in predictions]

    n = len(frailty_means)
    # Sort empirical values and compute uniform quantile positions
    sort_idx = np.argsort(frailty_means)
    sorted_frailty = frailty_means[sort_idx]
    sorted_pids = [pids[i] for i in sort_idx]

    # Plotting positions: (i+0.5)/n (Blom formula without offset, good for gamma)
    quantile_positions = (np.arange(1, n + 1) - 0.5) / n

    theta = model.theta_
    a = 1.0 / theta
    theoretical = gamma_dist.ppf(quantile_positions, a=a, scale=theta)

    return pd.DataFrame({
        "theoretical": theoretical,
        "empirical": sorted_frailty,
        "policy_id": sorted_pids,
    })


def cox_snell_residuals(
    model: SharedFrailtyModel,
    data: RecurrentEventData,
) -> pd.DataFrame:
    """
    Compute marginal Cox-Snell residuals for a fitted frailty model.

    For each risk interval j for policy i:
        r_ij = E[u_i | data] * exp(X_ij beta) * H_0(tstop_ij) * exposure_ij

    If the model is correct, the cumulative residuals per policy should be
    approximately Exponential(1)-distributed.

    Returns
    -------
    pd.DataFrame with columns:
        policy_id, tstart, tstop, event, residual, cumulative_residual
    """
    covariates = model.covariate_names_
    predictions = model.predict_frailty(data)
    frailty_map = {p["policy_id"]: p["frailty_mean"] for p in predictions}

    df = data.df.copy()
    if len(covariates) > 0:
        df["_lp"] = df[covariates].values.astype(float) @ model.coef_
    else:
        df["_lp"] = 0.0

    # Merge cumulative hazard
    bh = model.baseline_hazard_
    df = df.merge(
        bh[["time", "cumhaz"]].rename(columns={"time": "tstop"}),
        on="tstop",
        how="left",
    )
    df["cumhaz"] = df["cumhaz"].ffill().fillna(0.0)

    df["frailty_mean"] = df["policy_id"].map(frailty_map)
    df["residual"] = (
        df["frailty_mean"]
        * np.exp(df["_lp"])
        * df["cumhaz"]
        * df["exposure"]
    )

    df["cumulative_residual"] = df.groupby("policy_id")["residual"].cumsum()

    return df[["policy_id", "tstart", "tstop", "event", "residual", "cumulative_residual"]].copy()


def event_rate_by_frailty_decile(
    model: SharedFrailtyModel,
    data: RecurrentEventData,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """
    Compute observed vs predicted event rates by frailty decile.

    This is the most useful diagnostic for actuarial stakeholders because it
    directly answers: "Does the model correctly rank policyholders by risk?"

    A well-calibrated model should show observed rates tracking predicted rates
    closely across deciles. If the top decile has much higher observed rate
    than predicted, theta is underestimated.

    Parameters
    ----------
    model : SharedFrailtyModel
        Fitted model.
    data : RecurrentEventData
    n_deciles : int
        Number of quantile bands (10 = deciles, 5 = quintiles).

    Returns
    -------
    pd.DataFrame with columns:
        decile : int (1=lowest frailty, n_deciles=highest)
        n_policies : int
        frailty_mean_avg : float, average posterior frailty in this band
        observed_events : float
        total_exposure : float
        observed_rate : float (events per unit exposure)
        predicted_rate : float (model-predicted events per unit exposure)
        lift : float (observed_rate / overall_rate)
    """
    predictions = model.predict_frailty(data)
    pred_df = pd.DataFrame(predictions)

    # Assign deciles
    pred_df["decile"] = pd.qcut(
        pred_df["frailty_mean"],
        q=n_deciles,
        labels=list(range(1, n_deciles + 1)),
        duplicates="drop",
    ).astype(int)

    # Merge with actual data to get observed events and exposure
    event_df = (
        data.df.groupby("policy_id")
        .agg(observed_events=("event", "sum"), total_exposure=("exposure", "sum"))
        .reset_index()
    )

    merged = pred_df.merge(event_df, on="policy_id", how="left")

    overall_rate = merged["observed_events"].sum() / (merged["total_exposure"].sum() + 1e-10)

    result = (
        merged.groupby("decile")
        .agg(
            n_policies=("policy_id", "count"),
            frailty_mean_avg=("frailty_mean", "mean"),
            observed_events=("observed_events", "sum"),
            total_exposure=("total_exposure", "sum"),
        )
        .reset_index()
    )
    result["observed_rate"] = result["observed_events"] / (result["total_exposure"] + 1e-10)
    result["predicted_rate"] = result["frailty_mean_avg"] * overall_rate
    result["lift"] = result["observed_rate"] / (overall_rate + 1e-10)

    return result


def frailty_summary_stats(
    predictions: list[FrailtyPrediction],
    theta: float,
) -> pd.DataFrame:
    """
    Summary statistics of the posterior frailty distribution.

    Useful for the FrailtyReport and for communicating model output to
    non-technical actuarial audiences.

    Returns a long-format DataFrame with stat name and value columns.
    """
    frailty_vals = np.array([p["frailty_mean"] for p in predictions])
    credibility_vals = np.array([p["credibility_factor"] for p in predictions])

    rows = [
        {"statistic": "n_policies", "value": len(frailty_vals)},
        {"statistic": "frailty_mean", "value": frailty_vals.mean()},
        {"statistic": "frailty_std", "value": frailty_vals.std()},
        {"statistic": "frailty_min", "value": frailty_vals.min()},
        {"statistic": "frailty_p25", "value": np.percentile(frailty_vals, 25)},
        {"statistic": "frailty_median", "value": np.median(frailty_vals)},
        {"statistic": "frailty_p75", "value": np.percentile(frailty_vals, 75)},
        {"statistic": "frailty_max", "value": frailty_vals.max()},
        {"statistic": "frailty_variance_est", "value": theta},
        {"statistic": "credibility_mean", "value": credibility_vals.mean()},
        {"statistic": "pct_high_risk", "value": (frailty_vals > 1.5).mean() * 100},
        {"statistic": "pct_low_risk", "value": (frailty_vals < 0.5).mean() * 100},
    ]
    return pd.DataFrame(rows)
