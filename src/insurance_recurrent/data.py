"""
RecurrentEventData: data preparation for recurrent event models.

The core job of this class is to convert policy-level claims histories into the
counting-process format that survival models expect. This is where most real-world
pain lives: policies start at different times, claims reset the gap clock,
exposures vary, and there are edge cases everywhere.

Counting process format (Anderson-Gill style):
  Each row = one risk interval [tstart, tstop)
  event = 1 if a claim occurs at tstop, 0 if censored or just an interval boundary

Gap time vs total time:
  - Gap time: tstart resets to 0 after each claim. Models time-between-claims.
  - Total time: tstart is the calendar time since policy inception.
    Gap time is more natural for most insurance applications (claim rate
    as a function of time since last claim) but either can be appropriate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ._types import FloatArray, IntArray


@dataclass
class RecurrentEventData:
    """
    Counting-process representation of recurrent event data for insurance.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame in counting-process form. Required columns:
          - policy_id: str/int identifier
          - tstart: float, start of risk interval
          - tstop: float, end of risk interval (tstop > tstart)
          - event: int (0 or 1)
          - exposure: float (0, 1] — fraction of interval at risk
        Optional covariate columns can be anything else.

    time_scale : {"gap", "total"}
        Whether time resets after each event (gap) or accumulates (total).

    Examples
    --------
    Build from a raw events list:

    >>> records = [
    ...     {"policy_id": "P001", "tstart": 0.0, "tstop": 0.3, "event": 1, "exposure": 1.0},
    ...     {"policy_id": "P001", "tstart": 0.0, "tstop": 0.7, "event": 0, "exposure": 1.0},
    ... ]
    >>> data = RecurrentEventData.from_records(records)
    """

    df: pd.DataFrame
    time_scale: str = "gap"

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        required = {"policy_id", "tstart", "tstop", "event", "exposure"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")
        if (self.df["tstop"] <= self.df["tstart"]).any():
            raise ValueError("All intervals must have tstop > tstart")
        if not self.df["event"].isin([0, 1]).all():
            raise ValueError("event column must be binary (0 or 1)")
        if (self.df["exposure"] <= 0).any() or (self.df["exposure"] > 1).any():
            raise ValueError("exposure must be in (0, 1]")
        if self.time_scale not in ("gap", "total"):
            raise ValueError(f"time_scale must be 'gap' or 'total', got {self.time_scale!r}")

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_records(
        cls,
        records: list[dict],
        time_scale: str = "gap",
    ) -> "RecurrentEventData":
        """Build from a list of ClaimRecord dicts."""
        return cls(df=pd.DataFrame(records), time_scale=time_scale)

    @classmethod
    def from_policy_claims(
        cls,
        policies: pd.DataFrame,
        claims: pd.DataFrame,
        policy_id_col: str = "policy_id",
        inception_col: str = "inception_date",
        expiry_col: str = "expiry_date",
        claim_date_col: str = "claim_date",
        covariate_cols: Optional[list[str]] = None,
        time_scale: str = "gap",
        observation_end: Optional[pd.Timestamp] = None,
    ) -> "RecurrentEventData":
        """
        Build from a policies table and a claims table in date format.

        Parameters
        ----------
        policies : pd.DataFrame
            One row per policy. Must have policy_id, inception_date, expiry_date
            and any covariate columns you want to use as predictors.
        claims : pd.DataFrame
            One row per claim. Must have policy_id and claim_date.
        policy_id_col : str
            Name of the policy ID column (must be same in both tables).
        inception_col, expiry_col, claim_date_col : str
            Column names in the respective tables.
        covariate_cols : list[str], optional
            Columns from policies to include as covariates.
        time_scale : {"gap", "total"}
            Whether to use gap time or total time since inception.
        observation_end : pd.Timestamp, optional
            If provided, policies observed beyond expiry are censored here.
            Defaults to max(expiry_date) in the policies table.

        Notes
        -----
        Dates are converted to years (using 365.25 days/year) relative to
        each policy's inception date.
        """
        policies = policies.copy()
        claims = claims.copy()

        # Normalise date columns
        for col in [inception_col, expiry_col]:
            policies[col] = pd.to_datetime(policies[col])
        claims[claim_date_col] = pd.to_datetime(claims[claim_date_col])

        if observation_end is None:
            observation_end = policies[expiry_col].max()
        else:
            observation_end = pd.Timestamp(observation_end)

        covariate_cols = covariate_cols or []
        rows = []

        for _, policy in policies.iterrows():
            pid = policy[policy_id_col]
            inception = policy[inception_col]
            expiry = min(policy[expiry_col], observation_end)

            if expiry <= inception:
                continue  # degenerate policy, skip

            total_duration = (expiry - inception).days / 365.25

            # Claims for this policy, sorted
            pclaims = (
                claims[claims[policy_id_col] == pid][claim_date_col]
                .sort_values()
                .values
            )
            # Filter to claims within the observation window
            pclaims = [
                c for c in pclaims
                if inception < pd.Timestamp(c) <= expiry
            ]

            covs = {c: policy[c] for c in covariate_cols}

            # Build counting process intervals
            prev_time_total = 0.0  # total time since inception
            prev_time_gap = 0.0    # gap time since last claim

            for claim_dt in pclaims:
                t_total = (pd.Timestamp(claim_dt) - inception).days / 365.25
                if t_total <= prev_time_total:
                    continue  # same-day double claim — merge

                if time_scale == "gap":
                    tstart = prev_time_gap
                    tstop = t_total - prev_time_total
                else:
                    tstart = prev_time_total
                    tstop = t_total

                duration = tstop - tstart
                row = {
                    policy_id_col: pid,
                    "tstart": tstart,
                    "tstop": tstop,
                    "event": 1,
                    "exposure": min(duration, 1.0) if duration <= 1.0 else 1.0,
                }
                row.update(covs)
                rows.append(row)

                prev_time_total = t_total
                prev_time_gap = 0.0  # reset gap after claim

            # Censoring interval (from last claim/inception to end)
            if time_scale == "gap":
                tstart = prev_time_gap
                tstop = total_duration - prev_time_total
            else:
                tstart = prev_time_total
                tstop = total_duration

            if tstop > tstart:
                duration = tstop - tstart
                row = {
                    policy_id_col: pid,
                    "tstart": tstart,
                    "tstop": tstop,
                    "event": 0,
                    "exposure": min(duration, 1.0) if duration <= 1.0 else 1.0,
                }
                row.update(covs)
                rows.append(row)

        df = pd.DataFrame(rows)
        if "policy_id" not in df.columns and policy_id_col != "policy_id":
            df = df.rename(columns={policy_id_col: "policy_id"})

        return cls(df=df, time_scale=time_scale)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_policies(self) -> int:
        return self.df["policy_id"].nunique()

    @property
    def n_events(self) -> int:
        return int(self.df["event"].sum())

    @property
    def n_rows(self) -> int:
        return len(self.df)

    @property
    def policy_ids(self) -> list:
        return list(self.df["policy_id"].unique())

    @property
    def covariate_cols(self) -> list[str]:
        """Columns that are not the standard counting-process columns."""
        reserved = {"policy_id", "tstart", "tstop", "event", "exposure"}
        return [c for c in self.df.columns if c not in reserved]

    # ------------------------------------------------------------------
    # Data extraction helpers used by model fitters
    # ------------------------------------------------------------------

    def get_policy_arrays(
        self,
        covariates: Optional[list[str]] = None,
    ) -> dict[str, tuple[FloatArray, FloatArray, IntArray, FloatArray, Optional[FloatArray]]]:
        """
        Return per-policy arrays for the EM algorithm.

        Returns a dict keyed by policy_id, each value is:
          (tstart, tstop, event, exposure, X)
        where X is the covariate matrix (n_intervals x n_covs) or None.
        """
        covariates = covariates or []
        result: dict = {}
        for pid, group in self.df.groupby("policy_id", sort=False):
            g = group.sort_values("tstart")
            tstart = g["tstart"].values.astype(float)
            tstop = g["tstop"].values.astype(float)
            event = g["event"].values.astype(int)
            exposure = g["exposure"].values.astype(float)
            X = g[covariates].values.astype(float) if covariates else None
            result[pid] = (tstart, tstop, event, exposure, X)
        return result

    def summary(self) -> str:
        """Text summary of the dataset."""
        lines = [
            f"RecurrentEventData ({self.time_scale} time)",
            f"  Policies:    {self.n_policies:,}",
            f"  Events:      {self.n_events:,}",
            f"  Intervals:   {self.n_rows:,}",
            f"  Covariates:  {self.covariate_cols}",
        ]
        if self.n_policies > 0:
            events_per_policy = self.df.groupby("policy_id")["event"].sum()
            lines += [
                f"  Events/policy: mean={events_per_policy.mean():.2f}, "
                f"max={events_per_policy.max():.0f}",
                f"  % zero-claimers: {(events_per_policy == 0).mean() * 100:.1f}%",
            ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"RecurrentEventData(n_policies={self.n_policies}, "
            f"n_events={self.n_events}, time_scale={self.time_scale!r})"
        )
