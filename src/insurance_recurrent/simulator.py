"""
RecurrentEventSimulator: synthetic data generator with known DGP.

The main purpose is testing — we simulate from the exact model that
SharedFrailtyModel and JointFrailtyModel try to fit, then verify we
can recover the true parameters. This is the most reliable way to
validate a frailty implementation because the likelihood surface is
complex and small bugs can produce plausible-looking but wrong answers.

Data generating process:
  - Frailty: u_i ~ Gamma(1/theta, 1/theta) or Lognormal(0, sqrt(theta))
  - Baseline hazard: piecewise constant or Weibull
  - Covariates: drawn from specified distributions
  - Claim times: Poisson process with intensity u_i * h_0(t) * exp(X_i beta)
  - Lapse (terminal event): independent Poisson process (for joint model testing)
  - Observation period: [0, T_max] with left truncation possible
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd

from .data import RecurrentEventData


@dataclass
class RecurrentEventSimulator:
    """
    Simulate recurrent insurance claims with known shared frailty structure.

    Parameters
    ----------
    n_policies : int
        Number of policyholders to simulate.
    theta : float
        True frailty variance. 0 = no frailty (pure Poisson).
    baseline_rate : float
        Baseline claim rate per unit time (scale of the hazard).
    coef : dict[str, float]
        True regression coefficients. Keys become covariate column names.
    frailty_dist : {"gamma", "lognormal"}
        Marginal distribution of the frailty. Gamma is conjugate; lognormal
        is often more realistic for insurance (heavier right tail).
    observation_period : float
        Maximum observation window length in years.
    left_truncation_rate : float
        Fraction of policies with left truncation (mid-term inception).
        0 = no truncation.
    lapse_rate : float
        Background lapse hazard rate. 0 = no lapses (simple censoring at T_max).
        When > 0, policies lapse randomly (informative censoring for joint model).
    lapse_frailty_assoc : float
        Association between claim frailty and lapse hazard.
        0 = independent; 1 = same frailty drives both claims and lapse.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> sim = RecurrentEventSimulator(n_policies=500, theta=0.8, baseline_rate=0.3)
    >>> data = sim.simulate()
    >>> print(data.summary())
    """

    n_policies: int = 500
    theta: float = 0.5
    baseline_rate: float = 0.3
    coef: dict[str, float] = field(default_factory=dict)
    frailty_dist: Literal["gamma", "lognormal"] = "gamma"
    observation_period: float = 5.0
    left_truncation_rate: float = 0.0
    lapse_rate: float = 0.0
    lapse_frailty_assoc: float = 0.0
    seed: Optional[int] = None

    def simulate(
        self,
        time_scale: str = "gap",
        return_true_frailty: bool = False,
    ) -> RecurrentEventData | tuple[RecurrentEventData, pd.DataFrame]:
        """
        Simulate a dataset.

        Parameters
        ----------
        time_scale : {"gap", "total"}
            Time scale for the returned RecurrentEventData.
        return_true_frailty : bool
            If True, also return a DataFrame with the true frailty values
            for validation.

        Returns
        -------
        RecurrentEventData, or (RecurrentEventData, pd.DataFrame) if
        return_true_frailty=True.
        """
        rng = np.random.default_rng(self.seed)
        rows = []
        true_frailty_rows = []

        for i in range(self.n_policies):
            pid = f"P{i:06d}"

            # Draw frailty
            u = self._draw_frailty(rng)

            # Draw covariates
            covs = self._draw_covariates(rng)
            lp = sum(self.coef.get(k, 0.0) * v for k, v in covs.items())
            intensity = u * self.baseline_rate * np.exp(lp)

            # Left truncation
            if self.left_truncation_rate > 0 and rng.random() < self.left_truncation_rate:
                entry_time = rng.uniform(0, self.observation_period * 0.5)
            else:
                entry_time = 0.0

            # Lapse time (terminal event)
            lapse_intensity = self._compute_lapse_intensity(u, rng)
            if lapse_intensity > 0:
                lapse_time = rng.exponential(1.0 / lapse_intensity)
            else:
                lapse_time = np.inf

            exit_time = min(self.observation_period, lapse_time)
            if exit_time <= entry_time:
                # Policy lapses before it's even observed — skip
                continue

            # Simulate claim process (Poisson with rate = intensity)
            # Time from entry to exit
            duration = exit_time - entry_time
            claim_times_rel = self._simulate_poisson_process(intensity, duration, rng)
            claim_times_abs = [entry_time + t for t in claim_times_rel]

            # Build counting process intervals
            prev_abs = entry_time
            prev_gap = 0.0

            for ct in claim_times_abs:
                t_total = ct
                if time_scale == "gap":
                    tstart = prev_gap
                    tstop = ct - prev_abs
                else:
                    tstart = prev_abs - entry_time
                    tstop = ct - entry_time

                if tstop <= tstart:
                    continue

                interval_dur = tstop - tstart
                rows.append({
                    "policy_id": pid,
                    "tstart": tstart,
                    "tstop": tstop,
                    "event": 1,
                    "exposure": min(interval_dur, 1.0),
                    **covs,
                })
                prev_abs = ct
                prev_gap = 0.0

            # Censoring interval
            if time_scale == "gap":
                tstart = prev_gap
                tstop = exit_time - prev_abs
            else:
                tstart = prev_abs - entry_time
                tstop = exit_time - entry_time

            if tstop > tstart:
                interval_dur = tstop - tstart
                rows.append({
                    "policy_id": pid,
                    "tstart": tstart,
                    "tstop": tstop,
                    "event": 0,
                    "exposure": min(interval_dur, 1.0),
                    **covs,
                })

            true_frailty_rows.append({
                "policy_id": pid,
                "true_frailty": u,
                "n_claims": len(claim_times_abs),
                "observation_time": exit_time - entry_time,
                "lapsed": lapse_time < self.observation_period,
            })

        df = pd.DataFrame(rows)
        if len(df) == 0:
            df = pd.DataFrame(columns=["policy_id", "tstart", "tstop", "event", "exposure"])

        data = RecurrentEventData(df=df, time_scale=time_scale)

        if return_true_frailty:
            return data, pd.DataFrame(true_frailty_rows)
        return data

    def _draw_frailty(self, rng: np.random.Generator) -> float:
        """Draw a single frailty value from the specified distribution."""
        if self.theta == 0.0:
            return 1.0
        if self.frailty_dist == "gamma":
            # Gamma(a, b) with a=b=1/theta => mean=1, var=theta
            a = 1.0 / self.theta
            return float(rng.gamma(shape=a, scale=self.theta))
        elif self.frailty_dist == "lognormal":
            # Lognormal with mean=1: E[exp(Z)] = 1 => mu = -sigma^2/2
            # var = (exp(sigma^2) - 1) = theta => sigma^2 = log(1+theta)
            sigma2 = np.log(1.0 + self.theta)
            sigma = np.sqrt(sigma2)
            mu = -sigma2 / 2.0
            return float(np.exp(rng.normal(mu, sigma)))
        else:
            raise ValueError(f"Unknown frailty_dist: {self.frailty_dist!r}")

    def _draw_covariates(self, rng: np.random.Generator) -> dict[str, float]:
        """Draw covariate values. Currently binary for simplicity."""
        return {name: float(rng.choice([0.0, 1.0])) for name in self.coef.keys()}

    def _compute_lapse_intensity(self, frailty: float, rng: np.random.Generator) -> float:
        """Compute lapse intensity, optionally correlated with claim frailty."""
        if self.lapse_rate == 0.0:
            return 0.0
        if self.lapse_frailty_assoc == 0.0:
            return self.lapse_rate
        # Linear association: high-frailty policies lapse at higher rate
        return self.lapse_rate * (1.0 + self.lapse_frailty_assoc * (frailty - 1.0))

    @staticmethod
    def _simulate_poisson_process(rate: float, duration: float, rng: np.random.Generator) -> list[float]:
        """
        Simulate event times of a homogeneous Poisson process on [0, duration].

        Uses the inter-arrival time method: T_k ~ Exp(rate), cumsum until > duration.
        """
        if rate <= 0 or duration <= 0:
            return []
        times = []
        t = 0.0
        while True:
            gap = rng.exponential(1.0 / rate)
            t += gap
            if t > duration:
                break
            times.append(t)
        return times

    def parameter_table(self) -> pd.DataFrame:
        """Return a table of true DGP parameters for comparison with estimates."""
        rows = [
            {"parameter": "theta (frailty variance)", "true_value": self.theta},
            {"parameter": "baseline_rate", "true_value": self.baseline_rate},
        ]
        for name, val in self.coef.items():
            rows.append({"parameter": f"coef[{name}]", "true_value": val})
        return pd.DataFrame(rows)
