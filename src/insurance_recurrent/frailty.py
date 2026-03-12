"""
SharedFrailtyModel: gamma shared frailty for recurrent insurance claims.

The model
---------
Individual i has a latent frailty u_i ~ Gamma(1/theta, 1/theta), so E[u_i]=1
and Var[u_i]=theta. This multiplicatively shifts their baseline hazard:

    h_i(t) = u_i * h_0(t) * exp(X_i beta)

For insurance: u_i > 1 means an inherently higher-risk policyholder,
u_i < 1 means lower risk. After observing their claim history, we update
our beliefs via the posterior, which for gamma frailty has a closed form.

EM algorithm
------------
E-step: Compute E[u_i | data] and E[log u_i | data] using the gamma
        conjugate posterior.

M-step: Update beta (regression coefficients) and theta (frailty variance)
        using the expected log-likelihood with frailty replaced by E[u_i].

Breslow baseline hazard: non-parametric step function, updated in the M-step.

Bühlmann-Straub connection
--------------------------
The posterior mean E[u_i | N_i, exposure_i] is:

    E[u_i | data] = Z_i * (N_i / Lambda_i) + (1 - Z_i) * 1.0

where Z_i = Lambda_i / (Lambda_i + 1/theta) is the credibility factor,
N_i is observed events, Lambda_i = exp(X_i beta) * cumulative baseline hazard.

This is exactly Bühlmann-Straub credibility: the posterior mean blends the
individual's observed rate with the population mean, weighted by how much
data they have. theta is the credibility variance parameter.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import digamma, gammaln

from .data import RecurrentEventData
from ._types import FloatArray, FrailtyFitResult, FrailtyPrediction


class SharedFrailtyModel:
    """
    Gamma shared frailty model for recurrent events.

    Each policyholder shares a single frailty across all their claim events.
    This captures unobserved heterogeneity — the part of an individual's
    claim propensity that isn't explained by their rating factors.

    Parameters
    ----------
    theta_init : float
        Starting value for frailty variance. 1.0 is a reasonable default.
        0.0 would mean no frailty (pure Poisson).
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence criterion on relative change in log-likelihood.
    verbose : bool
        Print iteration progress.

    Examples
    --------
    >>> from insurance_recurrent import SharedFrailtyModel, RecurrentEventData
    >>> model = SharedFrailtyModel(theta_init=0.5)
    >>> model.fit(data, covariates=["age_band", "nclaims_prev"])
    >>> predictions = model.predict_frailty(data)
    """

    def __init__(
        self,
        theta_init: float = 1.0,
        max_iter: int = 200,
        tol: float = 1e-6,
        verbose: bool = False,
    ) -> None:
        if theta_init <= 0:
            raise ValueError("theta_init must be positive")
        self.theta_init = theta_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        # Set after fitting
        self.theta_: float
        self.coef_: FloatArray
        self.covariate_names_: list[str]
        self.baseline_hazard_: pd.DataFrame  # columns: time, hazard, cumulative_hazard
        self.log_likelihood_: float
        self.n_iter_: int
        self.converged_: bool
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        data: RecurrentEventData,
        covariates: Optional[Sequence[str]] = None,
    ) -> "SharedFrailtyModel":
        """
        Fit the shared frailty model via EM.

        Parameters
        ----------
        data : RecurrentEventData
            Prepared counting-process dataset.
        covariates : list[str], optional
            Covariate columns to include in the linear predictor.
            If None, fits intercept-only (pure frailty model).

        Returns
        -------
        self
        """
        covariates = list(covariates) if covariates else []
        self.covariate_names_ = covariates

        # Pull global arrays from data
        df = data.df.copy()
        df["_lp"] = 0.0  # log linear predictor, updated in M-step
        n_covs = len(covariates)
        self.coef_ = np.zeros(n_covs)

        theta = self.theta_init
        policy_ids = data.policy_ids
        policy_arrays = data.get_policy_arrays(covariates)

        prev_ll = -np.inf
        for iteration in range(1, self.max_iter + 1):
            # ---- Compute linear predictor --------------------------------
            if n_covs > 0:
                X_global = df[covariates].values.astype(float)
                df["_lp"] = X_global @ self.coef_
            else:
                df["_lp"] = 0.0

            # ---- E-step: compute Breslow baseline + posterior frailty ----
            baseline = self._breslow_baseline(df)
            # Merge cumulative hazard back to df
            df = df.merge(
                baseline[["time", "cumhaz"]].rename(columns={"time": "tstop"}),
                on="tstop",
                how="left",
            )
            # For intervals with same tstop, cumhaz is the cumulative to that point
            df["cumhaz"] = df["cumhaz"].ffill().fillna(0.0)

            # Per-policy posterior parameters
            posteriors = self._e_step(df, policy_arrays, theta, covariates)

            # ---- M-step: update beta and theta ---------------------------
            theta_new, coef_new = self._m_step(
                df, policy_arrays, posteriors, theta, covariates
            )
            self.coef_ = coef_new
            theta = theta_new

            # ---- Compute observed log-likelihood -------------------------
            ll = self._marginal_log_likelihood(df, policy_arrays, theta, covariates)

            rel_change = abs(ll - prev_ll) / (abs(prev_ll) + 1e-10)
            if self.verbose:
                print(
                    f"Iter {iteration:4d}: ll={ll:.4f}, theta={theta:.4f}, "
                    f"rel_change={rel_change:.2e}"
                )

            if rel_change < self.tol and iteration > 1:
                self.converged_ = True
                break
            prev_ll = ll
            # Drop the cumhaz column before next iteration re-merge
            df = df.drop(columns=["cumhaz"], errors="ignore")
        else:
            self.converged_ = False
            warnings.warn(
                f"EM did not converge in {self.max_iter} iterations. "
                "Try increasing max_iter or loosening tol.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Final baseline with converged parameters
        if n_covs > 0:
            df["_lp"] = df[covariates].values.astype(float) @ self.coef_
        baseline = self._breslow_baseline(df)

        self.theta_ = theta
        self.baseline_hazard_ = baseline
        self.log_likelihood_ = ll
        self.n_iter_ = iteration
        self._is_fitted = True
        self._data_ref = data  # keep ref for diagnostics

        return self

    def _breslow_baseline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Non-parametric Breslow estimator of the cumulative baseline hazard.

        H_0(t) = sum_{s <= t} [ dN(s) / sum_{j at risk at s} exp(X_j beta) * exposure_j ]

        We group by tstop because multiple policies may have events at the same time.
        """
        event_rows = df[df["event"] == 1].copy()
        if len(event_rows) == 0:
            return pd.DataFrame({"time": [0.0], "dN": [0.0], "denom": [1.0], "dhaz": [0.0], "cumhaz": [0.0]})

        # All distinct event times
        event_times = np.sort(event_rows["tstop"].unique())

        rows = []
        for t in event_times:
            # Risk set: intervals with tstart < t <= tstop
            at_risk = df[(df["tstart"] < t) & (df["tstop"] >= t)]
            dN = float(event_rows[event_rows["tstop"] == t]["event"].sum())
            denom = float(
                (np.exp(at_risk["_lp"].values) * at_risk["exposure"].values).sum()
            )
            if denom <= 0:
                denom = 1e-10
            rows.append({"time": t, "dN": dN, "denom": denom, "dhaz": dN / denom})

        bh = pd.DataFrame(rows)
        bh["cumhaz"] = bh["dhaz"].cumsum()
        return bh

    def _e_step(
        self,
        df: pd.DataFrame,
        policy_arrays: dict,
        theta: float,
        covariates: list[str],
    ) -> dict[str, dict]:
        """
        Compute posterior gamma parameters for each policy's frailty.

        Gamma frailty prior: u ~ Gamma(a0, b0) with a0 = b0 = 1/theta
        Gamma posterior: u | data ~ Gamma(a_i, b_i) where:
          a_i = 1/theta + N_i    (shape)
          b_i = 1/theta + Lambda_i  (rate)
        where Lambda_i = sum_j exp(X_ij beta) * H_0(tstop_ij) (expected events without frailty)
        """
        # Build cumhaz lookup
        bh = self.baseline_hazard_ if hasattr(self, "baseline_hazard_") and self._is_fitted else None

        posteriors = {}
        for pid, (tstart, tstop, event, exposure, X) in policy_arrays.items():
            # Linear predictor for this policy's intervals
            if X is not None and len(covariates) > 0:
                lp = X @ self.coef_
            else:
                lp = np.zeros(len(event))

            # Cumulative hazard at each tstop from Breslow estimate
            # Use the df cumhaz column we merged in
            policy_df = df[df["policy_id"] == pid]
            cumhaz_vals = policy_df["cumhaz"].values
            if len(cumhaz_vals) == 0:
                cumhaz_vals = np.zeros(len(event))

            # Expected cumulative intensity for each interval
            # For gap time: contribution = exp(lp) * (H0(tstop) - H0(tstart))
            Lambda_i = float(np.sum(np.exp(lp) * exposure * cumhaz_vals))

            N_i = float(event.sum())
            a0 = 1.0 / theta
            a_post = a0 + N_i
            b_post = a0 + Lambda_i

            # Posterior moments
            E_u = a_post / b_post          # posterior mean
            E_log_u = digamma(a_post) - np.log(b_post)  # posterior E[log u]

            posteriors[pid] = {
                "E_u": E_u,
                "E_log_u": E_log_u,
                "N_i": N_i,
                "Lambda_i": Lambda_i,
                "a_post": a_post,
                "b_post": b_post,
            }

        return posteriors

    def _m_step(
        self,
        df: pd.DataFrame,
        policy_arrays: dict,
        posteriors: dict,
        theta: float,
        covariates: list[str],
    ) -> tuple[float, FloatArray]:
        """
        Update theta and beta given posterior frailty estimates.

        Theta update (closed form for gamma frailty):
          Q(theta) = n * (1/theta) * log(1/theta) - n * log(Gamma(1/theta))
                     + (1/theta - 1) * sum E[log u_i] - (1/theta) * sum E[u_i]
          dQ/d(1/theta) = 0  =>  closed form not available, use 1D optimisation.

        Beta update: Partial likelihood with u_i replaced by E[u_i]:
          l(beta) = sum_i sum_j [event_ij * (log E[u_i] + X_ij beta + log h0(tstop))
                                 - E[u_i] * exp(X_ij beta) * H0(tstop) * exposure]
          This is a weighted Poisson log-likelihood.
        """
        n_policies = len(posteriors)

        # --- Theta update via 1D numerical optimisation ---
        def neg_q_theta(log_theta: float) -> float:
            th = np.exp(log_theta)
            a0 = 1.0 / th
            q = (
                n_policies * (a0 * np.log(a0) - gammaln(a0))
                + (a0 - 1.0) * sum(p["E_log_u"] for p in posteriors.values())
                - a0 * sum(p["E_u"] for p in posteriors.values())
            )
            return -q

        res = minimize(neg_q_theta, x0=np.log(theta), method="L-BFGS-B")
        theta_new = float(np.exp(res.x[0]))
        theta_new = np.clip(theta_new, 1e-4, 20.0)

        # --- Beta update via Newton-Raphson (IRLS-style) ---
        if len(covariates) == 0:
            return theta_new, np.zeros(0)

        # Build weighted design matrix using E[u_i] as weight adjustment
        # Objective: weighted Poisson deviance
        # We treat this as a Poisson GLM with offset = log(E[u_i]) + log(exposure)
        # and response = event count per interval.
        rows = []
        for pid, (tstart, tstop, event, exposure, X) in policy_arrays.items():
            E_u = posteriors[pid]["E_u"]
            policy_df = df[df["policy_id"] == pid].sort_values("tstart")
            cumhaz = policy_df["cumhaz"].values
            for k in range(len(event)):
                rows.append({
                    "event": event[k],
                    "E_u": E_u,
                    "exposure": exposure[k],
                    "cumhaz": cumhaz[k] if k < len(cumhaz) else 0.0,
                    **{f"x{j}": X[k, j] for j in range(X.shape[1])},
                })

        agg = pd.DataFrame(rows)
        X_mat = agg[[f"x{j}" for j in range(len(covariates))]].values.astype(float)
        y = agg["event"].values.astype(float)
        E_u = agg["E_u"].values
        exposure = agg["exposure"].values
        cumhaz = agg["cumhaz"].values

        # IRLS for Poisson with offset
        # mu = E_u * exposure * cumhaz * exp(X beta)
        # => log(mu) = log(E_u) + log(exposure) + log(cumhaz) + X beta
        # where log(cumhaz) treated as offset but we can have zeros...
        # Safe approach: weighted Poisson negative log-likelihood minimization.
        def neg_ll_beta(beta: FloatArray) -> float:
            lp = X_mat @ beta
            mu = E_u * exposure * (cumhaz + 1e-10) * np.exp(lp)
            mu = np.clip(mu, 1e-15, None)
            return -float(np.sum(y * np.log(mu) - mu))

        def grad_beta(beta: FloatArray) -> FloatArray:
            lp = X_mat @ beta
            mu = E_u * exposure * (cumhaz + 1e-10) * np.exp(lp)
            mu = np.clip(mu, 1e-15, None)
            resid = y - mu
            return -float(1.0) * (X_mat.T @ resid)

        res = minimize(
            neg_ll_beta,
            x0=self.coef_,
            jac=grad_beta,
            method="L-BFGS-B",
            options={"maxiter": 100},
        )
        coef_new = res.x

        return theta_new, coef_new

    def _marginal_log_likelihood(
        self,
        df: pd.DataFrame,
        policy_arrays: dict,
        theta: float,
        covariates: list[str],
    ) -> float:
        """
        Marginal log-likelihood integrating out frailty analytically.

        For gamma frailty with shape=1/theta, the marginal likelihood is:
          L_i = Gamma(1/theta + N_i) / Gamma(1/theta)
                * (1/theta)^(1/theta) / (1/theta + Lambda_i)^(1/theta + N_i)
                * prod_j [h_ij * exp(X_ij beta)]^event_ij
        """
        a0 = 1.0 / theta
        ll = 0.0

        bh_times = self.baseline_hazard_["time"].values if hasattr(self, "baseline_hazard_") else np.array([])
        bh_dhaz = self.baseline_hazard_["dhaz"].values if len(bh_times) > 0 else np.array([])

        for pid, (tstart, tstop, event, exposure, X) in policy_arrays.items():
            if X is not None and len(covariates) > 0:
                lp = X @ self.coef_
            else:
                lp = np.zeros(len(event))

            policy_df = df[df["policy_id"] == pid]
            cumhaz = policy_df["cumhaz"].values
            N_i = float(event.sum())
            Lambda_i = float(np.sum(np.exp(lp) * exposure * (cumhaz + 1e-10)))

            # Gamma marginalisation
            ll += gammaln(a0 + N_i) - gammaln(a0)
            ll += a0 * np.log(a0) - (a0 + N_i) * np.log(a0 + Lambda_i)

            # Contribution from event times (baseline hazard at event times)
            event_tstops = tstop[event == 1]
            for t in event_tstops:
                idx = np.searchsorted(bh_times, t, side="right") - 1
                if idx >= 0 and idx < len(bh_dhaz):
                    dhaz = bh_dhaz[idx]
                else:
                    dhaz = 1e-10
                # Add log(lp contribution)
                k = np.where(tstop == t)[0]
                if len(k) > 0:
                    ll += lp[k[0]]
                ll += np.log(max(dhaz, 1e-15))

        return ll

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_frailty(
        self,
        data: RecurrentEventData,
    ) -> list[FrailtyPrediction]:
        """
        Predict posterior frailty for each policy in the dataset.

        Returns the Bühlmann credibility estimate of each policy's risk level.
        A frailty_mean of 1.5 means the model believes this policyholder is
        50% more likely to claim than a random policyholder with the same
        rating factors.

        Parameters
        ----------
        data : RecurrentEventData
            Dataset containing the policies to predict. Can be the training
            data (in-sample) or new data with the same structure.

        Returns
        -------
        list[FrailtyPrediction]
            Sorted by policy_id.
        """
        self._check_fitted()

        covariates = self.covariate_names_
        policy_arrays = data.get_policy_arrays(covariates)

        # Merge cumulative hazard into data df
        df = data.df.copy()
        if len(covariates) > 0:
            df["_lp"] = df[covariates].values.astype(float) @ self.coef_
        else:
            df["_lp"] = 0.0

        bh = self.baseline_hazard_
        df = df.merge(
            bh[["time", "cumhaz"]].rename(columns={"time": "tstop"}),
            on="tstop",
            how="left",
        )
        df["cumhaz"] = df["cumhaz"].ffill().fillna(0.0)

        results = []
        theta = self.theta_

        for pid, (tstart, tstop, event, exposure, X) in sorted(policy_arrays.items(), key=lambda x: str(x[0])):
            if X is not None and len(covariates) > 0:
                lp = X @ self.coef_
            else:
                lp = np.zeros(len(event))

            policy_df = df[df["policy_id"] == pid].sort_values("tstart")
            cumhaz_vals = policy_df["cumhaz"].values

            N_i = float(event.sum())
            Lambda_i = float(np.sum(np.exp(lp) * exposure * (cumhaz_vals + 1e-10)))
            total_exposure = float(exposure.sum())

            a0 = 1.0 / theta
            a_post = a0 + N_i
            b_post = a0 + Lambda_i

            E_u = a_post / b_post
            Var_u = a_post / (b_post ** 2)

            # Bühlmann credibility factor
            Z = Lambda_i / (Lambda_i + a0)

            results.append(
                FrailtyPrediction(
                    policy_id=pid,
                    frailty_mean=float(E_u),
                    frailty_var=float(Var_u),
                    credibility_factor=float(Z),
                    n_events=float(N_i),
                    exposure=float(total_exposure),
                )
            )

        return results

    def predict_expected_events(
        self,
        data: RecurrentEventData,
        time_horizon: float = 1.0,
    ) -> pd.DataFrame:
        """
        Predict expected claim count over a future time horizon for each policy.

        This uses the posterior frailty mean: E[N_future | data] = E[u_i | data] * Lambda_base
        where Lambda_base = exp(X_i beta) * H_0(time_horizon).

        Parameters
        ----------
        data : RecurrentEventData
        time_horizon : float
            Future period length in the same units as the event times.

        Returns
        -------
        pd.DataFrame with columns: policy_id, frailty_mean, expected_events
        """
        self._check_fitted()

        frailty_preds = self.predict_frailty(data)

        # Baseline cumulative hazard at time_horizon
        bh = self.baseline_hazard_
        cumhaz_horizon = bh[bh["time"] <= time_horizon]["cumhaz"].max() if len(bh) > 0 else 0.0
        if np.isnan(cumhaz_horizon):
            cumhaz_horizon = 0.0

        covariates = self.covariate_names_
        policy_arrays = data.get_policy_arrays(covariates)

        rows = []
        for fp in frailty_preds:
            pid = fp["policy_id"]
            if pid in policy_arrays:
                _, _, _, _, X = policy_arrays[pid]
                if X is not None and len(covariates) > 0:
                    lp = float(X[-1] @ self.coef_)  # use last interval covariates
                else:
                    lp = 0.0
            else:
                lp = 0.0

            expected = fp["frailty_mean"] * np.exp(lp) * cumhaz_horizon
            rows.append({
                "policy_id": pid,
                "frailty_mean": fp["frailty_mean"],
                "credibility_factor": fp["credibility_factor"],
                "expected_events": expected,
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Summary / diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> FrailtyFitResult:
        """Return fitted model summary dict."""
        self._check_fitted()
        return FrailtyFitResult(
            n_policies=len(self._data_ref.policy_ids),
            n_events=self._data_ref.n_events,
            log_likelihood=self.log_likelihood_,
            theta=self.theta_,
            coef={
                name: float(val)
                for name, val in zip(self.covariate_names_, self.coef_)
            },
            se={},  # standard errors via Hessian — left for v0.2
            converged=self.converged_,
            n_iter=self.n_iter_,
        )

    def print_summary(self) -> None:
        """Print a formatted model summary."""
        self._check_fitted()
        s = self.summary()
        print("=" * 55)
        print("SharedFrailtyModel Summary")
        print("=" * 55)
        print(f"  Policies:          {s['n_policies']:,}")
        print(f"  Events:            {s['n_events']:,}")
        print(f"  Log-likelihood:    {s['log_likelihood']:.4f}")
        print(f"  Frailty variance (theta): {s['theta']:.4f}")
        print(f"  Converged:         {s['converged']} ({s['n_iter']} iters)")
        if s["coef"]:
            print("\n  Coefficients:")
            for name, val in s["coef"].items():
                print(f"    {name:<25s} {val:+.4f}")
        print("=" * 55)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

    @property
    def frailty_variance(self) -> float:
        """Estimated frailty variance theta (higher = more heterogeneity)."""
        self._check_fitted()
        return self.theta_

    @property
    def credibility_factors(self) -> pd.DataFrame:
        """
        Return a DataFrame summarising Bühlmann credibility factors by
        number of observed events.

        Shows how many events a policy needs to earn various levels of
        credibility. Useful for communicating the model to non-technical
        stakeholders.
        """
        self._check_fitted()
        theta = self.theta_
        rows = []
        for n in [0, 1, 2, 3, 5, 10, 20, 50]:
            # With Lambda_i = n (expected = observed, pure count argument)
            Z = n / (n + 1.0 / theta)
            rows.append({"n_events": n, "credibility_factor": Z})
        return pd.DataFrame(rows)
