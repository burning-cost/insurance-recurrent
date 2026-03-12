"""
JointFrailtyModel: shared frailty linking recurrent claims and terminal lapse event.

The problem with standard shared frailty models
------------------------------------------------
If high-frailty policyholders (frequent claimers) also lapse more frequently,
the censoring is informative — we never observe as many claims from the
highest-risk policies because they leave early. Standard frailty models that
treat lapse as non-informative censoring will underestimate theta.

The joint model
---------------
A single frailty u_i simultaneously drives:
1. The recurrent claim process: h_claim(t|u) = u * h0_claim(t) * exp(X beta)
2. The terminal lapse process:  h_lapse(t|u) = u^alpha * h0_lapse(t) * exp(W gamma)

alpha is the association parameter:
  alpha = 0: lapse is independent of frailty (reduces to standard model)
  alpha > 0: high-frailty policyholders lapse faster
  alpha < 0: high-frailty policyholders stay longer (churn management)

For insurance pricing, knowing alpha matters:
- If alpha > 0, standard models underestimate the claim rate of retained customers
  (the good risks stay, bad risks leave — adverse selection in reverse)
- If alpha < 0, your renewal book is getting progressively riskier

EM algorithm
------------
The E-step requires numerical integration over the frailty distribution
(gamma marginalisation does not have a closed form for the joint model
because the lapse term adds u^alpha, not u, to the exponent).

We use Gauss-Laguerre quadrature for gamma frailty — this gives exact
results for gamma-distributed frailty with manageable quadrature points.

Reference: Rondeau et al. (2007), "Maximum penalized likelihood estimation
in a gamma-frailty model with right censoring, left truncation, and
multiple times to recurrent events." Lifetime Data Analysis.

Lognormal frailty uses Gauss-Hermite quadrature instead.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, gamma as gamma_fn

from .data import RecurrentEventData
from ._types import FloatArray, FrailtyFitResult, FrailtyPrediction


class JointFrailtyModel:
    """
    Joint frailty model for recurrent claims and terminal event (lapse).

    Models the dependence between claim frequency and lapse through a
    shared latent frailty term. Handles informative censoring.

    Parameters
    ----------
    theta_init : float
        Starting frailty variance.
    alpha_init : float
        Starting association between frailty and lapse. 0 = independence.
    frailty_dist : {"gamma", "lognormal"}
        Frailty distribution. Gamma allows closed-form E-step for pure
        recurrent models; lognormal needs numerical integration.
    n_quad : int
        Number of quadrature points for numerical integration in E-step.
        15–20 is typically sufficient; increase if ll doesn't stabilise.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence criterion.
    verbose : bool

    Examples
    --------
    >>> model = JointFrailtyModel(alpha_init=0.5)
    >>> model.fit(claim_data, lapse_data, covariates=["age_band"])
    >>> model.association_  # estimated alpha
    """

    def __init__(
        self,
        theta_init: float = 1.0,
        alpha_init: float = 0.0,
        frailty_dist: str = "gamma",
        n_quad: int = 15,
        max_iter: int = 150,
        tol: float = 1e-5,
        verbose: bool = False,
    ) -> None:
        if frailty_dist not in ("gamma", "lognormal"):
            raise ValueError(f"frailty_dist must be 'gamma' or 'lognormal', got {frailty_dist!r}")
        self.theta_init = theta_init
        self.alpha_init = alpha_init
        self.frailty_dist = frailty_dist
        self.n_quad = n_quad
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(
        self,
        recurrent_data: RecurrentEventData,
        lapse_data: Optional[pd.DataFrame] = None,
        recurrent_covariates: Optional[Sequence[str]] = None,
        lapse_covariates: Optional[Sequence[str]] = None,
    ) -> "JointFrailtyModel":
        """
        Fit the joint frailty model.

        Parameters
        ----------
        recurrent_data : RecurrentEventData
            Counting-process recurrent claim data.
        lapse_data : pd.DataFrame, optional
            One row per policy with columns:
              - policy_id: identifier (must match recurrent_data)
              - lapse_time: float, time to lapse or censoring
              - lapsed: int (0/1), whether lapse was observed
            If None, the joint model degenerates to the standard shared
            frailty model (no informative censoring correction).
        recurrent_covariates : list[str], optional
            Covariates for the claim process.
        lapse_covariates : list[str], optional
            Covariates for the lapse process (can differ from claim covariates).

        Returns
        -------
        self
        """
        self.recurrent_covariates_ = list(recurrent_covariates or [])
        self.lapse_covariates_ = list(lapse_covariates or [])

        # Initialise parameters
        theta = self.theta_init
        alpha = self.alpha_init
        rec_coef = np.zeros(len(self.recurrent_covariates_))
        lapse_coef = np.zeros(len(self.lapse_covariates_))

        # Build quadrature nodes and weights
        nodes, weights = self._get_quadrature(theta)

        rec_policy_arrays = recurrent_data.get_policy_arrays(self.recurrent_covariates_)
        lapse_map = {}
        if lapse_data is not None:
            lapse_map = lapse_data.set_index("policy_id").to_dict("index")

        prev_ll = -np.inf

        for iteration in range(1, self.max_iter + 1):
            # Recompute quadrature nodes for current theta
            nodes, weights = self._get_quadrature(theta)

            # E-step: compute per-policy expected log-likelihood weights
            # (numerical integration over frailty)
            posteriors = self._e_step_joint(
                rec_policy_arrays, lapse_map, theta, alpha,
                rec_coef, lapse_coef, nodes, weights,
            )

            # M-step: update all parameters
            theta_new, alpha_new, rec_coef_new, lapse_coef_new = self._m_step_joint(
                rec_policy_arrays, lapse_map, posteriors, theta, alpha,
                rec_coef, lapse_coef,
            )

            theta = np.clip(theta_new, 1e-4, 20.0)
            alpha = np.clip(alpha_new, -5.0, 5.0)
            rec_coef = rec_coef_new
            lapse_coef = lapse_coef_new

            # Approximate log-likelihood
            ll = float(sum(p["ll_contrib"] for p in posteriors.values()))

            rel_change = abs(ll - prev_ll) / (abs(prev_ll) + 1e-10)
            if self.verbose:
                print(
                    f"Iter {iteration:4d}: ll={ll:.4f}, theta={theta:.4f}, "
                    f"alpha={alpha:.4f}, rel_change={rel_change:.2e}"
                )

            if rel_change < self.tol and iteration > 1:
                self.converged_ = True
                break
            prev_ll = ll
        else:
            self.converged_ = False
            warnings.warn(
                f"JointFrailtyModel EM did not converge in {self.max_iter} iterations.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.theta_ = theta
        self.association_ = alpha
        self.rec_coef_ = rec_coef
        self.lapse_coef_ = lapse_coef
        self.log_likelihood_ = ll
        self.n_iter_ = iteration
        self._is_fitted = True
        self._recurrent_data_ref = recurrent_data

        return self

    def _get_quadrature(self, theta: float) -> tuple[FloatArray, FloatArray]:
        """
        Get quadrature nodes and weights appropriate for the frailty distribution.

        For gamma frailty: use Gauss-Laguerre quadrature transformed to the
        Gamma(1/theta, 1/theta) distribution.

        For lognormal frailty: use Gauss-Hermite quadrature on the log scale.
        """
        if self.frailty_dist == "gamma":
            # Standard Gauss-Laguerre: int_0^inf f(x) e^{-x} dx ~ sum w_k f(x_k)
            # We want int_0^inf g(u) Gamma(1/theta, 1/theta) du
            # Transform: u = x * theta, so nodes become u_k = x_k * theta
            x, w = np.polynomial.laguerre.laggauss(self.n_quad)
            a = 1.0 / theta
            # Gamma(a, a) density: (a^a / Gamma(a)) u^(a-1) exp(-a*u)
            # Transform x = a*u, so u = x/a
            u_nodes = x / a
            # Weights absorb the exp(-x) from Laguerre: we need to multiply by
            # the ratio of the actual density to e^{-x}
            # density at u = x/a: (a^a/Gamma(a)) * (x/a)^{a-1} * exp(-x) / a
            # Divided by e^{-x} for Laguerre: (a^a/Gamma(a)) * (x/a)^{a-1} / a
            log_density_ratio = (
                a * np.log(a) - gammaln(a)
                + (a - 1.0) * (np.log(np.maximum(x, 1e-10)) - np.log(a))
                - np.log(a)
            )
            adjusted_weights = w * np.exp(log_density_ratio)
            return u_nodes, adjusted_weights
        else:
            # Gauss-Hermite for lognormal
            x, w = np.polynomial.hermite.hermgauss(self.n_quad)
            # Z ~ N(0,1), u = exp(mu + sigma*Z) with mu=-sigma^2/2
            sigma2 = np.log(1.0 + theta)
            sigma = np.sqrt(sigma2)
            mu = -sigma2 / 2.0
            u_nodes = np.exp(mu + np.sqrt(2) * sigma * x)
            # Standard GH: int f(x) e^{-x^2} dx ~ sum w_k f(x_k)
            # We need 1/sqrt(pi) factor
            adjusted_weights = w / np.sqrt(np.pi)
            return u_nodes, adjusted_weights

    def _e_step_joint(
        self,
        rec_policy_arrays: dict,
        lapse_map: dict,
        theta: float,
        alpha: float,
        rec_coef: FloatArray,
        lapse_coef: FloatArray,
        nodes: FloatArray,
        weights: FloatArray,
    ) -> dict:
        """
        Compute posterior weights for each policy using numerical quadrature.
        """
        posteriors = {}

        for pid, (tstart, tstop, event, exposure, X) in rec_policy_arrays.items():
            # Recurrent process contribution for each quadrature node
            if X is not None and len(self.recurrent_covariates_) > 0:
                lp_rec = float(X[-1] @ rec_coef) if len(X) > 0 else 0.0
            else:
                lp_rec = 0.0

            N_i = float(event.sum())
            Lambda_rec = float(np.sum(exposure))  # simplified: cumulative exposure

            # For each quadrature node u_k:
            # p(data | u_k) = u_k^N_i * exp(-u_k * Lambda_rec * exp(lp_rec))
            log_lik_rec = (
                N_i * np.log(np.maximum(nodes, 1e-10))
                - nodes * Lambda_rec * np.exp(lp_rec)
            )

            # Lapse contribution
            log_lik_lapse = np.zeros(self.n_quad)
            if pid in lapse_map:
                lapse_info = lapse_map[pid]
                lapse_time = float(lapse_info.get("lapse_time", 1.0))
                lapsed = int(lapse_info.get("lapsed", 0))

                if len(self.lapse_covariates_) > 0 and "covariates" in lapse_info:
                    lp_lapse = float(np.array(list(lapse_info["covariates"].values())) @ lapse_coef)
                else:
                    lp_lapse = 0.0

                # Exponential baseline for lapse: h0_lapse = 1 (unit rate)
                Lambda_lapse_base = lapse_time
                # log p(lapse | u) = lapsed * (alpha * log(u) + lp_lapse) - u^alpha * Lambda_lapse * exp(lp_lapse)
                # Approximate u^alpha using exp(alpha * log(u))
                alpha_log_u = alpha * np.log(np.maximum(nodes, 1e-10))
                log_lik_lapse = (
                    lapsed * (alpha_log_u + lp_lapse)
                    - np.exp(alpha_log_u) * Lambda_lapse_base * np.exp(lp_lapse)
                )

            # Total log-likelihood contribution at each node
            log_lik_total = log_lik_rec + log_lik_lapse

            # Normalise for numerical stability
            log_lik_max = log_lik_total.max()
            lik_rel = np.exp(log_lik_total - log_lik_max)

            # Marginal likelihood approximation
            marginal = float(np.sum(weights * lik_rel)) * np.exp(log_lik_max)
            ll_contrib = np.log(max(marginal, 1e-300))

            # Posterior moments E[u | data] and E[log u | data]
            denominator = np.sum(weights * lik_rel) + 1e-300
            E_u = float(np.sum(weights * nodes * lik_rel) / denominator)
            E_log_u = float(np.sum(weights * np.log(np.maximum(nodes, 1e-10)) * lik_rel) / denominator)

            posteriors[pid] = {
                "E_u": E_u,
                "E_log_u": E_log_u,
                "N_i": N_i,
                "Lambda_rec": Lambda_rec,
                "ll_contrib": ll_contrib,
            }

        return posteriors

    def _m_step_joint(
        self,
        rec_policy_arrays: dict,
        lapse_map: dict,
        posteriors: dict,
        theta: float,
        alpha: float,
        rec_coef: FloatArray,
        lapse_coef: FloatArray,
    ) -> tuple[float, float, FloatArray, FloatArray]:
        """Update all parameters given posterior frailty estimates."""
        from scipy.special import digamma

        n = len(posteriors)
        E_u_vals = np.array([p["E_u"] for p in posteriors.values()])
        E_log_u_vals = np.array([p["E_log_u"] for p in posteriors.values()])

        # Theta update: same Q-function as standard model
        def neg_q_theta(log_theta: float) -> float:
            th = np.exp(log_theta)
            a0 = 1.0 / th
            q = (
                n * (a0 * np.log(a0) - gammaln(a0))
                + (a0 - 1.0) * E_log_u_vals.sum()
                - a0 * E_u_vals.sum()
            )
            return -q

        res_th = minimize(neg_q_theta, x0=np.log(theta), method="L-BFGS-B")
        theta_new = float(np.exp(res_th.x[0]))

        # Alpha update: gradient of Q w.r.t. alpha using lapse data
        # For simplicity, do a grid search over alpha in [-3, 3]
        if lapse_map:
            alpha_grid = np.linspace(-3.0, 3.0, 61)
            q_vals = []
            for a in alpha_grid:
                q = 0.0
                for pid, p in posteriors.items():
                    if pid in lapse_map:
                        li = lapse_map[pid]
                        lapsed = int(li.get("lapsed", 0))
                        lapse_time = float(li.get("lapse_time", 1.0))
                        E_u = p["E_u"]
                        E_log_u = p["E_log_u"]
                        q += lapsed * a * E_log_u
                        q -= np.exp(a * E_log_u) * lapse_time
                q_vals.append(q)
            alpha_new = float(alpha_grid[np.argmax(q_vals)])
        else:
            alpha_new = alpha

        # Recurrent coef update (same as standard model)
        if len(self.recurrent_covariates_) == 0:
            rec_coef_new = np.zeros(0)
        else:
            rec_coef_new = rec_coef.copy()  # simplified — full Newton in next version

        lapse_coef_new = lapse_coef.copy()

        return theta_new, alpha_new, rec_coef_new, lapse_coef_new

    def predict_frailty(self, data: RecurrentEventData) -> list[FrailtyPrediction]:
        """
        Predict posterior frailty for each policy.

        For the joint model, frailty predictions use the recurrent process
        data only (without lapse outcomes, which aren't available at prediction time).
        """
        self._check_fitted()

        nodes, weights = self._get_quadrature(self.theta_)
        policy_arrays = data.get_policy_arrays(self.recurrent_covariates_)
        results = []

        for pid, (tstart, tstop, event, exposure, X) in sorted(
            policy_arrays.items(), key=lambda x: str(x[0])
        ):
            if X is not None and len(self.recurrent_covariates_) > 0:
                lp = float(X[-1] @ self.rec_coef_) if len(X) > 0 else 0.0
            else:
                lp = 0.0

            N_i = float(event.sum())
            Lambda_i = float(np.sum(exposure))

            log_lik = (
                N_i * np.log(np.maximum(nodes, 1e-10))
                - nodes * Lambda_i * np.exp(lp)
            )
            log_lik -= log_lik.max()
            lik = np.exp(log_lik)
            denom = np.sum(weights * lik) + 1e-300

            E_u = float(np.sum(weights * nodes * lik) / denom)
            E_u2 = float(np.sum(weights * nodes**2 * lik) / denom)
            Var_u = max(E_u2 - E_u**2, 0.0)

            a0 = 1.0 / self.theta_
            Z = Lambda_i / (Lambda_i + a0)

            results.append(FrailtyPrediction(
                policy_id=pid,
                frailty_mean=E_u,
                frailty_var=Var_u,
                credibility_factor=Z,
                n_events=N_i,
                exposure=float(exposure.sum()),
            ))

        return results

    def summary(self) -> dict:
        """Return fitted model summary."""
        self._check_fitted()
        return {
            "n_policies": self._recurrent_data_ref.n_policies,
            "n_events": self._recurrent_data_ref.n_events,
            "log_likelihood": self.log_likelihood_,
            "theta": self.theta_,
            "association_alpha": self.association_,
            "frailty_dist": self.frailty_dist,
            "converged": self.converged_,
            "n_iter": self.n_iter_,
        }

    def print_summary(self) -> None:
        self._check_fitted()
        s = self.summary()
        print("=" * 55)
        print("JointFrailtyModel Summary")
        print("=" * 55)
        print(f"  Policies:          {s['n_policies']:,}")
        print(f"  Events:            {s['n_events']:,}")
        print(f"  Log-likelihood:    {s['log_likelihood']:.4f}")
        print(f"  Frailty variance (theta): {s['theta']:.4f}")
        print(f"  Association (alpha):       {s['association_alpha']:.4f}")
        print(f"  Frailty dist:      {s['frailty_dist']}")
        print(f"  Converged:         {s['converged']} ({s['n_iter']} iters)")
        print("=" * 55)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
