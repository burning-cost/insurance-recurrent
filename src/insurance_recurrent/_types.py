"""
Type definitions and protocols for insurance-recurrent.

We use TypedDicts for structured data containers to keep the API explicit
without requiring dataclasses. Protocols define what the models must expose
so callers can swap implementations.
"""

from __future__ import annotations

from typing import Optional, Protocol, Sequence
from typing_extensions import TypedDict

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Data container types
# ---------------------------------------------------------------------------


class ClaimRecord(TypedDict):
    """A single claim event record in counting-process format."""

    policy_id: str
    """Unique policyholder/policy identifier."""

    tstart: float
    """Start of the risk interval (gap time or calendar time)."""

    tstop: float
    """End of the risk interval."""

    event: int
    """1 if a claim occurred at tstop, 0 if censored."""

    exposure: float
    """Fraction of the interval that was at risk (0–1 or 0–exposure_max)."""


class FrailtyFitResult(TypedDict):
    """Fitted model summary returned by SharedFrailtyModel.summary()."""

    n_policies: int
    n_events: int
    log_likelihood: float
    theta: float
    """Frailty variance (gamma shape parameter = 1/theta)."""

    coef: dict[str, float]
    se: dict[str, float]
    converged: bool
    n_iter: int


class FrailtyPrediction(TypedDict):
    """Per-policy frailty prediction."""

    policy_id: str
    frailty_mean: float
    """Posterior mean of the frailty (Bühlmann credibility estimate)."""

    frailty_var: float
    """Posterior variance of the frailty."""

    credibility_factor: float
    """Z = n_i / (n_i + 1/theta): how much weight the policy's own history gets."""

    n_events: float
    """Observed event count for this policy."""

    exposure: float
    """Total observed exposure for this policy."""


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class RecurrentModel(Protocol):
    """Minimal protocol that all frailty models must satisfy."""

    def fit(
        self,
        data: "RecurrentEventData",  # noqa: F821
        covariates: Optional[Sequence[str]] = None,
    ) -> "RecurrentModel": ...

    def predict_frailty(self, data: "RecurrentEventData") -> list[FrailtyPrediction]: ...

    @property
    def is_fitted(self) -> bool: ...


# ---------------------------------------------------------------------------
# Internal numpy aliases for readability
# ---------------------------------------------------------------------------

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

__all__ = [
    "ClaimRecord",
    "FrailtyFitResult",
    "FrailtyPrediction",
    "RecurrentModel",
    "FloatArray",
    "IntArray",
]
