"""
insurance-recurrent: shared frailty models for within-policyholder claim recurrence.

The problem this library solves
--------------------------------
Standard Poisson GLMs model annual claim frequency as if each policy-year were
independent. This misses a real effect: some policyholders are inherently
higher-risk in ways that rating factors don't capture. They're the ones who
claim repeatedly. A shared frailty model treats this unobserved heterogeneity
as a latent random effect — the "frailty" — and estimates it from the claim
history.

What you get
-----------
- SharedFrailtyModel: fits gamma shared frailty via EM algorithm
- JointFrailtyModel: handles informative lapse (high-risk policyholders leaving)
- RecurrentEventData: converts policy claim histories to counting-process format
- RecurrentEventSimulator: synthetic data with known DGP for testing
- Diagnostics: QQ plots, Cox-Snell residuals, event rate by frailty decile
- FrailtyReport: HTML report for sharing with pricing teams

Bühlmann credibility connection
--------------------------------
The posterior frailty mean E[u_i | data] is exactly the Bühlmann-Straub
credibility estimate. The frailty variance theta is the credibility variance
parameter. So if you're already credibility-loading renewal rates, you can
think of this model as doing that rigorously in a survival analysis framework.

Typical use cases: fleet/pet/home insurance. Less useful for personal motor
where most policyholders have zero or one claim per year (too sparse to estimate
individual frailty reliably).

References
----------
Cook & Lawless (2007): The Statistical Analysis of Recurrent Events, Springer.
Rondeau et al. (2003): Maximum penalized likelihood estimation in a gamma-frailty
    model. Lifetime Data Analysis.
Bühlmann & Gisler (2005): A Course in Credibility Theory. Springer.
"""

__version__ = "0.1.0"
__author__ = "Burning Cost"

from .data import RecurrentEventData
from .frailty import SharedFrailtyModel
from .joint import JointFrailtyModel
from .simulator import RecurrentEventSimulator
from .diagnostics import (
    frailty_qq_data,
    cox_snell_residuals,
    event_rate_by_frailty_decile,
    frailty_summary_stats,
)
from .report import FrailtyReport

__all__ = [
    "RecurrentEventData",
    "SharedFrailtyModel",
    "JointFrailtyModel",
    "RecurrentEventSimulator",
    "frailty_qq_data",
    "cox_snell_residuals",
    "event_rate_by_frailty_decile",
    "frailty_summary_stats",
    "FrailtyReport",
]
