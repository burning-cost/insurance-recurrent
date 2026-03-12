"""
FrailtyReport: HTML summary report for fitted frailty models.

Generates a self-contained HTML report covering:
- Model configuration and convergence summary
- Frailty distribution summary statistics
- Bühlmann credibility factor table (how much weight the model gives
  to individual policy history vs the population mean)
- Top and bottom decile policy tables (the "most surprising" policyholders)
- Diagnostic plots (embedded as base64 PNG if matplotlib is available,
  omitted gracefully if not)
- Model interpretation notes for actuarial reviewers

The report is designed to be shared directly with pricing teams who need
to understand and sign off on the model before it enters production.
"""

from __future__ import annotations

import base64
import io
import textwrap
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from .data import RecurrentEventData
from .frailty import SharedFrailtyModel
from .diagnostics import (
    frailty_summary_stats,
    event_rate_by_frailty_decile,
)
from ._types import FrailtyPrediction


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Frailty Model Report — {model_name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                margin: 0; padding: 24px; background: #f8f9fa; color: #212529; }}
        h1 {{ color: #1a1a2e; border-bottom: 3px solid #0066cc; padding-bottom: 8px; }}
        h2 {{ color: #0066cc; margin-top: 32px; }}
        h3 {{ color: #495057; }}
        table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
        th {{ background: #0066cc; color: white; padding: 8px 12px; text-align: left; }}
        td {{ padding: 7px 12px; border-bottom: 1px solid #dee2e6; }}
        tr:nth-child(even) {{ background: #f8f9fa; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin: 16px 0;
                 box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
        .metric {{ display: inline-block; background: #e7f3ff; border-radius: 6px;
                   padding: 12px 20px; margin: 6px; text-align: center; }}
        .metric .value {{ font-size: 1.8em; font-weight: bold; color: #0066cc; }}
        .metric .label {{ font-size: 0.85em; color: #6c757d; }}
        .warning {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px; }}
        .note {{ background: #d1ecf1; border-left: 4px solid #17a2b8; padding: 12px;
                 border-radius: 4px; margin: 8px 0; }}
        .highlight-high {{ color: #dc3545; font-weight: bold; }}
        .highlight-low {{ color: #28a745; font-weight: bold; }}
        img {{ max-width: 100%; }}
        footer {{ margin-top: 40px; color: #6c757d; font-size: 0.85em; text-align: center; }}
    </style>
</head>
<body>
    <h1>Frailty Model Report</h1>
    <p><strong>Model:</strong> {model_name} &nbsp;|&nbsp;
       <strong>Generated:</strong> {generated_at} &nbsp;|&nbsp;
       <strong>Library:</strong> insurance-recurrent v{version}</p>

    {convergence_warning}

    <div class="card">
        <h2>Model Overview</h2>
        <div>
            {metrics_html}
        </div>
    </div>

    <div class="card">
        <h2>Frailty Distribution</h2>
        {frailty_stats_html}
        <div class="note">
            <strong>Interpretation:</strong> The frailty variance (theta) measures unobserved
            heterogeneity between policyholders after accounting for rating factors.
            A theta of {theta:.3f} means the standard deviation of unobserved risk is
            {frailty_sd:.2f}x the mean — {theta_interpretation}.
        </div>
    </div>

    <div class="card">
        <h2>Bühlmann Credibility Factors</h2>
        <p>How much weight the model gives to a policy's own claim history vs the
        population average. After <em>N</em> observed claims, the posterior frailty
        blends individual history (weight Z) with prior mean 1.0 (weight 1-Z).</p>
        {credibility_table_html}
    </div>

    <div class="card">
        <h2>Event Rate by Frailty Decile</h2>
        <p>Observed vs model-predicted claim rates across the frailty distribution.
        Good calibration: observed rate and predicted rate should track closely.</p>
        {decile_table_html}
    </div>

    <div class="card">
        <h2>High-Risk Policies (Top 10 by Frailty)</h2>
        {top_policies_html}
    </div>

    <div class="card">
        <h2>Low-Risk Policies (Bottom 10 by Frailty)</h2>
        {bottom_policies_html}
    </div>

    {coefficients_html}

    <div class="card">
        <h2>Actuarial Notes</h2>
        <ul>
            <li>Frailty posteriors are shrunken estimates — they pull extreme values
            towards 1.0 for low-exposure policies. This is by design.</li>
            <li>Policies with no claims will have frailty &lt; 1.0 but not 0.0.
            The amount of shrinkage depends on theta and their exposure.</li>
            <li>Consider using <code>predict_frailty()</code> as an additional
            rating factor or as a trigger for manual referral of high-frailty
            accounts.</li>
            <li>Theta = {theta:.4f}. For context: theta &lt; 0.3 is mild heterogeneity,
            0.3–1.0 is moderate, &gt; 1.0 is substantial.</li>
        </ul>
    </div>

    <footer>
        Generated by insurance-recurrent &mdash; Burning Cost pricing tools
    </footer>
</body>
</html>
"""


class FrailtyReport:
    """
    Generate an HTML report summarising a fitted SharedFrailtyModel.

    Parameters
    ----------
    model : SharedFrailtyModel
        Fitted model.
    data : RecurrentEventData
        Dataset (training data for in-sample diagnostics).
    model_name : str
        Display name for the report header.

    Examples
    --------
    >>> report = FrailtyReport(model, data, model_name="Fleet Insurance Q1 2026")
    >>> report.save("frailty_report.html")
    >>> html = report.render()
    """

    def __init__(
        self,
        model: SharedFrailtyModel,
        data: RecurrentEventData,
        model_name: str = "SharedFrailtyModel",
    ) -> None:
        model._check_fitted()
        self.model = model
        self.data = data
        self.model_name = model_name
        self._predictions: Optional[list[FrailtyPrediction]] = None

    def _get_predictions(self) -> list[FrailtyPrediction]:
        if self._predictions is None:
            self._predictions = self.model.predict_frailty(self.data)
        return self._predictions

    def render(self) -> str:
        """Render the report to an HTML string."""
        from . import __version__

        predictions = self._get_predictions()
        model = self.model
        summary = model.summary()
        theta = model.theta_

        # Convergence warning
        conv_warning = ""
        if not summary["converged"]:
            conv_warning = (
                '<div class="warning"><strong>Warning:</strong> EM algorithm did not '
                f"converge in {summary['n_iter']} iterations. Results may be unreliable. "
                "Try increasing max_iter or loosening tol.</div>"
            )

        # Key metrics
        metrics_html = "".join([
            self._metric(f"{summary['n_policies']:,}", "Policies"),
            self._metric(f"{summary['n_events']:,}", "Claim Events"),
            self._metric(f"{theta:.4f}", "Frailty Variance (theta)"),
            self._metric(f"{summary['log_likelihood']:.2f}", "Log-Likelihood"),
            self._metric(
                "Yes" if summary["converged"] else "No",
                f"Converged ({summary['n_iter']} iters)",
            ),
        ])

        # Frailty stats table
        stats_df = frailty_summary_stats(predictions, theta)
        frailty_stats_html = stats_df.to_html(index=False, classes="", border=0)

        # Theta interpretation
        frailty_sd = np.sqrt(theta)
        if theta < 0.3:
            theta_interp = "mild unobserved heterogeneity"
        elif theta < 1.0:
            theta_interp = "moderate unobserved heterogeneity"
        else:
            theta_interp = "substantial unobserved heterogeneity"

        # Credibility table
        cred_df = model.credibility_factors
        credibility_table_html = cred_df.to_html(index=False, classes="", border=0)

        # Decile table
        try:
            decile_df = event_rate_by_frailty_decile(model, self.data)
            decile_table_html = decile_df.round(4).to_html(index=False, classes="", border=0)
        except Exception:
            decile_table_html = "<p>Decile diagnostics not available.</p>"

        # Top / bottom policies
        pred_df = pd.DataFrame(predictions).sort_values("frailty_mean", ascending=False)
        top10 = pred_df.head(10)[["policy_id", "frailty_mean", "credibility_factor", "n_events", "exposure"]]
        bottom10 = pred_df.tail(10)[["policy_id", "frailty_mean", "credibility_factor", "n_events", "exposure"]]
        top_policies_html = top10.round(4).to_html(index=False, classes="", border=0)
        bottom_policies_html = bottom10.round(4).to_html(index=False, classes="", border=0)

        # Coefficients
        if summary["coef"]:
            coef_df = pd.DataFrame(
                [{"covariate": k, "coefficient": v} for k, v in summary["coef"].items()]
            )
            coefficients_html = (
                '<div class="card"><h2>Regression Coefficients</h2>'
                + coef_df.round(4).to_html(index=False, classes="", border=0)
                + "<p>Coefficients are on the log hazard scale. "
                "A coefficient of 0.5 means that variable multiplies the claim rate by exp(0.5) = 1.65.</p>"
                "</div>"
            )
        else:
            coefficients_html = ""

        return _HTML_TEMPLATE.format(
            model_name=self.model_name,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            version=__version__,
            convergence_warning=conv_warning,
            metrics_html=metrics_html,
            frailty_stats_html=frailty_stats_html,
            theta=theta,
            frailty_sd=frailty_sd,
            theta_interpretation=theta_interp,
            credibility_table_html=credibility_table_html,
            decile_table_html=decile_table_html,
            top_policies_html=top_policies_html,
            bottom_policies_html=bottom_policies_html,
            coefficients_html=coefficients_html,
        )

    def save(self, path: str) -> None:
        """Save the rendered HTML report to a file."""
        html = self.render()
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Report saved to {path}")

    @staticmethod
    def _metric(value: str, label: str) -> str:
        return (
            f'<div class="metric">'
            f'<div class="value">{value}</div>'
            f'<div class="label">{label}</div>'
            f"</div>"
        )
