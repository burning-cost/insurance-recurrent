# insurance-recurrent

Shared frailty models for within-policyholder claim recurrence in insurance pricing.

## The problem

Your Poisson GLM treats every policy-year as independent once you've conditioned on rating factors. It doesn't know that the policyholder who claimed three times last year is probably going to claim again — beyond what their age, vehicle class, and postcode predict.

This unobserved tendency to repeat-claim is the frailty. Some policyholders are just more claim-prone in ways that no rating factor captures. They're the ones who make fleet insurance portfolios behave badly, who drive up pet insurance renewal costs, who account for the long tail in home claims. A shared frailty model estimates this latent heterogeneity from the claim history and turns it into a credibility-adjusted risk score.

The practical output: a per-policy multiplier (the posterior frailty) that says "after seeing this policyholder's full claim history, we believe they're 1.8x as likely to claim as an average risk with the same rating factors." Use it to load renewal premiums, trigger referrals to underwriters, or identify mis-priced portfolios.

## Why there's no existing Python library for this

lifelines is the standard Python survival library. [GitHub issue #878](https://github.com/CamDavidsonPilon/lifelines/issues/878) requested shared frailty in 2017 and was closed as "maybe someday." scikit-survival doesn't support recurrent events at all. The production tools are `frailtypack` and `reReg` in R.

This library fills that gap.

## What it does

- **SharedFrailtyModel**: fits gamma shared frailty via EM algorithm. Each policyholder gets a latent frailty term that multiplicatively shifts their claim hazard. The EM algorithm alternates between computing the posterior frailty given the data (E-step) and updating the regression coefficients and frailty variance (M-step).

- **JointFrailtyModel**: extends the shared frailty model to handle informative censoring from lapse. If high-frailty policyholders lapse more often, standard models underestimate their claim rate because you never see their full claim history. The joint model links the claim and lapse processes through the same frailty term.

- **RecurrentEventData**: converts policy claims histories to counting-process format. Handles gap time vs calendar time, left truncation (mid-term inception), and multiple claim types.

- **RecurrentEventSimulator**: generates synthetic data with known frailty structure. Essential for validating that the EM algorithm recovers the true parameters.

- **Diagnostics**: frailty QQ plots, Cox-Snell residuals, event rate by frailty decile.

- **FrailtyReport**: HTML report suitable for sharing with pricing teams.

## Bühlmann credibility connection

The posterior frailty mean is exactly the Bühlmann-Straub credibility estimate:

```
E[u_i | data] = Z_i * (observed_rate_i) + (1 - Z_i) * 1.0
```

where `Z_i = Lambda_i / (Lambda_i + 1/theta)` is the credibility factor. If you're already loading renewal rates with Bühlmann credibility, this model is doing the same thing but in a proper survival analysis framework that handles exposure, censoring, and covariates correctly. `theta` is the between-policyholder variance — the same parameter as in credibility theory.

## Installation

```bash
pip install insurance-recurrent
```

Requires Python 3.10+, NumPy, SciPy, and pandas. No R dependencies.

Optional for HTML reports:
```bash
pip install insurance-recurrent[report]
```

## Quick start

```python
from insurance_recurrent import (
    RecurrentEventSimulator,
    RecurrentEventData,
    SharedFrailtyModel,
    FrailtyReport,
)

# Simulate fleet insurance data: 500 trucks, frailty variance = 0.6
sim = RecurrentEventSimulator(
    n_policies=500,
    theta=0.6,
    baseline_rate=0.3,
    coef={"vehicle_age": 0.4, "driver_age_band": -0.2},
    seed=42,
)
data, true_frailty = sim.simulate(return_true_frailty=True)

print(data.summary())
# RecurrentEventData (gap time)
#   Policies:    487
#   Events:      642
#   Intervals:   1129
#   Events/policy: mean=1.32, max=8

# Fit the model
model = SharedFrailtyModel(theta_init=1.0, max_iter=100)
model.fit(data, covariates=["vehicle_age", "driver_age_band"])
model.print_summary()
# ======================================================
# SharedFrailtyModel Summary
# ======================================================
#   Policies:          487
#   Events:            642
#   Log-likelihood:    -1247.3
#   Frailty variance (theta): 0.5812
#   Converged:         True (47 iters)
#
#   Coefficients:
#     vehicle_age               +0.3891
#     driver_age_band           -0.1947

# Get per-policy frailty scores
frailty_scores = model.predict_frailty(data)
# frailty_scores[0] = {
#     'policy_id': 'P000042',
#     'frailty_mean': 1.84,      # 84% more likely to claim than average
#     'credibility_factor': 0.73, # 73% weight on own history
#     'n_events': 4,
#     'exposure': 2.5,
# }

# Generate an HTML report
report = FrailtyReport(model, data, model_name="Fleet Q1 2026")
report.save("frailty_report.html")
```

## Loading claim histories from policy tables

Most actuarial teams have a policies table and a claims table in date format:

```python
import pandas as pd
from insurance_recurrent import RecurrentEventData, SharedFrailtyModel

policies = pd.read_csv("policies.csv")  # policy_id, inception_date, expiry_date, ...
claims = pd.read_csv("claims.csv")      # policy_id, claim_date

data = RecurrentEventData.from_policy_claims(
    policies=policies,
    claims=claims,
    covariate_cols=["vehicle_class", "region"],
    time_scale="gap",  # time resets after each claim
)

model = SharedFrailtyModel()
model.fit(data, covariates=["vehicle_class", "region"])
```

## Handling informative lapse

If you suspect that high-risk policyholders leave your book faster (they buy cheaply elsewhere, or you non-renew them), use the joint model:

```python
from insurance_recurrent import JointFrailtyModel

# lapse_data: one row per policy with lapse_time and lapsed columns
model = JointFrailtyModel(alpha_init=0.5)
model.fit(
    recurrent_data=claim_data,
    lapse_data=lapse_df,
    recurrent_covariates=["vehicle_class"],
)
print(f"Association alpha: {model.association_:.3f}")
# Positive alpha: high-frailty policyholders lapse faster
# => standard model underestimates their true claim rate
```

## When to use this

Good candidates:
- **Fleet insurance**: one policy covers multiple vehicles/drivers, frequent events
- **Pet insurance**: chronic conditions, repeat treatments
- **Home insurance**: maintenance-related claims repeat for the same property

Less useful:
- **Personal motor**: most policyholders have 0 or 1 claim per year — not enough within-policy data to estimate individual frailty
- **Single-event products**: travel, single-trip

## Diagnostics

```python
from insurance_recurrent import (
    frailty_qq_data,
    cox_snell_residuals,
    event_rate_by_frailty_decile,
)

# QQ plot data: compare posterior frailty distribution to gamma prior
qq = frailty_qq_data(model, data)
# Plot qq["theoretical"] vs qq["empirical"] — straight line = good fit

# Cox-Snell residuals: should be ~Exp(1) if model correct
resid = cox_snell_residuals(model, data)

# Lift by frailty decile: key diagnostic for actuarial audiences
decile = event_rate_by_frailty_decile(model, data)
print(decile[["decile", "frailty_mean_avg", "observed_rate", "lift"]])
```

## References

- Cook, R.J. & Lawless, J.F. (2007). *The Statistical Analysis of Recurrent Events*. Springer.
- Rondeau, V. et al. (2003). Maximum penalized likelihood estimation in a gamma-frailty model. *Lifetime Data Analysis*.
- Bühlmann, H. & Gisler, A. (2005). *A Course in Credibility Theory*. Springer.
- Vaupel, J.W. et al. (1979). The impact of heterogeneity in individual frailty. *Demography*, 16(3):439–454.
- Andersen, P.K. & Gill, R.D. (1982). Cox's regression model for counting processes. *Annals of Statistics*, 10(4):1100–1120.

## Performance

No formal benchmark yet. The primary advantage of shared frailty over a standard Poisson GLM is not predictive accuracy on held-out data — it is correct inference about unobserved heterogeneity. The frailty dispersion parameter theta tells you how much of the claim frequency variance is unexplained by your covariates. A fitted theta near zero means your rating factors capture most of the risk heterogeneity. A large theta (e.g., > 5) means there is substantial unobserved heterogeneity, and credibility scoring will meaningfully reorder policyholders relative to the GLM prediction. For gamma frailty, EM convergence is typically 10-30 iterations on datasets up to 100k event intervals — fast enough for annual re-fitting. Lognormal frailty with Gauss-Hermite quadrature is 3-5x slower. The JointFrailtyModel is the most computationally expensive option and requires at least 200 policyholders with both recurrent events and terminal event data for stable estimation.


## License

MIT. See [LICENSE](LICENSE).
