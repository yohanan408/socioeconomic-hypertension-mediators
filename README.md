# To What Extent Do Physical Activities and Dietary Habits Link Income and Hypertension?

## An Explanatory Observational Mediation Analysis
**Live Application Link:** [[Launch Policy Simulation Dashboard](https://streamlit.app)](https://socioeconomic-hypertension-mediators-kjc46445zwdev9zdwblvfn.streamlit.app/)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Statsmodels](https://img.shields.io/badge/statsmodels-0.14.4-green.svg)](https://www.statsmodels.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.38.0-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <em>An end-to-end observational mediation pipeline examining the relationships between socioeconomic position, lifestyle behaviours, and hypertension across 253,680 BRFSS 2015 records.</em>
</p>

---
##  Portfolio Executive Summary
* **The Core Strategic Problem:** Standard business analytics rely heavily on passive correlations, leading teams to waste capital on generic marketing campaigns that attack superficial symptoms while root-cause confounding barriers remain unaddressed.
* **The Causal Architecture:** Sequential Ignorability is a key assumption underlying causal mediation analysis: conditional on the observed covariates, there should be no unmeasured confounding of the treatment–mediator and mediator–outcome relationships. Rather than relying on unadjusted bivariate associations, we fitted multivariate logistic regression models using statsmodels, adjusting for Age, Sex, BMI, and Education level across the mediation paths. This specification reduces confounding from the measured covariates and provides adjusted coefficients for the empirical mediation decomposition. However, Sequential Ignorability cannot be established from the observed data alone; the resulting estimates should therefore be interpreted conditional on the validity of the underlying mediation assumptions.
* **The Business & Public Health Value:** The interactive simulation estimates a projected 0.6 percentage-point reduction in hypertension prevalence under a specified behavioural-access scenario, corresponding to approximately 1,586 projected cases in the modelled population.

###  1. The Problem

Cardiovascular disease remains the leading cause of mortality worldwide, and its burden falls disproportionately on lower-income populations. This socio-economic "health gap" is well-documented, but the **operational question** for public health agencies is not *whether* the gap exists. It is *how* to intervene.

Given a limited budget measured in millions of dollars, a health ministry must decide:

> *Should we invest in free community fitness infrastructure, subsidize fresh produce supply chains, or expand baseline healthcare coverage?*

Each intervention targets a different causal pathway. Investing in the wrong channel wastes public funds and yields marginal health returns. This study provides the precise, quantified answer by asking:

**To what exact extent is the income–hypertension disparity driven by modifiable lifestyle behaviours (diet and exercise) versus direct, unmeasured systemic economic barriers?**

---

###  2. The Data / Inputs

| Attribute | Detail |
|---|---|
| **Source** | CDC Behavioral Risk Factor Surveillance System (BRFSS) — Heart Disease Health Indicators (2015) |
| **Sample Size** | 253,680 BRFSS 2015 records |
| **Target Variable** | `HighBP` — binary clinical indicator of hypertension diagnosis |
| **Primary Exposure** | `Income` — 8-level ordinal household income scale |
| **Behavioural Mediators** | `PhysActivity` (binary), `Fruits` (binary), `Veggies` (binary) |
| **Confounding Covariates** | `Age` (13-level ordinal), `Sex` (binary), `BMI` (continuous), `Education` (6-level ordinal) |

**Variable Typology.** BMI operates as the sole continuous covariate; the remaining 7 features are binary flags or ordinal ranks, yielding a lean 9-variable model that satisfies the principle of parsimony for structural equation estimation.

#### Duplicate Response Patterns

Approximately 23,899 records share identical values across the selected response variables. These records were retained rather than deduplicated because identical response patterns do not necessarily represent duplicate observations of the same respondent. Removing them would change the observed sample distribution and could alter prevalence estimates and model coefficients.

The project therefore treats these records as observations in the supplied dataset rather than assuming that identical feature vectors represent erroneous duplicates.

---

###  3. My Approach

```
┌──────────────┐     ┌──────────────────┐     ┌────────────────────┐     ┌──────────────────┐
│  Data Prep   │ ──> │  Bivariate EDA   │ ──> │  Multivariate LR   │ ──> │  Mediation       │
│  (9 columns) │     │  (χ², Cramér's V)│     │  (Path A & B)      │     │  (Product Method)│
└──────────────┘     └──────────────────┘     └────────────────────┘     └──────────────────┘
                                                                                  │
                                                                                  v
                                                                         ┌──────────────────┐
                                                                         │  Policy          │
                                                                         │  Simulation      │
                                                                         │  (Streamlit)     │
                                                                         └──────────────────┘
```

#### Stage 1 — Unadjusted Bivariate Screening

Before constructing parametric models, we mapped raw association strengths using:

- **Spearman Rank Correlation Matrix** — confirmed directionality and screened for multicollinearity (max pairwise |r| = 0.45 between Income and Education; all independent variables retain distinct variance).
- **Chi-Square (χ²) Tests of Independence** — established statistically significant baseline links (all p < 0.0001, driven by the massive sample).
- **Cramér's V** — computed via single-pass algebra (`V = √(χ² / (n · min(r,k)-1))`) to avoid redundant computation at scale.

Given the large sample size, the analysis considers effect sizes alongside statistical significance when assessing the strength of associations.
#### Stage 2 — Multivariate Logistic Regression (Why Not ML)

Three independent Path A models estimated the effect of Income on each mediator, adjusting for Age, Sex, BMI, and Education. A single Path B model estimated the simultaneous effects of Income, all three mediators, and all covariates on HighBP.

**Predictive machine learning algorithms (random forests, gradient boosting, neural nets) were intentionally rejected.** This is an *explanatory* causal study, not a prediction task. Black-box models produce uninterpretable feature-importance metrics that conflate correlation with causation and cannot isolate the specific β coefficients required for formal mediation algebra. The logistic models provide adjusted coefficient estimates for the mediation decomposition while controlling for the measured covariates. The resulting interpretation remains conditional on the mediation assumptions, including the assumption that important unmeasured confounding is adequately controlled.

All models were fitted via `statsmodels.formula.api.logit` with Maximum Likelihood Estimation.

#### Stage 3 — Product-of-Coefficients Mediation

Standard software wrappers (e.g., `statsmodels.stats.mediation`) rely on intensive non-parametric simulation loops or numerical integrations that create a severe processing bottleneck at N > 250k. To resolve this while maintaining mathematical rigor, the **Product of Coefficients Method** (MacKinnon, Fairchild & Fritz, 2007) was implemented algebraically.**The simulator applies the fitted Path B coefficients to hypothetical changes in mediator prevalence.**:

| Metric | Formula |
|---|---|
| Indirect Effect (IE) | β<sub>Path A</sub> × β<sub>Path B</sub> |
| Total Effect (TE) | β<sub>Direct</sub> + IE |
| Proportion Mediated (%) | (IE / TE) × 100 |

These projections are generated from observational mediation models and should not be interpreted as causal guarantees or clinical predictions. Results depend on the fitted model specification, mediation assumptions, observed covariate adjustment, and generalisability of the BRFSS 2015 sample.
---

###  4. The Outcome

#### Quantified Transmission Channels

| Mediator | Proportion Mediated (%) | Path B Odds Ratio | Interpretation |
|---|---|---|---|
| **Physical Activity** | **20.68%** — Primary Bridge | **OR = 0.8220** (17.80% lower odds) | Higher income significantly drives exercise access; regular physical activity drops the independent odds of developing hypertension by 17.80% while holding all other demographic covariates constant. |
| **Vegetable Consumption** | **8.44%** — Modest Link | **OR = 0.9350** (6.50% lower odds) | Income moderately improves vegetable intake, but the downstream protective shift yields a modest 6.50% reduction in hypertension odds. |
| **Fruit Consumption** | **5.77%** — Structural Bottleneck | **OR = 0.9000** (10.00% lower odds) | Regular fruit intake is clinically protective, reducing hypertension odds by 10.00%. However, income is an exceptionally weak driver of fruit habits (V = 0.0811), exposing massive non-income structural constraints. |
| **Combined Behaviours** | **34.89%** | — | The three estimated mediator-specific proportions sum to approximately one-third under the project's product-of-coefficients decomposition. This combined figure should be interpreted as a model-based decomposition rather than a directly observed causal contribution. |
| **Adjusted direct income component** | **~65.11%** | OR = 0.907 (Income, fully controlled) | The remaining component of the model-based income association after accounting for the specified behavioural mediators and measured covariates. It should not be interpreted as a direct estimate of a specific unmeasured systemic mechanism. |

#### Illustrative Policy Implications

1. **Evaluate access-oriented physical activity interventions** — The relatively larger model-based mediation estimate for physical activity suggests that access to opportunities for physical activity could be an area for further intervention research.

2. **Investigate food-access mechanisms** — The comparatively smaller estimated income–fruit pathway suggests that income alone may not capture the factors shaping fruit consumption. Future analysis could examine food availability, geographic access, pricing, and other structural factors.

3. **Examine non-behavioural pathways** — The remaining income association after accounting for the specified behavioural mediators motivates further investigation of healthcare access, economic conditions, environmental factors, and other potential pathways not included in the current model.
---

###  5. What I Learned

#### Engineering: Escaping the Simulation Bottleneck

The initial implementation attempted to use the high-level `statsmodels.stats.mediation` wrapper, which internally computes confidence intervals via non-parametric bootstrapping of the joint distribution of Path A and Path B coefficients. At N = 253,680, this created a **severe processing bottleneck**: Each mediation model required minutes of computation time, making iterative model exploration infeasible.

The fix was a full architectural pivot to **parametric algebraic extraction**. By directly reading `model.params` and `model.conf_int()` from the fitted `LogitResultsWrapper` objects and computing the Product-of-Coefficients formulae in raw NumPy, the mediation quantification dropped from minutes to **milliseconds**, while preserving the project's point-estimate calculation

#### UI/UX: Ghost Signs and Delta Polarity

During the Streamlit dashboard development, an initial naive implementation passed `delta=f"{result.absolute_risk_reduction:.1%}"` directly to `st.metric()`. Because a *reduction* in hypertension prevalence is a clinically positive outcome, but the absolute risk reduction is mathematically positive (42.9% → 42.0% produces ARR = +0.9%), the metric card rendered a **green "up-arrow" delta** communicating the exact wrong signal to a public-health policymaker.

The fix required explicitly inverting the sign: `delta=f"{-abs(result.absolute_risk_reduction):.1%}"` combined with `delta_color="normal"` to ensure that clinically beneficial reductions display as **red downward trends** across all active slider permutations. This was a critical UX lesson: **mathematical correctness is not the same as communicative clarity**.

---

###  6. Deployment & Strategic Decision Support

The project is deployed as an **Interactive Streamlit Mediation-Based Scenario Simulator**. Not a real-time prediction API, but a what-if planning tool for public-health stakeholders.

**Architecture.** The dashboard (`src/app.py`) is a pure rendering layer. Every mathematical operation is delegated to `src.simulation.py`, which encodes the estimated Path B coefficients and executes the counterfactual projection logic:

```
Log-Odds Shift = (min(baseline_prop × (1 + slider/100), 1.0) − baseline_prop)
                × low_income_weight × path_b_beta
```

**Sidebar baseline states** anchor to the pooled averages of records where Income < 5.0:

| Behaviour | Low-Income Baseline |
|---|---|
| Physical Activity | 62.89% |
| Fruit Consumption | 57.65% |
| Vegetable Consumption | 72.08% |

**Controls.** Three sliders (0%–50%) allow the user to simulate a targeted percentage increase in behavioural access for the low-income population. The engine instantly recomputes the projected hypertension prevalence, absolute risk reduction, and estimated cases averted.

**To launch:**

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

---

###  Repository Structure

```
.
├── src/
│   ├── app.py                   # Streamlit dashboard (UI only, no math)
│   ├── data_prep.py             # Data ingestion & column selection
│   ├── baseline_screening.py    # Chi-Square & Cramér's V screening
│   ├── statistical_models.py    # Multivariate logistic regression
│   ├── mediation_engine.py      # Product-of-Coefficients algebra
│   ├── simulation.py            # Policy projection engine
│   ├── visualizations.py        # Forest plots & summary charts
│   └── logger.py                # Centralised logging config
├── .gitignore
├── requirements.txt
├── tests.py                     # Unit tests (unittest framework)
├── Does_Income_Affect_Hypertension_.ipynb  # Original research notebook
└── README.md
```

---

<p align="center">
  <em>Built with Python, statsmodels, Streamlit, and the CDC BRFSS Heart Disease Health Indicators dataset.</em>
  <br>
  <em>Not a clinical decision tool. For research and policy planning purposes only.</em>
</p>
