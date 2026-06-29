"""
statistical_models.py
=====================
Multivariate logistic regression modelling for the causal mediation
pipeline.

Two families of models are fitted:

  • Path A (assignment) – models each binary mediator
    (PhysActivity, Fruits, Veggies) as a function of Income + covariates.

  • Path B (outcome) – models HighBP as a function of Income, all
    three mediators, and covariates simultaneously.

All models are estimated via Maximum Likelihood with
``statsmodels.formula.api.logit``.  Coefficients are automatically
exponentiated to Odds Ratios (OR) together with 95 % Wald-type
Confidence Intervals.
"""

import numpy as np
import pandas as pd

from logger import logger


# ── covariate set used in every model ─────────────────────────────────
COVARIATES = ["Age", "Sex", "BMI", "Education"]

# ── mediators ---------------------------------------------------------
MEDIATORS = ["PhysActivity", "Fruits", "Veggies"]


# ── internal helpers ──────────────────────────────────────────────────

def _extract_or_ci(model, var: str) -> dict[str, float]:
    """Extract Odds Ratio, 95 % CI, and p-value for a given predictor.

    Parameters
    ----------
    model : LogitResultsWrapper
        Fitted logit model.
    var : str
        Predictor name (must appear in ``model.params``).

    Returns
    -------
    dict
        Keys: ``or_val``, ``ci_lower``, ``ci_upper``, ``p_value``.
    """
    or_val = np.exp(model.params[var])
    ci_lower, ci_upper = np.exp(model.conf_int().loc[var])
    return {
        "or_val": or_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": model.pvalues[var],
    }


# ── public API ────────────────────────────────────────────────────────

def fit_path_A_models(df):
    """Fit three independent logistic regressions – one per mediator.

    Each model uses the formula::

        mediator ~ Income + Age + Sex + BMI + Education

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing all required columns.

    Returns
    -------
    dict
        Mapping ``{mediator_name: fitted_model}``.
    """
    import statsmodels.formula.api as smf

    models = {}
    formula_base = "{} ~ Income + " + " + ".join(COVARIATES)

    for mediator in MEDIATORS:
        formula = formula_base.format(mediator)
        logger.info("Fitting Path A model – %s", formula)
        model = smf.logit(formula=formula, data=df).fit(disp=0)
        logger.info("Path A model converged – %s (LL: %.2f)",
                    mediator, model.llf)
        models[mediator] = model

    return models


def fit_path_B_model(df):
    """Fit the outcome model for HighBP.

    Formula::

        HighBP ~ Income + PhysActivity + Fruits + Veggies + Age + Sex + BMI + Education

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing all required columns.

    Returns
    -------
    LogitResultsWrapper
        Fitted model.
    """
    import statsmodels.formula.api as smf

    mediators_str = " + ".join(MEDIATORS)
    covariates_str = " + ".join(COVARIATES)
    formula = f"HighBP ~ Income + {mediators_str} + {covariates_str}"

    logger.info("Fitting Path B model – %s", formula)
    model = smf.logit(formula=formula, data=df).fit(disp=0)
    logger.info("Path B model converged (Pseudo R²: %.4f)", model.prsquared)

    return model


def report_path_A(models) -> pd.DataFrame:
    """Compile a clean DataFrame of Path A ORs and CIs for Income.

    Parameters
    ----------
    models : dict
        Output of ``fit_path_A_models``.

    Returns
    -------
    pd.DataFrame
        Indexed by mediator name with columns ``OR``, ``CI Lower``,
        ``CI Upper``, ``p-value``.
    """
    rows = []
    for mediator, model in models.items():
        info = _extract_or_ci(model, "Income")
        rows.append({
            "Mediator":       mediator,
            "OR":             info["or_val"],
            "CI Lower":       info["ci_lower"],
            "CI Upper":       info["ci_upper"],
            "p-value":        info["p_value"],
        })
    return pd.DataFrame(rows).set_index("Mediator").round(4)


def report_path_B(model) -> pd.DataFrame:
    """Compile a clean DataFrame of ORs and CIs for every predictor
    in the Path B model (excluding the intercept).

    Parameters
    ----------
    model : LogitResultsWrapper
        Output of ``fit_path_B_model``.

    Returns
    -------
    pd.DataFrame
        Indexed by predictor name with columns ``OR``, ``CI Lower``,
        ``CI Upper``, ``p-value``.
    """
    rows = []
    for var in model.params.index:
        if var == "Intercept":
            continue
        info = _extract_or_ci(model, var)
        rows.append({
            "Predictor":  var,
            "OR":         info["or_val"],
            "CI Lower":   info["ci_lower"],
            "CI Upper":   info["ci_upper"],
            "p-value":    info["p_value"],
        })
    return pd.DataFrame(rows).set_index("Predictor").round(3)
