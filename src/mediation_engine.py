"""
mediation_engine.py
===================
Algebraic mediation quantification via the Product of Coefficients
Method (MacKinnon, Fairchild & Fritz, 2007).

Given fitted logistic-regression models for Path A (Income → mediator)
and Path B (mediators → HighBP), the module extracts raw log-odds
coefficients and computes:

  1. Indirect Effect (IE)   = β_PathA × β_PathB
  2. Total Effect  (TE)    = β_Direct + IE
  3. Proportion Mediated (%)  = (IE / TE) × 100

All calculations are deterministic and closed-form – no simulation
loops are required, making the module highly efficient for the
N = 253 680 dataset.
"""

import pandas as pd

from logger import logger


# ── mediators in the study ────────────────────────────────────────────
MEDIATORS = ["PhysActivity", "Fruits", "Veggies"]


def compute_mediation(path_A_models,
                      path_B_model,
                      ) -> pd.DataFrame:
    """Run the full Product-of-Coefficients mediation pipeline.

    Parameters
    ----------
    path_A_models : dict[str, LogitResultsWrapper]
        Mapping ``{mediator: fitted_logit_model}`` as returned by
        ``statistical_models.fit_path_A_models``.
    path_B_model : LogitResultsWrapper
        Fitted outcome model for ``HighBP`` as returned by
        ``statistical_models.fit_path_B_model``.

    Returns
    -------
    pd.DataFrame
        Columns:
            ``Mediator``, ``Path_A (β)``, ``Path_B (β)``,
            ``Indirect Effect``, ``Direct Effect``, ``Total Effect``,
            ``Proportion Mediated (%)``.
    """
    logger.info("Computing mediation via Product of Coefficients method")

    # Direct effect of Income on HighBP (c' path) ──────────────────────
    direct = path_B_model.params["Income"]

    rows = []
    for mediator in MEDIATORS:
        # Path A coefficient for Income in the mediator model ──────────
        beta_a = path_A_models[mediator].params["Income"]

        # Path B coefficient for the mediator in the outcome model ─────
        beta_b = path_B_model.params[mediator]

        # Indirect effect ──────────────────────────────────────────────
        ie = beta_a * beta_b

        # Total effect (direct + indirect) ─────────────────────────────
        te = direct + ie

        # Proportion mediated (%) ──────────────────────────────────────
        pm = (ie / te) * 100

        rows.append({
            "Mediator":                mediator,
            "Path_A (β)":              beta_a,
            "Path_B (β)":              beta_b,
            "Indirect Effect":         ie,
            "Direct Effect":           direct,
            "Total Effect":            te,
            "Proportion Mediated (%)": pm,
        })

    result = pd.DataFrame(rows).set_index("Mediator")

    logger.info("Mediation results computed:")
    for _, row in result.iterrows():
        logger.info("  %s mediates %.2f%% of the income–hypertension link",
                    row.name, row["Proportion Mediated (%)"])

    return result.round(4)
