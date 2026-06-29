"""
baseline_screening.py
=====================
Unadjusted bivariate association screening for the mediation pipeline.

Computes Chi-Square contingency tables and Cramér's V statistics
for:
  • Path A – Income → each mediator (PhysActivity, Fruits, Veggies)
  • Path B – each mediator → HighBP

Cramér's V is calculated via a single pass of the pre-computed
chi² statistic to avoid redundant computation over the large
(N = 253 680) dataset.

Because of the extreme sample size, almost any non-zero association
will produce p‑values < 0.0001.  Effect sizes (Cramér's V) are
therefore the interpretative focus.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from src.logger import logger


# ── helper ────────────────────────────────────────────────────────────

def _cramers_v(chi2: float, n: int, shape: tuple[int, int]) -> float:
    """Compute Cramér's V from a pre-calculated chi² statistic.

    Parameters
    ----------
    chi2 : float
        Chi-square test statistic.
    n : int
        Total number of observations in the contingency table.
    shape : tuple[int, int]
        (rows, cols) of the contingency table.

    Returns
    -------
    float
        Cramér's V in [0, 1].
    """
    r, k = shape
    denominator = n * (min(r, k) - 1)
    return np.sqrt(chi2 / denominator) if denominator > 0 else 0.0


# ── public API ────────────────────────────────────────────────────────

MEDIATORS = ["PhysActivity", "Fruits", "Veggies"]


def run_bivariate_eda(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute unadjusted Chi-Square and Cramér's V tables for
    Path A (Income → mediators) and Path B (mediators → HighBP).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the columns ``Income``, ``HighBP``, and all
        columns listed in ``MEDIATORS``.

    Returns
    -------
    dict[str, pd.DataFrame]
        ``{"path_A": DataFrame, "path_B": DataFrame}`` with columns
        ``Chi-Square (χ²)``, ``p-value``, and ``Cramér's V``.
    """
    logger.info("Starting bivariate EDA – Path A (Income → mediators) & Path B (mediators → HighBP)")
    results: dict[str, list[dict]] = {"path_A": [], "path_B": []}

    for mediator in MEDIATORS:
        # ── Path A: Income → mediator ────────────────────────
        ct_a = pd.crosstab(df["Income"], df[mediator])
        chi2_a, p_a, _, _ = chi2_contingency(ct_a)
        v_a = _cramers_v(chi2_a, ct_a.sum().sum(), ct_a.shape)

        results["path_A"].append({
            "Mediator Variable": mediator,
            "Chi-Square (χ²)": round(chi2_a, 1),
            "p-value": "< 0.0001" if p_a < 0.0001 else f"{p_a:.4f}",
            "Cramér's V": round(v_a, 4),
        })

        # ── Path B: mediator → HighBP ────────────────────────
        ct_b = pd.crosstab(df["HighBP"], df[mediator])
        chi2_b, p_b, _, _ = chi2_contingency(ct_b)
        v_b = _cramers_v(chi2_b, ct_b.sum().sum(), ct_b.shape)

        results["path_B"].append({
            "Mediator Variable": mediator,
            "Chi-Square (χ²)": round(chi2_b, 1),
            "p-value": "< 0.0001" if p_b < 0.0001 else f"{p_b:.4f}",
            "Cramér's V": round(v_b, 4),
        })

    logger.info("Bivariate EDA complete – Path A: %d mediators, Path B: %d mediators",
                len(results["path_A"]), len(results["path_B"]))

    return {
        "path_A": pd.DataFrame(results["path_A"]).set_index("Mediator Variable"),
        "path_B": pd.DataFrame(results["path_B"]).set_index("Mediator Variable"),
    }
