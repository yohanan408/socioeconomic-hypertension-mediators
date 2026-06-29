"""
visualizations.py
=================
Publication-style visualisations for the mediation pipeline.

All calculation logic is strictly separated from layout / styling
concerns.

Functions
---------
- ``plot_forest_plot(model)`` – true medical forest plot with
  asymmetric 95 % CI error bars against a red null line at 1.0.
- ``plot_mediation_summary(percentages)`` – horizontal bar chart
  of the final mediation percentages (20.68 %, 8.44 %, 5.77 %).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from logger import logger


# ── forest plot ───────────────────────────────────────────────────────

def plot_forest_plot(model,
                     figsize: tuple[float, float] = (10, 6),
                     title: str = "Forest Plot: Odds Ratios & 95 % CI"
                     ) -> plt.Figure:
    """Render a medical forest plot of Odds Ratios from a fitted
    logistic model.

    Parameters
    ----------
    model : LogitResultsWrapper
        Fitted logit model (e.g. from ``statistical_models``).
    figsize : tuple[float, float]
        Figure dimensions in inches.
    title : str
        Plot title.

    Returns
    -------
    plt.Figure
        The matplotlib figure object (not shown – call ``plt.show()``
        or ``fig.savefig()`` externally).
    """
    # ---- data extraction (pure calculation) ----
    predictors = [v for v in model.params.index if v != "Intercept"]
    or_vals = np.exp(model.params[predictors])
    ci_raw = np.exp(model.conf_int().loc[predictors])
    lowers = ci_raw.iloc[:, 0]
    uppers = ci_raw.iloc[:, 1]

    err_left  = or_vals - lowers
    err_right = uppers - or_vals
    asym_err  = [err_left, err_right]

    logger.info("Generating forest plot – %d predictors", len(predictors))

    # ---- plotting (layout concerns only) ----
    fig, ax = plt.subplots(figsize=figsize)

    ax.errorbar(x=or_vals, y=predictors, xerr=asym_err,
                fmt="o", color="black", markersize=8,
                capsize=4, elinewidth=1.5,
                label="Odds Ratio (95 % CI)")

    ax.axvline(x=1.0, color="red", linestyle="--", linewidth=1.5,
               label="No Effect (OR = 1.0)")

    for i, (var, val) in enumerate(zip(predictors, or_vals)):
        ax.text(val + 0.015, i, f"{val:.3f}",
                va="center", fontweight="bold", fontsize=10)

    ax.set_title(title, fontsize=13, pad=15)
    ax.set_xlabel("Odds Ratio (OR)")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()

    return fig


# ── mediation summary bar chart ───────────────────────────────────────

def plot_mediation_summary(percentages: dict[str, float] | None = None,
                           figsize: tuple[float, float] = (10, 5),
                           title: str = (
                               "Proportion of Income–Hypertension Link "
                               "Explained by Each Mediator"
                           ),
                           ) -> plt.Figure:
    """Horizontal bar chart of the final Proportion Mediated values.

    Parameters
    ----------
    percentages : dict[str, float] | None
        Mapping ``{mediator_name: percentage}``.  If ``None``, uses
        the project defaults (20.68, 8.44, 5.77).
    figsize : tuple[float, float]
        Figure dimensions.
    title : str
        Plot title.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    if percentages is None:
        percentages = {
            "Fruit Consumption":      5.77,
            "Vegetable Consumption":  8.44,
            "Physical Activity":     20.68,
        }

    mediators = list(percentages.keys())
    vals      = list(percentages.values())

    logger.info("Generating mediation summary bar chart – %d mediators", len(mediators))

    fig, ax = plt.subplots(figsize=figsize)
    sns.set_style("whitegrid")

    bars = ax.barh(mediators, vals, color=sns.color_palette("viridis",
                                                             len(mediators)))

    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", fontweight="bold", fontsize=11)

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Percentage Mediated (%)")
    ax.set_xlim(0, max(vals) * 1.25)
    fig.tight_layout()

    return fig
