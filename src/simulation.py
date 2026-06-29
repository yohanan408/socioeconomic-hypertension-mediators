"""
simulation.py
=============
Causal policy simulation engine for the hypertension-mediation
dashboard.

Encapsulates all mathematical logic needed to translate hypothetical
intervention lifts in behavioural access into projected reductions
in hypertension prevalence.

The engine uses the validated Path B log-odds coefficients from the
Product-of-Coefficients mediation pipeline together with baseline
behavioural prevalence estimates to compute:

  1. The population-wide shift in each mediator after the intervention.
  2. The corresponding change in the log-odds of hypertension.
  3. The new absolute hypertension prevalence and number of cases
     averted.

All hard-coded coefficients were pre-computed by
``src.statistical_models`` on the N = 253 680 BRFSS 2015 sample
and are stored as module-level constants for traceability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple
from src.logger import logger


# ── Validated coefficients (from fitted logistic models) ──────────────
# Path B log-odds coefficients (β) and Odds Ratios from the outcome
# model: HighBP ~ Income + PhysActivity + Fruits + Veggies + Age + Sex
#               + BMI + Education
MEDIATORS = ("PhysActivity", "Fruits", "Veggies")

PATH_B_BETA: Dict[str, float] = {
    "PhysActivity": -0.1966,   # ln(0.822)
    "Fruits":       -0.1057,   # ln(0.900)
    "Veggies":      -0.0675,   # ln(0.935)
}

PATH_B_OR: Dict[str, float] = {
    "PhysActivity": 0.822,
    "Fruits":       0.900,
    "Veggies":      0.935,
}

# Baseline behavioural prevalence by income tier (from notebook EDA)
LOW_INCOME_PREV: Dict[str, float] = {
    "PhysActivity": 0.629,
    "Fruits":       0.577,
    "Veggies":      0.721,
}

HIGH_INCOME_PREV: Dict[str, float] = {
    "PhysActivity": 0.794,
    "Fruits":       0.651,
    "Veggies":      0.838,
}

# Proportion of population defined as "low-income" (Income < 5)
LOW_INCOME_WEIGHT = 0.23

# Baseline hypertension prevalence in the full sample
BASELINE_HYPERTENSION_PREVALENCE = 0.429
TOTAL_POPULATION = 253_680


# ── result container ──────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """Structured output of a single policy simulation run.

    Attributes
    ----------
    intervention_pct : Dict[str, float]
        The slider values that were fed in (one per mediator).
    new_mediator_prevalence : Dict[str, float]
        Post-intervention prevalence of each mediator among
        low-income individuals.
    overall_hypertension_prevalence : float
        Projected hypertension prevalence across the full population.
    cases_averted : int
        Estimated number of hypertension cases prevented.
    absolute_risk_reduction : float
        Baseline minus projected prevalence (percentage points).
    mediator_contributions : Dict[str, float]
        Each mediator's contribution to the total log-odds shift.
    """
    intervention_pct: Dict[str, float]
    new_mediator_prevalence: Dict[str, float]
    overall_hypertension_prevalence: float
    cases_averted: int
    absolute_risk_reduction: float
    mediator_contributions: Dict[str, float]


# ── simulation engine ─────────────────────────────────────────────────

def run_intervention(
    phys_pct: float = 0.0,
    fruits_pct: float = 0.0,
    veggies_pct: float = 0.0,
) -> SimulationResult:
    """Project the population-level impact of increasing behavioural
    access for low-income groups.

    Parameters
    ----------
    phys_pct : float
        Percentage lift in physical-activity access (0–50).
    fruits_pct : float
        Percentage lift in fruit-consumption access (0–50).
    veggies_pct : float
        Percentage lift in vegetable-consumption access (0–50).

    Returns
    -------
    SimulationResult
        Full simulation outcome including projected prevalence,
        cases averted, and per-mediator contributions.
    """
    intervention = {
        "PhysActivity": phys_pct,
        "Fruits": fruits_pct,
        "Veggies": veggies_pct,
    }

    logger.info("Running policy simulation: %s", intervention)

    new_prevalence: Dict[str, float] = {}
    total_log_odds_shift = 0.0
    contributions: Dict[str, float] = {}

    for mediator in MEDIATORS:
        pct = intervention[mediator] / 100.0
        base = LOW_INCOME_PREV[mediator]
        # New mediator prevalence among low-income after the lift ──
        new_prev = base * (1.0 + pct)
        new_prev = min(new_prev, 1.0)  # sanity clamp
        new_prevalence[mediator] = new_prev

        # Population-wide shift in mediator prevalence ─────────────
        delta_mediator = (new_prev - base) * LOW_INCOME_WEIGHT

        # Contribution to log-odds of hypertension ─────────────────
        contrib = PATH_B_BETA[mediator] * delta_mediator
        contributions[mediator] = contrib
        total_log_odds_shift += contrib

    # Convert baseline prevalence → baseline odds → new odds → new prev
    base_odds = BASELINE_HYPERTENSION_PREVALENCE / (
        1.0 - BASELINE_HYPERTENSION_PREVALENCE
    )
    new_odds = base_odds * math.exp(total_log_odds_shift)
    new_prev_htn = new_odds / (1.0 + new_odds)

    arr = BASELINE_HYPERTENSION_PREVALENCE - new_prev_htn
    cases_averted = round(arr * TOTAL_POPULATION)

    logger.info(
        "Simulation result: HTN prevalence %.1f%% → %.1f%%, "
        "%d cases averted",
        BASELINE_HYPERTENSION_PREVALENCE * 100,
        new_prev_htn * 100,
        cases_averted,
    )

    return SimulationResult(
        intervention_pct=intervention,
        new_mediator_prevalence=new_prevalence,
        overall_hypertension_prevalence=new_prev_htn,
        cases_averted=cases_averted,
        absolute_risk_reduction=arr,
        mediator_contributions=contributions,
    )
