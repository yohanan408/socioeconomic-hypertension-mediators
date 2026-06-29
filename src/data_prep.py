"""
data_prep.py
============
Data ingestion and column selection module for the BRFSS 2015
hypertension-mediation pipeline.

Loads the raw CDC BRFSS Heart Disease Health Indicators dataset,
prints an initial missing-value audit, and filters down to the
parsimonious 9-variable set required for the causal mediation
analysis workflow.

Methodological notes on intentional data retention
---------------------------------------------------
1. Missing values: The raw dataset contains zero missing entries
   across all 22 columns.  No imputation is necessary.

2. Duplicate rows: Although ~23 899 rows appear as exact duplicates,
   they are deliberately **not** dropped.  The dataset comprises
   low-cardinality binary / ordinal columns with BMI as the sole
   continuous variable.  With N = 253 680 and a limited number of
   unique variable combinations, the *Pigeonhole Principle* guarantees
   that many distinct individuals will give identical survey answers.
   Dropping these rows would systematically erase real respondents,
   break the random-sampling structure, and invalidate the population
   weights that underpin the downstream inference.
"""

import os
import pandas as pd

from src.logger import logger


# ── column selection ──────────────────────────────────────────────────
COLUMNS = [
    "HighBP",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "Income",
    "Age",
    "Sex",
    "BMI",
    "Education",
]


def load_and_filter(path: str = "heart_disease_health_indicators_BRFSS2015.csv"
                    ) -> pd.DataFrame:
    """Load the raw BRFSS CSV and return a DataFrame with only the 9
    parsimonious columns needed for the mediation pipeline.

    Parameters
    ----------
    path : str
        Path (absolute or relative) to the source CSV file.

    Returns
    -------
    pd.DataFrame
        Subsetted DataFrame with shape (N, 9), preserving all rows.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    logger.info("Loading dataset from %s", path)
    df_raw = pd.read_csv(path)
    logger.info("Raw dataset loaded – shape: %s", df_raw.shape)

    # ── Missing-value audit ───────────────────────────────────────────
    null_counts = df_raw.isnull().sum()
    logger.info("Missing-value audit:\n%s", null_counts.to_string())

    total_missing = null_counts.sum()
    if total_missing == 0:
        logger.info("No missing values detected – imputation skipped")
    else:
        logger.warning("Dataset contains %d missing values", total_missing)

    # ── Filter columns ────────────────────────────────────────────────
    df = df_raw[COLUMNS].copy()
    logger.info("Filtered to %d parsimonious columns – shape: %s",
                len(COLUMNS), df.shape)

    logger.info("Data preparation complete – returning DataFrame with %d rows, %d cols",
                df.shape[0], df.shape[1])

    return df
