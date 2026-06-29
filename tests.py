"""
tests.py
========
Lightweight unit tests for the core backend modules of the
hypertension-mediation pipeline.

Run with::

    python -m unittest tests.py -v
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.baseline_screening import _cramers_v, MEDIATORS
from src.data_prep import COLUMNS


# ── data_prep tests ───────────────────────────────────────────────────

class TestDataPrep(unittest.TestCase):
    """Verify that data_prep exports the correct column specification."""

    def test_columns_contains_highbp(self):
        self.assertIn("HighBP", COLUMNS)

    def test_columns_contains_income(self):
        self.assertIn("Income", COLUMNS)

    def test_columns_contains_all_mediators(self):
        for mediator in MEDIATORS:
            with self.subTest(mediator=mediator):
                self.assertIn(mediator, COLUMNS)

    def test_columns_contains_all_covariates(self):
        for cov in ("Age", "Sex", "BMI", "Education"):
            with self.subTest(covariate=cov):
                self.assertIn(cov, COLUMNS)

    def test_column_count_is_nine(self):
        self.assertEqual(len(COLUMNS), 9)


# ── baseline_screening tests ──────────────────────────────────────────

class TestCramersV(unittest.TestCase):
    """Verify the Cramér's V helper produces bounded results."""

    def test_returns_zero_for_perfect_independence(self):
        v = _cramers_v(0.0, 1000, (2, 2))
        self.assertEqual(v, 0.0)

    def test_returns_positive_for_nonzero_chi2(self):
        v = _cramers_v(10442.4, 253_680, (8, 2))
        self.assertGreater(v, 0.0)

    def test_value_is_at_most_one(self):
        v = _cramers_v(10442.4, 253_680, (8, 2))
        self.assertLessEqual(v, 1.0)

    def test_value_is_at_least_zero(self):
        v = _cramers_v(10442.4, 253_680, (8, 2))
        self.assertGreaterEqual(v, 0.0)

    def test_handles_1x1_table_gracefully(self):
        v = _cramers_v(10.0, 100, (1, 1))
        self.assertEqual(v, 0.0)

    def test_produces_expected_physical_activity_v(self):
        # Known value from the validated pipeline
        v = _cramers_v(10442.4, 253_680, (8, 2))
        self.assertAlmostEqual(v, 0.2029, places=4)


if __name__ == "__main__":
    unittest.main()
