"""
app.py
======
Streamlit dashboard for the Causal Mediation Policy Simulation.

This module is strictly a **rendering layer**.  Every mathematical
computation — from coefficient extraction to population-risk
projection — is delegated to backend modules
(``src.mediation_engine``, ``src.simulation``).

Usage
-----
.. code-block:: bash

   streamlit run src/app.py
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from src.simulation import (
    MEDIATORS,
    PATH_B_OR,
    PATH_B_BETA,
    LOW_INCOME_PREV,
    HIGH_INCOME_PREV,
    BASELINE_HYPERTENSION_PREVALENCE,
    TOTAL_POPULATION,
    run_intervention,
    SimulationResult,
)


# ── page config ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Hypertension Policy Simulator",
    page_icon="\U0001F4CA",
    layout="wide",
)


# ── sidebar: background information ───────────────────────────────────

with st.sidebar:
    st.markdown("## \U0001F3E5 Policy Simulation Dashboard")
    st.markdown(
        "Adjust the sliders below to simulate the impact of increasing "
        "behavioural health access for lower-income populations on "
        "population-wide hypertension prevalence."
    )

    st.markdown("### \U0001F52C Causal Engine")
    st.markdown(
        "All projections use the validated **Product-of-Coefficients** "
        "mediation model (MacKinnon, Fairchild & Fritz, 2007) fitted on "
        "N = 253 680 BRFSS 2015 records."
    )

    st.markdown("### \U0001F4D6 Baseline Parameters")
    cols = st.columns(2)
    with cols[0]:
        st.metric("Hypertension Prevalence", f"{BASELINE_HYPERTENSION_PREVALENCE:.0%}")
    with cols[1]:
        st.metric("Population", f"{253_680:,}")

    st.markdown("---")
    st.caption(
        "Built on the CDC BRFSS Heart Disease Health Indicators dataset."
    )


# ── main panel ────────────────────────────────────────────────────────

st.title("Causal Mediation Policy Simulator")
st.markdown(
    "Use the controls below to simulate targeted public-health "
    "interventions and see their projected effect on hypertension "
    "prevalence in real time."
)

# ── intervention sliders ──────────────────────────────────────────────

st.markdown("### \U0001F39B Intervention Controls")
st.markdown(
    "Each slider represents a **relative lift** in behavioural access "
    "for the low-income population (0% = no change, 50% = maximum "
    "realistic intervention)."
)

col1, col2, col3 = st.columns(3)

with col1:
    phys_pct = st.slider(
        "\U0001F3C3 Physical Activity",
        min_value=0, max_value=50, value=0, step=1,
        help="Percentage increase in physical-activity access for low-income groups.",
    )
with col2:
    fruits_pct = st.slider(
        "\U0001F34E Fruit Consumption",
        min_value=0, max_value=50, value=0, step=1,
        help="Percentage increase in fruit-consumption access for low-income groups.",
    )
with col3:
    veggies_pct = st.slider(
        "\U0001F955 Vegetable Consumption",
        min_value=0, max_value=50, value=0, step=1,
        help="Percentage increase in vegetable-consumption access for low-income groups.",
    )


# ── run simulation ────────────────────────────────────────────────────

result: SimulationResult = run_intervention(
    phys_pct=float(phys_pct),
    fruits_pct=float(fruits_pct),
    veggies_pct=float(veggies_pct),
)


# ── outcome metric cards ──────────────────────────────────────────────

st.markdown("### \U0001F4CA Projected Impact")

mcol1, mcol2, mcol3, mcol4 = st.columns(4)

with mcol1:
    st.metric(
        label="Hypertension Prevalence",
        value=f"{result.overall_hypertension_prevalence:.1%}",
        delta=f"{-abs(result.absolute_risk_reduction):.1%}",
        delta_color="normal",
        help="A negative delta confirms a successful reduction in population hypertension prevalence.",
    )

with mcol2:
    st.metric(
        label="Cases Averted",
        value=f"{result.cases_averted:,}",
        delta=f"{result.cases_averted / BASELINE_HYPERTENSION_PREVALENCE / TOTAL_POPULATION * 100:.1f} % of baseline",
        delta_color="normal",
        help="Estimated number of hypertension cases prevented.",
    )

with mcol3:
    st.metric(
        label="Population Protected",
        value=f"{result.absolute_risk_reduction / BASELINE_HYPERTENSION_PREVALENCE:.1%}",
        delta=None,
        help="Proportion of baseline hypertension cases eliminated.",
    )

total_lift = sum(result.intervention_pct.values())
with mcol4:
    st.metric(
        label="Total Intervention Lift",
        value=f"{total_lift:.0f} %",
        delta=None,
        help="Sum of all slider values (capped at 150).",
    )


# ── detailed breakdown chart ──────────────────────────────────────────

st.markdown("### \U0001F4CB Per-Mediator Contribution to Risk Reduction")

contrib_df = pd.DataFrame([
    {
        "Mediator":       m,
        "Odds Ratio":     PATH_B_OR[m],
        "Path B \u03b2": PATH_B_BETA[m],
        "Baseline (Low-Income)": f"{LOW_INCOME_PREV[m]:.0%}",
        "Post-Intervention":     f"{result.new_mediator_prevalence[m]:.0%}",
        "Log-Odds Shift":        result.mediator_contributions[m],
    }
    for m in MEDIATORS
])

st.dataframe(contrib_df, use_container_width=True, hide_index=True)

# ── bar chart of mediator log-odds contributions ──────────────────────

st.markdown("### \U0001F4C8 Contribution to Log-Odds Shift")

chart_data = pd.DataFrame({
    "Mediator": MEDIATORS,
    "Log-Odds Contribution": [
        result.mediator_contributions[m] for m in MEDIATORS
    ],
})

st.bar_chart(chart_data, x="Mediator", y="Log-Odds Contribution")


# ── comparison table: before / after ──────────────────────────────────

st.markdown("### \U0001F4D0 Before vs. After")

comparison = pd.DataFrame({
    "Metric": [
        "Hypertension Prevalence",
        "PhysActivity (Low-Income)",
        "Fruits (Low-Income)",
        "Veggies (Low-Income)",
    ],
    "Before": [
        f"{BASELINE_HYPERTENSION_PREVALENCE:.1%}",
        f"{LOW_INCOME_PREV['PhysActivity']:.1%}",
        f"{LOW_INCOME_PREV['Fruits']:.1%}",
        f"{LOW_INCOME_PREV['Veggies']:.1%}",
    ],
    "After": [
        f"{result.overall_hypertension_prevalence:.1%}",
        f"{result.new_mediator_prevalence['PhysActivity']:.1%}",
        f"{result.new_mediator_prevalence['Fruits']:.1%}",
        f"{result.new_mediator_prevalence['Veggies']:.1%}",
    ],
    "Change": [
        f"{-result.absolute_risk_reduction:+.1%}",
        f"{result.new_mediator_prevalence['PhysActivity'] - LOW_INCOME_PREV['PhysActivity']:+.1%}",
        f"{result.new_mediator_prevalence['Fruits'] - LOW_INCOME_PREV['Fruits']:+.1%}",
        f"{result.new_mediator_prevalence['Veggies'] - LOW_INCOME_PREV['Veggies']:+.1%}",
    ],
})

st.dataframe(comparison, use_container_width=True, hide_index=True)


# ── footer ────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Disclaimer: Projections are based on observational mediation "
    "analysis and should not be interpreted as causal guarantees. "
    "Results depend on the validity of the Product-of-Coefficients "
    "assumptions and the generalisability of the BRFSS 2015 sample."
)
