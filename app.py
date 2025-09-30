# app.py
"""
Streamlit app for apple tree lifecycle yield.
It first tries to import the SimPy model from `apple_tree_sim.py`.
If that import fails (e.g., files in different folders), it falls back to a local definition.
Run:
  streamlit run app.py
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Try importing the model from a sibling file ----------------------------
try:
    from apple_tree_sim import LifecycleParams, simulate  # type: ignore
    MODEL_SOURCE = "external (apple_tree_sim.py)"
except Exception as e:
    MODEL_SOURCE = f"embedded fallback (reason: {type(e).__name__}: {e})"
    # ----------------- Embedded SimPy model fallback --------------------------
    import simpy

    @dataclass
    class LifecycleParams:
        juvenile_years: int = 10
        exp_start_year: int = 20
        plateau_start_year: int = 40
        decline_start_year: int = 80
        end_year: int = 100
        plateau_yield: float = 100.0
        slow_max_fraction: float = 0.15
        decline_fraction: float = 0.10

    def _exp_bridge(y: float, y0: float, y1: float, t: float, t0: float, t1: float) -> float:
        if t1 <= t0:
            return y1
        if y0 <= 0:
            y0 = max(y0, 1e-6)
        if y1 <= 0:
            return 0.0
        try:
            k = math.log(y1 / y0) / (t1 - t0)
            return y0 * math.exp(k * (t - t0))
        except (ValueError, ZeroDivisionError):
            alpha = (t - t0) / (t1 - t0)
            return (1 - alpha) * y0 + alpha * y1

    def yield_at_year(year: int, p: LifecycleParams) -> float:
        if year < 0:
            return 0.0
        if year >= p.end_year:
            return 0.0

        plateau = p.plateau_yield
        slow_target = p.slow_max_fraction * plateau
        decline_floor = (1.0 - p.decline_fraction) * plateau

        if year < p.juvenile_years:
            f = (year / max(p.juvenile_years, 1)) ** 2
            return f * slow_target

        if year < p.exp_start_year:
            return slow_target

        if year < p.plateau_start_year:
            return _exp_bridge(
                y=year, y0=slow_target, y1=plateau,
                t=year, t0=p.exp_start_year, t1=p.plateau_start_year
            )

        if year < p.decline_start_year:
            return plateau

        if year < p.end_year:
            span = max((p.end_year - 1) - p.decline_start_year, 1)
            alpha = min(max((year - p.decline_start_year) / span, 0.0), 1.0)
            return (1 - alpha) * plateau + alpha * decline_floor

        return 0.0

    class AppleTree:
        def __init__(self, env: simpy.Environment, params: LifecycleParams):
            self.env = env
            self.params = params
            self.history: List[Tuple[int, float]] = []
            self.process = env.process(self.run())

        def run(self):
            year = 0
            while year <= self.params.end_year:
                y = yield_at_year(year, self.params)
                self.history.append((year, y))
                yield self.env.timeout(1)
                year += 1

    def simulate(params: LifecycleParams) -> List[Tuple[int, float]]:
        env = simpy.Environment()
        tree = AppleTree(env, params)
        env.run(until=params.end_year + 1)
        return tree.history
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Apple Tree Lifecycle Yield", layout="centered")
st.title("🍎 Apple Tree Lifecycle Yield (SimPy + Streamlit)")
st.caption(f"Model source: {MODEL_SOURCE}")

st.markdown(
    """
Use the sliders to change lifecycle phase boundaries and plateau yield.

**Phases**  
• Slow growth → *Juvenile ends*  
• Exponential growth → *Plateau start*  
• Plateau → *Decline start*  
• Decline → *End year* (drop to 0; tree removed)
"""
)

with st.sidebar:
    st.header("Parameters")

    plateau_yield = st.number_input(
        "Plateau yield (kg/year)", min_value=1.0, max_value=10000.0, value=100.0, step=10.0
    )
    slow_max_fraction = st.slider(
        "Slow phase max fraction of plateau",
        min_value=0.01, max_value=0.50, value=0.15, step=0.01,
    )
    decline_fraction = st.slider(
        "Total decline by removal (fraction of plateau)",
        min_value=0.00, max_value=0.50, value=0.10, step=0.01,
        help="0.10 means ~10% lower than plateau by the final year prior to removal.",
    )

    st.divider()
    st.caption("Lifecycle boundaries (years)")
    juvenile = st.slider("Juvenile ends", 1, 30, 10, 1)
    exp_start = st.slider("Exponential start", 1, 50, 20, 1)
    plateau_start = st.slider("Plateau start", 10, 70, 40, 1)
    decline_start = st.slider("Decline start", 20, 95, 80, 1)
    end_year = st.slider("End year (removal)", 30, 150, 100, 1)

    # Keep valid ordering
    if not (0 < juvenile <= exp_start < plateau_start < decline_start < end_year):
        st.warning(
            "Adjusting boundaries to maintain order: "
            "Juvenile ≤ Exp start < Plateau start < Decline start < End year."
        )
        seq = [juvenile, exp_start, plateau_start, decline_start, end_year]
        seq_sorted = sorted(seq)
        for i in range(1, len(seq_sorted)):
            if seq_sorted[i] <= seq_sorted[i - 1]:
                seq_sorted[i] = seq_sorted[i - 1] + 1
        juvenile, exp_start, plateau_start, decline_start, end_year = seq_sorted

params = LifecycleParams(
    juvenile_years=juvenile,
    exp_start_year=exp_start,
    plateau_start_year=plateau_start,
    decline_start_year=decline_start,
    end_year=end_year,
    plateau_yield=float(plateau_yield),
    slow_max_fraction=float(slow_max_fraction),
    decline_fraction=float(decline_fraction),
)

st.subheader("Simulated annual yields")

data = simulate(params)
df = pd.DataFrame(data, columns=["Year", "Yield (kg)"])

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Peak yield (kg)", f"{df['Yield (kg)'].max():.1f}")
col2.metric("Total yield (kg)", f"{df['Yield (kg)'].sum():.0f}")
prod_years = (df["Yield (kg)"] > 0).sum()
col3.metric("Productive years", f"{int(prod_years)}")

# Plot
fig, ax = plt.subplots()
ax.plot(df["Year"], df["Yield (kg)"], linewidth=2)
ax.set_xlabel("Year")
ax.set_ylabel("Yield (kg)")
ax.set_title("Apple Tree Annual Yield")
ax.grid(True, linestyle="--", alpha=0.4)
st.pyplot(fig, clear_figure=True)

st.dataframe(df, use_container_width=True)

st.info("Tip: Drag sliders (e.g., move **Juvenile ends** from 10 → 5) to see the curve respond.")
