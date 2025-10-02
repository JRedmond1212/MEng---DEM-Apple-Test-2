# Web app/tabs/lifecycle_tab.py
import math
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import streamlit as st

# Optional: import external model if present, else fallback to embedded SimPy model
try:
    from apple_tree_sim import LifecycleParams, simulate  # type: ignore
    MODEL_SOURCE = "external (apple_tree_sim.py)"
except Exception as e:
    MODEL_SOURCE = f"embedded fallback (reason: {type(e).__name__}: {e})"
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
        """Exponential interpolation from (t0,y0)→(t1,y1) evaluated at t."""
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
            alpha = (t - t0) / max((t1 - t0), 1e-9)
            return (1 - alpha) * y0 + alpha * y1

    def yield_at_year(year: int, p: LifecycleParams) -> float:
        if year < 0 or year >= p.end_year:
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
            # IMPORTANT: pass all six args (bug-proof with keywords)
            return _exp_bridge(
                y=year, y0=slow_target, y1=plateau, t=year, t0=p.exp_start_year, t1=p.plateau_start_year
            )
        if year < p.decline_start_year:
            return plateau
        span = max((p.end_year - 1) - p.decline_start_year, 1)
        a = min(max((year - p.decline_start_year) / span, 0.0), 1.0)
        return (1 - a) * plateau + a * decline_floor

    class AppleTree:
        def __init__(self, env: "simpy.Environment", params: LifecycleParams):
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


def render():
    """Render the lifecycle tab UI + charts."""
    st.caption(f"Lifecycle model source: {MODEL_SOURCE}")

    with st.sidebar:
        st.header("Lifecycle Parameters")
        plateau_yield = st.number_input("Plateau yield (kg/year)", 1.0, 10000.0, 100.0, 10.0)
        slow_max_fraction = st.slider("Slow phase max fraction of plateau", 0.01, 0.50, 0.15, 0.01)
        decline_fraction = st.slider("Total decline by removal (fraction of plateau)", 0.00, 0.50, 0.10, 0.01)
        st.divider()
        st.caption("Lifecycle boundaries (years)")
        juvenile = st.slider("Juvenile ends", 1, 30, 10, 1)
        exp_start = st.slider("Exponential start", 1, 50, 20, 1)
        plateau_start = st.slider("Plateau start", 10, 70, 40, 1)
        decline_start = st.slider("Decline start", 20, 95, 80, 1)
        end_year = st.slider("End year (removal)", 30, 150, 100, 1)

        if not (0 < juvenile <= exp_start < plateau_start < decline_start < end_year):
            st.warning("Adjusted boundaries to maintain order.")
            seq = sorted([juvenile, exp_start, plateau_start, decline_start, end_year])
            for i in range(1, len(seq)):
                if seq[i] <= seq[i-1]:
                    seq[i] = seq[i-1] + 1
            juvenile, exp_start, plateau_start, decline_start, end_year = seq

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

    # Simulate
    data = simulate(params)
    df = pd.DataFrame(data, columns=["Year", "Yield (kg)"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Peak yield (kg)", f"{df['Yield (kg)'].max():.1f}")
    c2.metric("Total yield (kg)", f"{df['Yield (kg)'].sum():.0f}")
    c3.metric("Productive years", f"{int((df['Yield (kg)'] > 0).sum())}")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(df["Year"], df["Yield (kg)"], linewidth=2)
        ax.set_xlabel("Year"); ax.set_ylabel("Yield (kg)"); ax.set_title("Apple Tree Annual Yield")
        ax.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig, clear_figure=True)
    except Exception:
        st.line_chart(df.set_index("Year")["Yield (kg)"])

    st.dataframe(df, use_container_width=True)
