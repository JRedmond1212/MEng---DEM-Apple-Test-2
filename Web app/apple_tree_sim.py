# apple_tree_sim.py
"""
Discrete-event model (SimPy) for an apple tree's annual yield lifecycle.

Phases (default):
- 0–10 years: slow growth
- 20–40 years: exponential growth
- 40–80 years: plateau
- 80–100 years: ~10% reduction then drop to 0 at 100 (tree removed)

All phase boundaries and key magnitudes are parameterized so a UI (Streamlit) can adjust them.
"""

from dataclasses import dataclass
from typing import List, Tuple
import simpy
import math


@dataclass
class LifecycleParams:
    juvenile_years: int = 10           # end of slow growth
    exp_start_year: int = 20           # start exponential growth
    plateau_start_year: int = 40       # start plateau
    decline_start_year: int = 80       # start decline
    end_year: int = 100                # removal (yield = 0 at and after this)
    plateau_yield: float = 100.0       # mature annual yield (e.g., kg/year)
    slow_max_fraction: float = 0.15    # fraction of plateau reached at the end of juvenile
    decline_fraction: float = 0.10     # total reduction by end_year (10%)


def _exp_bridge(y: float, y0: float, y1: float, t: float, t0: float, t1: float) -> float:
    """
    Smooth exponential bridge from y(t0)=y0 to y(t1)=y1 for t in [t0,t1].
    y(t) = y0 * exp(k*(t - t0)), with k chosen so y(t1)=y1.
    Handles edge cases gracefully (fallback to linear if needed).
    """
    if t1 <= t0:
        return y1
    if y0 <= 0:
        # If starting at (or near) zero, use a shifted exponential-like curve
        # to avoid log(0). Start from a tiny positive number.
        y0 = max(y0, 1e-6)
    if y1 <= 0:
        return 0.0
    try:
        k = math.log(y1 / y0) / (t1 - t0)
        return y0 * math.exp(k * (t - t0))
    except (ValueError, ZeroDivisionError):
        # Fallback linear
        alpha = (t - t0) / (t1 - t0)
        return (1 - alpha) * y0 + alpha * y1


def yield_at_year(year: int, p: LifecycleParams) -> float:
    """
    Piecewise yield function with smooth-ish transitions.

    Rules:
    - y < 0: 0
    - 0 <= y < juvenile_years: slow quadratic ramp up to slow_max_fraction * plateau
    - juvenile_years <= y < exp_start_year: hold the slow-phase last value
      (you can treat this as a 'quiet' pre-exponential period)
    - exp_start_year <= y < plateau_start_year: exponential growth to plateau
    - plateau_start_year <= y < decline_start_year: plateau
    - decline_start_year <= y < end_year: linear decline to (1 - decline_fraction) * plateau by (end_year - 1)
    - y >= end_year: 0
    """
    if year < 0:
        return 0.0
    if year >= p.end_year:
        return 0.0

    # Values we will bridge between
    plateau = p.plateau_yield
    slow_target = p.slow_max_fraction * plateau
    decline_floor = (1.0 - p.decline_fraction) * plateau

    # Slow phase: quadratic ramp
    if year < p.juvenile_years:
        # f goes 0 -> 1 across the juvenile period
        f = (year / max(p.juvenile_years, 1)) ** 2
        return f * slow_target

    # Quiet pre-exp phase: hold at the slow phase last value
    if year < p.exp_start_year:
        return slow_target

    # Exponential growth to plateau
    if year < p.plateau_start_year:
        return _exp_bridge(
            y=year,
            y0=slow_target,
            y1=plateau,
            t=year,
            t0=p.exp_start_year,
            t1=p.plateau_start_year,
        )

    # Plateau
    if year < p.decline_start_year:
        return plateau

    # Decline (gentle, ~10% by the final year before removal)
    if year < p.end_year:
        # Map [decline_start_year, end_year-1] -> [plateau, decline_floor]
        span = max((p.end_year - 1) - p.decline_start_year, 1)
        alpha = min(max((year - p.decline_start_year) / span, 0.0), 1.0)
        return (1 - alpha) * plateau + alpha * decline_floor

    # Fallback
    return 0.0


class AppleTree:
    """Represents one apple tree in the SimPy environment."""

    def __init__(self, env: simpy.Environment, params: LifecycleParams):
        self.env = env
        self.params = params
        self.history: List[Tuple[int, float]] = []  # (year, yield)
        self.process = env.process(self.run())

    def run(self):
        year = 0
        # We simulate up to and including end_year to record the drop to 0 at removal
        while year <= self.params.end_year:
            y = yield_at_year(year, self.params)
            self.history.append((year, y))
            # advance one year
            yield self.env.timeout(1)
            year += 1


def simulate(params: LifecycleParams) -> List[Tuple[int, float]]:
    """Run a single-tree simulation and return (year, yield) pairs."""
    env = simpy.Environment()
    tree = AppleTree(env, params)
    env.run(until=params.end_year + 1)
    return tree.history


if __name__ == "__main__":
    # Quick smoke test
    p = LifecycleParams()
    data = simulate(p)
    for yr, val in data[:15]:
        print(f"Year {yr:3d} -> yield {val:6.2f}")
    print("...")
    for yr, val in data[-5:]:
        print(f"Year {yr:3d} -> yield {val:6.2f}")
