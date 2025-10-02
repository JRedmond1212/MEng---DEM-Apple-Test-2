# app.py
"""
Streamlit app: Apple tree lifecycle (SimPy) + UK station monthly mean temperatures.

Run locally:
  python -m streamlit run "Web app/app.py"

On Streamlit Cloud:
  - Main file: Web app/app.py
  - requirements.txt at repo root:
      streamlit>=1.36
      simpy>=4.1
      pandas>=2.2
      matplotlib>=3.9
      requests>=2.31
  - (optional) runtime.txt: 3.12
"""

import math
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import pandas as pd
import requests

# ---------------- SimPy model (import or fallback) ----------------
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
            return _exp_bridge(year, slow_target, plateau, p.exp_start_year, p.plateau_start_year)
        if year < p.decline_start_year:
            return plateau
        span = max((p.end_year - 1) - p.decline_start_year, 1)
        a = min(max((year - p.decline_start_year) / span, 0.0), 1.0)
        return (1 - a) * plateau + a * decline_floor

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


# ---------------- Station data: robust download & parse ----------------
BASE_URL = "https://www.metoffice.gov.uk/pub/data/weather/uk/climate/stationdata/"
DEFAULT_STATION_SLUG = "armagh"

# Dropdown presets (slug -> label)
PRESET_STATIONS = {
    "armagh": "Armagh (NI)",
    "oxford": "Oxford",
    "heathrow": "Heathrow (London)",
    "rothamsted": "Rothamsted",
    "bradford": "Bradford",
    "durham": "Durham",
    "stornoway": "Stornoway",
    "camborne": "Camborne",
    "eastbourne": "Eastbourne",
}

# Regex & helpers
YEAR_ROW_RE = re.compile(r"^\s*(\d{4})\b")
LEADING_NUM_RE = re.compile(r"^(-?\d+(?:\.\d+)?)")

# Header hints for context-based labelling
TMAX_HINTS = ["tmax", "maximum temperature", "mean maximum", "monthly mean maximum", "mean max"]
TMIN_HINTS = ["tmin", "minimum temperature", "mean minimum", "monthly mean minimum", "mean min"]

JAN_DEC = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
NUM_1_12 = [str(i) for i in range(1,13)]

def _ua() -> Dict[str, str]:
    return {"User-Agent": "Mozilla/5.0 (compatible; StreamlitApp/1.0)"}

def _slug_to_url(station_slug_or_url: str) -> str:
    """Accept either a slug (e.g., 'oxford') or a full .txt URL; return a usable URL."""
    s = station_slug_or_url.strip()
    if s.lower().startswith("http"):
        return s
    slug = s.lower()
    if not slug.endswith("data.txt"):
        slug = f"{slug}data.txt"
    return BASE_URL + slug

def _download_text(url: str, timeout: int = 30) -> str:
    r = requests.get(url, timeout=timeout, headers=_ua())
    r.raise_for_status()
    return r.text

def _token_to_float(tok: str) -> Optional[float]:
    """
    Convert a monthly token to float:
      - '---' or '' -> NaN (keeps month position)
      - '12.3*' / '12.3E' -> 12.3
      - 'ANN' / 'Provisional...' / non-numeric -> None (stop row)
    """
    t = tok.strip()
    if not t or t == "---":
        return float("nan")
    tl = t.lower()
    if tl == "ann" or tl.startswith("provisional"):
        return None
    m = LEADING_NUM_RE.match(t)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None

# ---------- Wide-table detection (Year with 12 month columns) ----------
def _is_month_header_line(line: str) -> bool:
    parts = line.strip().split()
    if not parts: return False
    if parts[0].lower() != "year": return False
    cols = [p.upper() for p in parts[1:]]
    if len(cols) < 12: return False
    first12 = cols[:12]
    return (first12 == JAN_DEC) or (first12 == NUM_1_12)

def _scan_wide_tables(lines: List[str]) -> Dict[int, pd.DataFrame]:
    """Parse 'wide' monthly tables (Year + 12 month columns)."""
    tables: Dict[int, pd.DataFrame] = {}
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if _is_month_header_line(line):
            rows: List[Tuple[int, int, float]] = []
            j = i + 1
            while j < n:
                L = lines[j].rstrip("\n")
                if not L.strip(): break
                if _is_month_header_line(L): break
                if YEAR_ROW_RE.match(L):
                    parts = L.split()
                    try:
                        year = int(parts[0])
                    except ValueError:
                        j += 1; continue
                    vals: List[float] = []
                    for tok in parts[1:]:
                        v = _token_to_float(tok)
                        if v is None: break
                        vals.append(v)
                        if len(vals) == 12: break
                    if len(vals) == 12:
                        for m_idx, v in enumerate(vals, start=1):
                            rows.append((year, m_idx, v))
                j += 1
            df = pd.DataFrame(rows, columns=["Year","Month","value"])
            tables[i] = df
            i = j; continue
        i += 1
    return tables

# ---------- Long-table (row/month) detection: headers with 'mm' ----------
def _is_mm_header(line: str) -> bool:
    parts = re.split(r"\s+", line.strip().lower())
    return ("mm" in parts) and ("year" in parts or "yyyy" in parts or "yr" in parts)

def _parse_mm_table(lines: List[str], hdr_idx: int) -> pd.DataFrame:
    header = lines[hdr_idx].strip()
    header_tokens = re.split(r"\s+", header)
    cols = [c.strip().lower() for c in header_tokens]
    col_index = {name: idx for idx, name in enumerate(cols)}

    def _find_col_like(names: List[str]) -> Optional[int]:
        for nm in names:
            if nm in col_index:
                return col_index[nm]
        return None

    year_col = _find_col_like(["year","yyyy","yr"])
    mm_col   = _find_col_like(["mm","month"])
    if year_col is None or mm_col is None:
        return pd.DataFrame(columns=["Year","Month","value","col"])

    # Named tmax/tmin if present
    tmax_col = _find_col_like(["tmax","tx","tmax(degc)","tmax(c)","maxtemp","tmax_c"])
    tmin_col = _find_col_like(["tmin","tn","tmin(degc)","tmin(c)","mintemp","tmin_c"])

    rows_all: List[List[Optional[float]]] = []
    j = hdr_idx + 1
    while j < len(lines):
        L = lines[j].rstrip("\n")
        if not L.strip(): break
        if _is_mm_header(L) or _is_month_header_line(L): break
        parts = re.split(r"\s+", L.strip())
        if len(parts) <= max(year_col, mm_col):
            j += 1; continue
        try:
            year = int(parts[year_col]); mm = int(parts[mm_col])
        except Exception:
            j += 1; continue
        if not (1 <= mm <= 12):
            j += 1; continue

        row_float: List[Optional[float]] = [None]*len(parts)
        for idx, tok in enumerate(parts):
            if idx in (year_col, mm_col): 
                continue
            v = _token_to_float(tok)
            if v is None:  # stop at ANN/provisional marker
                break
            row_float[idx] = v
        rows_all.append([year, mm, row_float, parts])
        j += 1

    if not rows_all:
        return pd.DataFrame(columns=["Year","Month","value","col"])

    def _extract_series(col_idx: int, name: str) -> pd.DataFrame:
        out = []
        for year, mm, row_float, _ in rows_all:
            v = row_float[col_idx] if col_idx < len(row_float) else None
            if v is None: 
                continue
            out.append((int(year), int(mm), float(v), name))
        return pd.DataFrame(out, columns=["Year","Month","value","col"])

    series_frames = []
    if tmax_col is not None:
        series_frames.append(_extract_series(tmax_col, "tmax"))
    if tmin_col is not None:
        series_frames.append(_extract_series(tmin_col, "tmin"))

    # If missing, infer two best numeric columns (most complete; higher mean → tmax)
    if len(series_frames) < 2:
        col_scores: Dict[int, Tuple[int, float]] = {}
        for _, _, row_float, _ in rows_all:
            for idx, val in enumerate(row_float):
                if val is None: 
                    continue
                cnt, s = col_scores.get(idx, (0, 0.0))
                col_scores[idx] = (cnt+1, s+float(val))
        if col_scores:
            cand = sorted(col_scores.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))
            if len(cand) >= 2:
                idx1, (c1, s1) = cand[0]; idx2, (c2, s2) = cand[1]
                mean1, mean2 = s1/max(c1,1), s2/max(c2,1)
                tmax_idx, tmin_idx = (idx1, idx2) if mean1 >= mean2 else (idx2, idx1)
                series_frames = [
                    _extract_series(tmax_idx, "tmax"),
                    _extract_series(tmin_idx, "tmin"),
                ]

    if not series_frames:
        return pd.DataFrame(columns=["Year","Month","value","col"])

    return pd.concat(series_frames, ignore_index=True)

def _scan_mm_tables(lines: List[str]) -> Dict[int, pd.DataFrame]:
    tables: Dict[int, pd.DataFrame] = {}
    for i, line in enumerate(lines):
        if _is_mm_header(line):
            df = _parse_mm_table(lines, i)
            if not df.empty:
                tables[i] = df
    return tables

def _label_from_wide(lines: List[str]) -> Dict[str, pd.DataFrame]:
    wide = _scan_wide_tables(lines)
    if not wide: return {}
    cands = []
    for _, df in wide.items():
        if df.empty: continue
        cands.append((df["value"].mean(skipna=True), df))
    if len(cands) < 2: 
        return {}
    cands.sort(key=lambda x: x[0])
    return {"tmin": cands[0][1], "tmax": cands[-1][1]}

def _label_from_mm(lines: List[str]) -> Dict[str, pd.DataFrame]:
    mm_tables = _scan_mm_tables(lines)
    if not mm_tables:
        return {}
    combo = pd.concat(mm_tables.values(), ignore_index=True)
    if "col" not in combo.columns:
        return {}
    out: Dict[str, pd.DataFrame] = {}
    for var in ("tmax","tmin"):
        dfv = combo.loc[combo["col"] == var, ["Year","Month","value"]].copy()
        if not dfv.empty:
            out[var] = dfv
    return out

def _build_tmean_df(text: str, start_year: int = 1950) -> pd.DataFrame:
    lines = text.splitlines()

    labeled = _label_from_mm(lines)
    if "tmax" not in labeled or "tmin" not in labeled:
        fallback = _label_from_wide(lines)
        for k, v in fallback.items():
            labeled.setdefault(k, v)

    if "tmax" not in labeled or "tmin" not in labeled:
        raise ValueError("Could not obtain both tmax and tmin from file (mm/wide formats).")

    df_tmax = labeled["tmax"].rename(columns={"value": "tmax"})
    df_tmin = labeled["tmin"].rename(columns={"value": "tmin"})

    for df in (df_tmax, df_tmin):
        if "Month" not in df.columns and "mm" in df.columns:
            df.rename(columns={"mm":"Month"}, inplace=True)

    df = pd.merge(df_tmax, df_tmin, on=["Year","Month"], how="inner")
    df = df.loc[df["Year"] >= int(start_year)].copy()
    if df.empty:
        raise ValueError(f"No monthly records at or after start_year={start_year}.")

    df["tmean"] = (df["tmax"] + df["tmin"]) / 2.0
    df["Date"] = pd.to_datetime(df[["Year","Month"]].assign(DAY=1)) + pd.offsets.MonthEnd(0)
    return df.sort_values(["Year","Month"]).reset_index(drop=True)

# ---------------- Streamlit UI ----------------
def main():
    import streamlit as st
    try:
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except ModuleNotFoundError:
        plt = None
        HAS_MPL = False

    st.set_page_config(page_title="Apple Tree Lifecycle & UK Stations", layout="wide")
    st.title("🍎 Apple Tree Lifecycle & 🌡️ UK Station Climate")
    st.caption(f"Model source: {MODEL_SOURCE}")

    tab_model, tab_station = st.tabs(["Lifecycle model", "Station temps"])

    # ---------- Tab 1: lifecycle model ----------
    with tab_model:
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
        data = simulate(params)
        df = pd.DataFrame(data, columns=["Year", "Yield (kg)"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Peak yield (kg)", f"{df['Yield (kg)'].max():.1f}")
        c2.metric("Total yield (kg)", f"{df['Yield (kg)'].sum():.0f}")
        c3.metric("Productive years", f"{int((df['Yield (kg)'] > 0).sum())}")

        if HAS_MPL:
            fig, ax = plt.subplots()
            ax.plot(df["Year"], df["Yield (kg)"], linewidth=2)
            ax.set_xlabel("Year"); ax.set_ylabel("Yield (kg)"); ax.set_title("Apple Tree Annual Yield")
            ax.grid(True, linestyle="--", alpha=0.4)
            st.pyplot(fig, clear_figure=True)
        else:
            st.line_chart(df.set_index("Year")["Yield (kg)"])

        st.dataframe(df, use_container_width=True)

    # ---------- Tab 2: Station temps ----------
    with tab_station:
        st.subheader("Monthly Mean Temperature (Met Office, fetched live)")
        st.caption(
            "Mean computed as (tmax + tmin) / 2. Supports wide (JAN..DEC / 1..12) and 'mm' row formats. "
            "You can pick a preset station or use 'Custom' to enter a slug or full .txt URL."
        )

        # Station picker
        preset_labels = [PRESET_STATIONS[s] for s in PRESET_STATIONS]
        station_keys = list(PRESET_STATIONS.keys())
        choice = st.selectbox("Station", preset_labels + ["Custom…"], index=0)
        if choice == "Custom…":
            default_custom = DEFAULT_STATION_SLUG
            custom_in = st.text_input(
                "Enter station slug (e.g., oxford) or full .txt URL",
                value=default_custom, help="Example slug: oxford  |  Example URL: https://.../oxforddata.txt"
            )
            station_slug_or_url = custom_in.strip()
            station_label = station_slug_or_url or DEFAULT_STATION_SLUG
        else:
            idx = (preset_labels + ["Custom…"]).index(choice)
            station_slug_or_url = station_keys[idx]
            station_label = choice

        # Start year filter
        start_year = st.number_input("Start year", min_value=1850, max_value=2100, value=1950, step=1)

        # Cache per (station, start_year)
        @st.cache_data(show_spinner=False, ttl=60*60*24)
        def get_station_df(station_slug_or_url_cached: str, start_year_cached: int) -> pd.DataFrame:
            url = _slug_to_url(station_slug_or_url_cached)
            text = _download_text(url)
            return _build_tmean_df(text, start_year=start_year_cached)

        # Load & plot
        try:
            dfm = get_station_df(station_slug_or_url, int(start_year))
        except Exception as ex:
            st.error(
                f"Could not load data for **{station_label}**.\n\nError: {ex}"
            )
            if st.toggle("Show first 160 lines of raw file (debug)"):
                try:
                    url = _slug_to_url(station_slug_or_url)
                    raw = _download_text(url)
                    st.code("\n".join(raw.splitlines()[:160]))
                except Exception as ex2:
                    st.write(f"(Re-download failed: {ex2})")
            return

        min_year, max_year = int(dfm["Year"].min()), int(dfm["Year"].max())
        yr0, yr1 = st.slider(
            "Year range",
            min_value=min_year, max_value=max_year,
            value=(max(min_year, max_year - 30), max_year), step=1
        )
        df_sel = dfm.query("@yr0 <= Year <= @yr1").copy()

        try:
            import matplotlib.pyplot as plt
            fig2, ax2 = plt.subplots()
            ax2.plot(df_sel["Date"], df_sel["tmean"], linewidth=1.7)
            ax2.set_xlabel("Date"); ax2.set_ylabel("Mean temperature (°C)")
            ax2.set_title(f"{station_label}: monthly mean temperature — {yr0} to {yr1}")
            ax2.grid(True, linestyle="--", alpha=0.35)
            st.pyplot(fig2, clear_figure=True)
        except Exception:
            st.line_chart(df_sel.set_index("Date")["tmean"])

        with st.expander("Monthly climatology over selected years"):
            clim = (
                df_sel.assign(mon=lambda d: d["Date"].dt.month)
                      .groupby("mon")["tmean"].mean()
                      .reindex(range(1, 13))
            )
            clim.index = pd.Index(
                ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], name="Month"
            )
            st.bar_chart(clim)

        st.dataframe(
            df_sel[["Year", "Month", "tmax", "tmin", "tmean", "Date"]],
            use_container_width=True
        )

# --------- Only run UI when Streamlit provides a ScriptRunContext ---------
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    _HAS_CTX = get_script_run_ctx() is not None
except Exception:
    _HAS_CTX = False

if _HAS_CTX:
    import streamlit as st  # noqa
    main()
else:
    if __name__ == "__main__":
        print(
            "This is a Streamlit app. Launch it with:\n"
            '  python -m streamlit run "Web app/app.py"\n'
            "If you're on Streamlit Cloud, this message is normal during build steps."
        )
