# Web app/tabs/station_temps_tab.py
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

BASE_URL = "https://www.metoffice.gov.uk/pub/data/weather/uk/climate/stationdata/"
DEFAULT_STATION_SLUG = "armagh"
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

YEAR_ROW_RE = re.compile(r"^\s*(\d{4})\b")
LEADING_NUM_RE = re.compile(r"^(-?\d+(?:\.\d+)?)")
JAN_DEC = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
NUM_1_12 = [str(i) for i in range(1,13)]

def _ua() -> Dict[str, str]:
    return {"User-Agent": "Mozilla/5.0 (compatible; StreamlitApp/1.0)"}

def _slug_to_url(station_slug_or_url: str) -> str:
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

def _is_month_header_line(line: str) -> bool:
    parts = line.strip().split()
    if not parts: return False
    if parts[0].lower() != "year": return False
    cols = [p.upper() for p in parts[1:]]
    if len(cols) < 12: return False
    first12 = cols[:12]
    return (first12 == JAN_DEC) or (first12 == NUM_1_12)

def _scan_wide_tables(lines: List[str]) -> Dict[int, pd.DataFrame]:
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
            if v is None:
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

def render():
    """Render the station temperature tab."""
    st.subheader("Monthly Mean Temperature (Met Office, fetched live)")
    st.caption("Mean = (tmax + tmin) / 2. Supports wide (Year+12 cols) and 'mm' row formats.")

    # Station picker
    preset_labels = [PRESET_STATIONS[s] for s in PRESET_STATIONS]
    station_keys = list(PRESET_STATIONS.keys())
    choice = st.selectbox("Station", preset_labels + ["Custom…"], index=0)
    if choice == "Custom…":
        custom_in = st.text_input(
            "Enter station slug (e.g., oxford) or full .txt URL",
            value=DEFAULT_STATION_SLUG
        )
        station_slug_or_url = custom_in.strip()
        station_label = station_slug_or_url or DEFAULT_STATION_SLUG
    else:
        idx = (preset_labels + ["Custom…"]).index(choice)
        station_slug_or_url = station_keys[idx]
        station_label = choice

    start_year = st.number_input("Start year", min_value=1850, max_value=2100, value=1950, step=1)

    @st.cache_data(show_spinner=False, ttl=60*60*24)
    def get_station_df(station_slug_or_url_cached: str, start_year_cached: int) -> pd.DataFrame:
        url = _slug_to_url(station_slug_or_url_cached)
        text = _download_text(url)
        return _build_tmean_df(text, start_year=start_year_cached)

    try:
        dfm = get_station_df(station_slug_or_url, int(start_year))
    except Exception as ex:
        st.error(f"Could not load data for **{station_label}**.\n\nError: {ex}")
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

    st.dataframe(df_sel[["Year", "Month", "tmax", "tmin", "tmean", "Date"]], use_container_width=True)
