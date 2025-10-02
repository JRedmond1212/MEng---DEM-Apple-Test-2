# Web app/tabs/uk_production_tab.py
"""
UK Production – DEFRA AUK Chapter 7, Table 7.11 (Fresh fruit)

One-graph view with a single multiselect to compare any of:
  • 7.11a (production, £m unless specified):
      - orchard_area_kha (thousand hectares)
      - value_dessert_apples_m (£m)
      - value_culinary_apples_m (£m)
  • 7.11b (prices, £/t):
      - price_dessert_apples_pt
      - price_culinary_apples_pt
  • 7.11c (supply & use, thousand tonnes unless noted):
      - total_production_kt, imports_eu_kt, imports_row_kt,
        exports_eu_kt, exports_row_kt, total_new_supply_kt,
        change_in_stocks_kt, total_domestic_uses_kt
      - prod_pct_of_new_supply (percentage)

Rules for the single chart:
  - If 1 unit group selected → raw units on single y-axis.
  - If 2 unit groups selected → twin y-axes (left/right).
  - If ≥3 unit groups selected → switch to indexed view (yr0 = 100) on single y-axis.
    (You can also force normalization via a toggle.)

Requirements:
  odfpy>=1.4.1 (ODS support)
"""

from __future__ import annotations
import re
from typing import Dict, List, Tuple, Optional

import pandas as pd
import requests
import streamlit as st

GOV_PAGE = (
    "https://www.gov.uk/government/statistics/"
    "agriculture-in-the-united-kingdom-2023/chapter-7-crops#fresh-fruit"
)

YEAR_TOKEN = re.compile(r"^\d{4}$")

# ---- Label map: key -> (label_substring, display_name, unit_group) ----
# unit_group ∈ {"kha","m_gbp","price_pt","kt","pct"}
LABELS: Dict[str, Tuple[str, str, str]] = {
    # 7.11a
    "orchard_area_kha":        ("orchard fruit (thousand hectares)", "Orchard area (kha)", "kha"),
    "value_dessert_apples_m":  ("value of production dessert apples", "Value: dessert apples (£m)", "m_gbp"),
    "value_culinary_apples_m": ("value of production culinary apples", "Value: culinary apples (£m)", "m_gbp"),
    # 7.11b
    "price_dessert_apples_pt": ("prices dessert apples", "Price – dessert apples (£/t)", "price_pt"),
    "price_culinary_apples_pt":("prices culinary apples", "Price – culinary apples (£/t)", "price_pt"),
    # 7.11c
    "total_production_kt":     ("total production", "Total production (kt)", "kt"),
    "imports_eu_kt":           ("imports from the eu", "Imports from EU (kt)", "kt"),
    "imports_row_kt":          ("imports from the rest of the world", "Imports from RoW (kt)", "kt"),
    "exports_eu_kt":           ("exports to the eu", "Exports to EU (kt)", "kt"),
    "exports_row_kt":          ("exports to the rest of the world", "Exports to RoW (kt)", "kt"),
    "total_new_supply_kt":     ("total new supply", "Total new supply (kt)", "kt"),
    "change_in_stocks_kt":     ("change in stocks", "Change in stocks (kt)", "kt"),
    "total_domestic_uses_kt":  ("total domestic uses", "Total domestic uses (kt)", "kt"),
    "prod_pct_of_new_supply":  ("production as % of total new supply", "Production as % of new supply (%)", "pct"),
}

# Convenience lookups
DISPLAY = {k: v[1] for k, v in LABELS.items()}
UNIT_GROUP = {k: v[2] for k, v in LABELS.items()}


def _ua() -> Dict[str, str]:
    return {"User-Agent": "Mozilla/5.0 (compatible; StreamlitApp/1.0)"}

def _require_odfpy():
    try:
        import odf  # noqa: F401
    except ImportError:
        st.error(
            "This tab reads the official DEFRA ODS spreadsheet and requires `odfpy`.\n\n"
            "Install locally:\n    pip install odfpy\n\n"
            "On Streamlit Cloud, add to ROOT `requirements.txt` and redeploy:\n    odfpy>=1.4.1"
        )
        st.stop()

def _find_ch7_ods_url() -> str:
    html = requests.get(GOV_PAGE, headers=_ua(), timeout=30).text
    m = re.search(r'https?://[^"\']+chapter7[^"\']+\.ods', html, flags=re.I)
    if not m:
        raise RuntimeError("Could not locate the Chapter 7 ODS link on the GOV.UK page.")
    return m.group(0)

def _read_sheet_as_str(ods_url: str, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(ods_url, engine="odf", sheet_name=sheet_name, header=None, dtype=str)
    return df.applymap(lambda x: str(x).strip() if pd.notna(x) else "")

def _get_table_7_11(ods_url: str) -> Tuple[str, pd.DataFrame]:
    xls = pd.ExcelFile(ods_url, engine="odf")
    if "Table_7_11" in xls.sheet_names:
        return "Table_7_11", _read_sheet_as_str(ods_url, "Table_7_11")
    # Helpful debug if it fails
    raise RuntimeError("Worksheet 'Table_7_11' not found. Sheets: " + ", ".join(xls.sheet_names))

def _find_year_row_and_first_col(sheet: pd.DataFrame) -> Tuple[int, int, List[int]]:
    best_row = -1
    best_start_col = -1
    best_years: List[int] = []
    rows, cols = sheet.shape
    for r in range(min(rows, 250)):
        candidates: List[Tuple[int, int]] = []
        for c in range(cols):
            s = sheet.iat[r, c]
            if YEAR_TOKEN.match(s):
                candidates.append((c, int(s)))
        if len(candidates) < 5:
            continue
        candidates.sort(key=lambda t: t[0])
        cs = [c for c, _ in candidates]
        ys = [y for _, y in candidates]
        run_cols = [cs[0]]; run_years = [ys[0]]
        best_run_cols = run_cols[:]; best_run_years = run_years[:]
        for i in range(1, len(ys)):
            if ys[i] == ys[i-1] + 1 and cs[i] == cs[i-1] + 1:
                run_cols.append(cs[i]); run_years.append(ys[i])
            else:
                if len(run_years) > len(best_run_years):
                    best_run_cols, best_run_years = run_cols[:], run_years[:]
                run_cols, run_years = [cs[i]], [ys[i]]
        if len(run_years) > len(best_run_years):
            best_run_cols, best_run_years = run_cols, run_years
        if len(best_run_years) > len(best_years):
            best_years = best_run_years
            best_row = r
            best_start_col = best_run_cols[0]
    if best_row < 0:
        raise ValueError("Could not find a header row containing a consecutive run of years.")
    return best_row, best_start_col, best_years

def _find_label_row(sheet: pd.DataFrame, label_substring: str, scan_left_cols: int = 8) -> Optional[int]:
    target = label_substring.lower()
    rows, cols = sheet.shape
    for r in range(rows):
        for c in range(min(scan_left_cols, cols)):
            if target in sheet.iat[r, c].lower():
                return r
    return None

def _clean_number(token: str) -> float:
    s = token.replace(",", "").replace("£", "").replace("%", "").strip()
    return pd.to_numeric(s, errors="coerce")

def _read_numeric_row(sheet: pd.DataFrame, row_idx: int, first_year_col: int, years: List[int]) -> pd.Series:
    vals = []
    for i in range(len(years)):
        col = first_year_col + i
        raw = sheet.iat[row_idx, col] if col < sheet.shape[1] else ""
        vals.append(_clean_number(raw))
    return pd.Series(vals, index=years, dtype="float64")

@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def _load_series() -> Tuple[str, str, pd.DataFrame, Dict[str, str], Dict[str, str], List[str]]:
    """
    Returns:
      ods_url, sheet_name, df_all (Year + series),
      display_map, unit_group_map, missing_keys
    """
    _require_odfpy()
    ods_url = _find_ch7_ods_url()
    sheet_name, sheet = _get_table_7_11(ods_url)
    _, first_col, years = _find_year_row_and_first_col(sheet)

    data = {"Year": years}
    missing: List[str] = []

    for key, (label_sub, _disp, _ug) in LABELS.items():
        row = _find_label_row(sheet, label_sub)
        if row is None:
            missing.append(key)
            continue
        ser = _read_numeric_row(sheet, row, first_col, years)
        data[key] = ser.values

    df = pd.DataFrame(data)
    return ods_url, sheet_name, df, DISPLAY, UNIT_GROUP, missing

def render():
    st.subheader("UK Production — DEFRA Table 7.11 (Fresh fruit)")
    st.caption("Single chart with a multiselect. If you pick >2 unit groups, the view switches to an indexed comparison (yr0 = 100).")

    try:
        ods_url, sheet_name, df_all, display, unit_group, missing = _load_series()
        st.caption(f"Source ODS: {ods_url}")
        st.caption(f"Using sheet: **{sheet_name}**")
        if missing:
            st.caption("Not found (labels may have changed): " + ", ".join(display.get(k, k) for k in missing))
    except Exception as ex:
        st.error(f"Could not load DEFRA Table 7.11 from the ODS.\n\nError: {ex}")
        with st.expander("Debug: preview first 25×25 of 'Table_7_11'", expanded=False):
            try:
                _require_odfpy()
                url = _find_ch7_ods_url()
                xls = pd.ExcelFile(url, engine="odf")
                st.write("**All sheet names:**", xls.sheet_names)
                if "Table_7_11" in xls.sheet_names:
                    preview = pd.read_excel(
                        xls, sheet_name="Table_7_11", engine="odf",
                        header=None, dtype=str, nrows=25, usecols=range(25)
                    )
                    st.dataframe(preview, use_container_width=True)
            except Exception as ex2:
                st.write(f"(Preview failed: {ex2})")
        return

    if df_all.empty:
        st.info("No rows found.")
        return

    # Year range
    min_y, max_y = int(df_all["Year"].min()), int(df_all["Year"].max())
    yr0, yr1 = st.slider(
        "Year range",
        min_value=min_y, max_value=max_y,
        value=(max(min_y, 2000), max_y), step=1,
    )
    df = df_all.loc[(df_all["Year"] >= yr0) & (df_all["Year"] <= yr1)].copy()
    if df.empty:
        st.info("No data in the selected year range.")
        return

    # Multiselect of all available series (show only those present)
    available_keys = [k for k in LABELS.keys() if k in df.columns]
    options = [display[k] for k in available_keys]
    default_disp = [display[k] for k in ["total_production_kt", "imports_eu_kt", "exports_eu_kt"] if k in available_keys]
    selected_disp = st.multiselect("Select series to plot (one chart)", options=options, default=default_disp)
    selected_keys = [k for k in available_keys if display[k] in selected_disp]

    # Optional toggle for normalized view
    normalize_toggle = st.toggle("Normalize (index=100 at first year in range)", value=False, help="Useful when comparing different units on one axis.")

    if not selected_keys:
        st.info("Pick at least one series.")
        return

    # Determine unit groups selected
    sel_groups = sorted({unit_group[k] for k in selected_keys})
    num_groups = len(sel_groups)

    # Decide plotting mode
    mode = "raw_single"
    if normalize_toggle:
        mode = "indexed"
    else:
        if num_groups == 1:
            mode = "raw_single"
        elif num_groups == 2:
            mode = "raw_dual"
        else:
            mode = "indexed"

    # Build chart
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        if mode == "raw_single":
            # One unit group only → single y-axis, raw values
            for k in selected_keys:
                ax.plot(df["Year"], df[k], linewidth=1.9, label=display[k])
            # Label by unit group
            ug = sel_groups[0]
            ylabel = {
                "kha": "Thousand hectares",
                "m_gbp": "£ million",
                "price_pt": "£ per tonne",
                "kt": "Thousand tonnes",
                "pct": "Percent (%)",
            }[ug]
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Year")
            ax.set_title("Table 7.11 — selected series")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend()

        elif mode == "raw_dual":
            # Two unit groups → twin y-axes
            g1, g2 = sel_groups[0], sel_groups[1]
            left_keys = [k for k in selected_keys if unit_group[k] == g1]
            right_keys = [k for k in selected_keys if unit_group[k] == g2]
            for k in left_keys:
                ax.plot(df["Year"], df[k], linewidth=1.9, label=display[k])
            ax.set_xlabel("Year")
            ax.grid(True, linestyle="--", alpha=0.35)
            ylab_map = {
                "kha": "Thousand hectares",
                "m_gbp": "£ million",
                "price_pt": "£ per tonne",
                "kt": "Thousand tonnes",
                "pct": "Percent (%)",
            }
            ax.set_ylabel(ylab_map[g1])

            axr = ax.twinx()
            for k in right_keys:
                axr.plot(df["Year"], df[k], linewidth=1.9, label=display[k])
            axr.set_ylabel(ylab_map[g2])

            # Merge legends
            lines_l, labels_l = ax.get_legend_handles_labels()
            lines_r, labels_r = axr.get_legend_handles_labels()
            ax.legend(lines_l + lines_r, labels_l + labels_r, loc="best")
            ax.set_title("Table 7.11 — selected series (dual axes)")

        else:  # indexed
            # Normalize each series to 100 at first year in range
            base_idx = df.index.min()
            for k in selected_keys:
                base = df.loc[base_idx, k]
                series = (df[k] / base * 100.0) if pd.notna(base) and base != 0 else pd.NA
                ax.plot(df["Year"], series, linewidth=1.9, label=display[k])
            ax.set_ylabel("Index (first year = 100)")
            ax.set_xlabel("Year")
            ax.set_title("Table 7.11 — indexed comparison")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend()

        st.pyplot(fig, clear_figure=True)
    except Exception:
        # Fallback to Streamlit native line chart (no dual axes/index labels)
        if mode == "indexed":
            dfi = df[["Year"] + selected_keys].copy()
            base_idx = dfi.index.min()
            for k in selected_keys:
                base = dfi.loc[base_idx, k]
                dfi[k] = (dfi[k] / base * 100.0) if pd.notna(base) and base != 0 else pd.NA
            st.line_chart(dfi.set_index("Year")[selected_keys])
        else:
            st.line_chart(df.set_index("Year")[selected_keys])

    with st.expander("Show data (selected range)"):
        st.dataframe(df[["Year"] + selected_keys].reset_index(drop=True), use_container_width=True)
