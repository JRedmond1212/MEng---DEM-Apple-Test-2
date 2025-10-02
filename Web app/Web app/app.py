# Web app/app.py
"""
Streamlit UI shell: wires up tabs and delegates to modules in ./tabs
Run:
  python -m streamlit run "Web app/app.py"
"""

import streamlit as st
from tabs.lifecycle_tab import render as render_lifecycle
from tabs.station_temps_tab import render as render_station_temps
from tabs.uk_production_tab import render as render_uk_production

st.set_page_config(
    page_title="Apple Lifecycle • UK Stations • UK Production",
    layout="wide"
)

st.title("🍎 Apple Tree Lifecycle • 🌡️ UK Station Climate • 📊 UK Production")
st.caption("Tabs are modular — edit code in ./tabs/<tab>_tab.py to change behaviour.")

tab1, tab2, tab3 = st.tabs(["Lifecycle model", "Station temps", "UK Production"])

with tab1:
    render_lifecycle()

with tab2:
    render_station_temps()

with tab3:
    render_uk_production()




