from pathlib import Path

import streamlit as st

app = st.App(Path(__file__).resolve().with_name("streamlit_ui.py"))
