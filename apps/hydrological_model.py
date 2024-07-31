import streamlit as st

# 名言
if st.session_state["language"] == "中文":
    saying_path = "data/markdown/saying_hydrological_model_ZN.md"
else:
    saying_path = "data/markdown/saying_hydrological_model_EN.md"

with open(saying_path, "r", encoding="utf-8") as f:
    st.markdown(f.read())
