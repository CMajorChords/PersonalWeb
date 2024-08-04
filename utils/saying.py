import streamlit as st

def write_saying(page_name: str):
    """
    在页面中编辑名言

    :param page_name: 页面名称
    :return: None
    """
    if st.session_state["language"] == "中文":
        saying_path = f"data/markdown/saying_{page_name}_ZN.md"
    else:
        saying_path = f"data/markdown/saying_{page_name}_EN.md"
    with open(saying_path, "r", encoding="utf-8") as f:
        # st.sidebar.container(border=True).markdown(f.read())
        st.markdown(f.read())
    # st.divider()
