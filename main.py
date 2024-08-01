import streamlit as st

# basic settings
st.set_page_config(layout="wide",
                   page_icon=":droplet:",
                   menu_items={'Report a bug': "https://docs.qq.com/doc/DU21SRXFYbXVBeWtF",
                               'About': """This is a website for sharing **AI** and **hydrology** knowledge, 
                               and everything is produced by *方正*""", },
                   )

# logo
st.logo("data/image/logo.png", icon_image="data/image/头像无背景.png")

# language
if "language" not in st.session_state:
    st.session_state["language"] = "中文"

# pages
if st.session_state["language"] == "中文":
    page_home = st.Page("apps/home.py", title="fun💤", default=True)
    page_research_progress = st.Page("apps/research_progress/research_progress.py", title="研究进展")
    page_technical_notes = st.Page("apps/technical_notes.py", title="技术笔记")
    page_ai_assistant = st.Page("apps/ai_assistant/ai_assistant.py", title="AI助手")
    page_data_analysis = st.Page("apps/data_analysis.py", title="数据分析")
    page_hydrological_model = st.Page("apps/hydrological_model.py", title="水文模型")
    pages = {"主页": [page_home, ],
             "博客": [page_research_progress, page_technical_notes, ],
             "工具": [page_ai_assistant, page_data_analysis, page_hydrological_model, ],
             }
else:
    page_home = st.Page("apps/home.py", title="fun💤", icon="🏠", default=True)
    page_research_progress = st.Page("apps/research_progress/research_progress.py", title="Research Progress")
    page_technical_notes = st.Page("apps/technical_notes.py", title="Technical Notes")
    page_ai_assistant = st.Page("apps/ai_assistant/ai_assistant.py", title="AI Assistant")
    page_data_analysis = st.Page("apps/data_analysis.py", title="Data Analysis")
    page_hydrological_model = st.Page("apps/hydrological_model.py", title="Hydro Model")
    pages = {"Home": [page_home, ],
             "Blogs": [page_research_progress, page_technical_notes, ],
             "Tools": [page_ai_assistant, page_data_analysis, page_hydrological_model, ],
             }

# sidebar
if st.session_state["language"] == "中文":
    with st.sidebar.container(border=True):
        st.markdown("**主页**")
        st.page_link(page_home, icon="🏠")
        st.markdown("**博客**")
        st.page_link(page_research_progress, icon="🔬")
        st.page_link(page_technical_notes, icon="🛠️")
        st.markdown("**工具**")
        st.page_link(page_ai_assistant, icon="🤖")
        st.page_link(page_data_analysis, icon="📈")
        st.page_link(page_hydrological_model, icon="⚙️")
else:
    with st.sidebar.container(border=True):
        st.markdown("**Home**")
        st.page_link(page_home, icon="🏠")
        st.markdown("**Blogs**")
        st.page_link(page_research_progress, icon="🔬")
        st.page_link(page_technical_notes, icon="🛠️")
        st.markdown("**Tools**")
        st.page_link(page_ai_assistant, icon="🤖")
        st.page_link(page_data_analysis, icon="📈")
        st.page_link(page_hydrological_model, icon="⚙️")
# 设置语言选择
with st.sidebar.container(border=True):
    st.radio(label="Choose language",
             options=["中文", "English"],
             index=0,
             key="language",
             label_visibility="collapsed",
             horizontal=True,
             )

# navigation
page = st.navigation(pages, position="hidden")
page.run()
