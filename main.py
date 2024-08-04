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


@st.cache_resource
def get_pages(language: str):
    if language == "中文":
        home = st.Page("apps/home.py", title="fun💤", default=True)
        research_progress = st.Page("apps/research_progress/research_progress.py", title="研究进展")
        technical_notes = st.Page("apps/technical_notes.py", title="技术笔记")
        ai_assistant = st.Page("apps/ai_assistant/ai_assistant.py", title="AI助手")
        data_analysis = st.Page("apps/data_analysis.py", title="数据分析")
        hydrological_model = st.Page("apps/hydrological_model.py", title="水文模型")
        pages_dict = {"主页": [home, ],
                      "博客": [research_progress, technical_notes, ],
                      "工具": [ai_assistant, data_analysis, hydrological_model, ],
                      }
    else:
        home = st.Page("apps/home.py", title="fun💤", icon="🏠", default=True)
        research_progress = st.Page("apps/research_progress/research_progress.py", title="Research Progress")
        technical_notes = st.Page("apps/technical_notes.py", title="Technical Notes")
        ai_assistant = st.Page("apps/ai_assistant/ai_assistant.py", title="AI Assistant")
        data_analysis = st.Page("apps/data_analysis.py", title="Data Analysis")
        hydrological_model = st.Page("apps/hydrological_model.py", title="Hydro Model")
        pages_dict = {"Home": [home, ],
                      "Blogs": [research_progress, technical_notes, ],
                      "Tools": [ai_assistant, data_analysis, hydrological_model, ],
                      }
    return (home,
            research_progress,
            technical_notes,
            ai_assistant,
            data_analysis,
            hydrological_model,
            pages_dict)


(page_home,
 page_research_progress,
 page_technical_notes,
 page_ai_assistant,
 page_data_analysis,
 page_hydrological_model,
 pages
 ) = get_pages(st.session_state["language"])

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
