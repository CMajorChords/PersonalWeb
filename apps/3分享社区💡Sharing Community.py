import streamlit as st
import os

# 设置页面基本布局
st.set_page_config(
    page_title="fun💤-分享社区",
    page_icon="image/头像无背景.png",
    layout="wide",
)


# 设置 language变量
if "language" not in st.session_state:
    st.session_state.language = "中文"
st.session_state.language = st.session_state.language


# 侧边栏设置：
with st.sidebar:
    language = st.radio(
        "选择语言" if st.session_state.language == "中文" else "Select language",
        ("中文", "English"),
        key="language",
        label_visibility="collapsed"
    )


# 页面标题
if st.session_state["language"] == "中文":
    st.markdown(
        "&emsp;&emsp;向混乱进军，因为那里才大有可为。"
    )
    st.markdown("<p style='text-align: right;'>——Steven Weinberg，2003</p>", unsafe_allow_html=True)
elif st.session_state["language"] == "English":
    st.markdown(
        "&emsp;&emsp;Go for the messes - that's where the action is."
    )
    st.markdown("<p style='text-align: right;'>——Steven Weinberg，2003</p>", unsafe_allow_html=True)


# 显示markdown
# markdown文件夹路径
md_folder_path = "creative_contents/我的分享"
# markdown文件夹下的文件名(按照创建时间排序)
md_name = sorted(os.listdir(md_folder_path), key=lambda x: os.stat(os.path.join(md_folder_path, x)).st_ctime)
# markdown文件夹下的文件名（不含后缀）
tab_name = list(map(lambda x: x.split(".")[0], md_name))
# 设置一个标签太多的时候可以滑动
if st.session_state["language"] == "中文":
    st.caption("", help="标签太多？将鼠标放在标签上按住shift可以快速滑动标签，左右方向键也可以切换不同标签的内容")
else:
    st.caption("", help="Too many tabs? Put the mouse on the tab, hold down the shift key to slide quickly, "
                        "and the left and right arrow keys can also switch the content of different tabs")
# 显示markdown
for tab, tab_name, md_name in zip(
        st.tabs(tab_name),
        tab_name,
        md_name
):
    with tab:
        with open(os.path.join(md_folder_path, md_name), "r", encoding="utf-8") as f:
            st.markdown(f.read(), unsafe_allow_html=True)



