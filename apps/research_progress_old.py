import streamlit as st
import datetime
import os


# 设置st.session_state
if "selected_time" not in st.session_state:
    st.session_state.selected_time = datetime.date(2023, 8, 21)
st.session_state.language = st.session_state.language
st.session_state.selected_time = st.session_state.selected_time


# 设置页面标题
if st.session_state["language"] == "中文":
    st.markdown('''&emsp;&emsp;PUB(无资料地区预报)需要发展基于对水文学在多时空尺度上的作用有深入“理解”的新预报方法。
                事实上，PUB预示着地表水文学中的一个主要变化的典范，
                从当前以“率定”为基础的地表水文学转变为以“理解”为基础的令人激动的新地表水文学。''')
    st.markdown("<p style='text-align: right;'>——PUB 科学计划，2003</p>", unsafe_allow_html=True)
else:
    st.markdown('''&emsp;&emsp;PUB(Predictions in Ungauged Basins) needs to develop new forecasting methods 
                based on a deep understanding of hydrological progress on multiple spatiotemporal scales. 
                In fact, PUB heralds a major change in surface hydrology, 
                from the current surface hydrology based on "calibration" 
                to the exciting new surface hydrology based on "understanding".''')
    st.markdown("<p style='text-align: right;'>——PUB Science and Implementation Plan, 2003</p>", unsafe_allow_html=True)


# 显示不同语言的markdown
# markdown文件夹路径
folder_path = "creative_contents/科研进展/ZN" if st.session_state[
                                                     "language"] == "English" else "creative_contents/科研进展/CN"


@st.cache_resource(show_spinner=False)
def get_files_data(path):
    # 获取所有.md文件
    md_files = (file for file in os.listdir(path) if file.endswith(".md"))
    # 解析文件名中的日期和标题
    dates_titles_files = ((datetime.datetime.strptime(file.split(" ", 1)[0], "%Y-%m-%d").date(),  # 日期
                           file.split(" ", 1)[1].split(".md")[0],  # 标题
                           file,  # 文件名
                           ) for file in md_files)
    # 按日期排序
    sorted_dates_titles_files = sorted(dates_titles_files, key=lambda x: x[0])
    # 获取排序后的日期、文件名、标题
    sorted_dates, sorted_titles, sorted_files, image_exists = [], [], [], []
    for date, title, file in sorted_dates_titles_files:
        sorted_dates.append(date)  # .strftime("%Y-%m-%d")
        sorted_titles.append(title)
        sorted_files.append(file)
        image_exists.append(os.path.exists(os.path.join(path, file.split(" ")[0] + ".png")))  # 检查对应的PNG文件是否存在
    return sorted_dates, sorted_titles, sorted_files, image_exists


dates, titles, files, png_exists = get_files_data(folder_path)
st.select_slider(
    label="选择日期" if st.session_state["language"] == "中文" else "Select date",
    options=dates,
    label_visibility="collapsed",
    value=dates[0],
    key="selected_time",
    on_change=lambda: st.session_state.update(selected_title=titles[dates.index(st.session_state.selected_time)]),
    help="滑动选择对应日期的科研进展" if st.session_state[
                                             "language"] == "中文" else "Slide to select the research progress of the corresponding date"
)
st.selectbox(
    label="科研进展标题" if st.session_state["language"] == "中文" else "Research Progress Title",
    options=titles,
    index=0,
    label_visibility="collapsed",
    on_change=lambda: st.session_state.update(selected_time=dates[titles.index(st.session_state.selected_title)]),
    key="selected_title"
)
# 如果有同名的png文件，先显示png文件
research_idx = dates.index(st.session_state.selected_time)
with st.container(height=800,
                  border=False,
                  ):
    if png_exists[research_idx]:
        col, _ = st.columns([3, 2])
        with col:
            st.image(os.path.join(folder_path, files[research_idx].split(" ")[0] + ".png"), use_column_width=True)
    with open(os.path.join(folder_path, files[research_idx]), encoding="utf-8") as f:
        st.markdown(f.read(), unsafe_allow_html=True)
