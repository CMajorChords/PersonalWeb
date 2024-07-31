"""这个文件专门用于缓存通用的耗时函数"""
import streamlit as st
import pandas as pd
import base64
import zipfile
import io
import os
# import json
# from streamlit_lottie import st_lottie
if "language" not in st.session_state:
    st.session_state["language"] = "中文"

@st.cache_resource
def image(image, caption=None, use_column_width=False, width=None, clamp=False):
    st.image(image, caption=caption, use_column_width=use_column_width, width=width, clamp=clamp)

@st.cache_resource
def audio(audio):
    st.audio(audio)

@st.cache_resource
def markdown(os, unsafe_allow_html=True):
    with open(os, encoding="utf-8") as f:
        st.markdown(f.read(), unsafe_allow_html=unsafe_allow_html)

# @st.cache_resource
# def lottie(animation_source: str,
#            speed: int = 1,
#            reverse: bool = False,
#            loop: bool | int = True,
#            height: int = None,
#            width: int = None
#            ):
#     # 如果输入的是一个url
#     if animation_source.startswith("http"):
#         animation = animation_source
#     # 如果输入的是一个json文件路径
#     else:
#         with open(animation_source) as f:
#             animation = json.load(f)
#     st_lottie(
#         animation_source=animation,
#         speed = speed,
#         reverse=reverse,
#         loop=loop,
#         height=height,
#         width=width
#     )

@st.cache_resource
def zipfiles(directory_path, filename):
    zippeddata = io.BytesIO()
    with zipfile.ZipFile(zippeddata, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, arcname=os.path.relpath(file_path, directory_path))
    return zippeddata

@st.cache_resource
def balloons():
    st.balloons()

@st.cache_resource
def snow():
    st.snow()

@st.cache_resource
def pdf(path: str, width: str = "100%", height: int = 800):
    """
    在streamlit中显示pdf文件

    """
    with open(path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<div style="display:flex; justify-content:center;"><iframe src="data:application/pdf;base64,{base64_pdf}" width={width} height={height} type="application/pdf"></iframe></div>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# @st.cache_data
# def set_page():
#     # 设置页面背景
#     # linear-gradient(180deg, #5fb3d4, #ff90a8)蓝色到粉色
#     # linear-gradient(180deg, #4472c4, #ed7d31)蓝色到橙色
#     # linear-gradient(180deg, #d6e1f4, #eeddd2)浅蓝色到浅橙色
#     # < style >
#     #     [data-testid="stHeader"] {
#     #     background-color: #00000000  !important;
#     #     background-size: cover;
#     #     }
#     # [data-testid="stSidebar"] {
#     #     background: linear-gradient(90deg, #d39ab4, #8da9c7) !important;
#     #     background-size: cover;
#     # }
#     # .stApp > header {
#     #     background-color: transparent;
#     # }
#     # </style>
#     apps = '''
#     <style>
#     #MainMenu {visibility: hidden;} footer {visibility: hidden;}
#     <style>
#     '''
#     st.markdown("""
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             </style>
#             """, unsafe_allow_html=True)

@st.cache_data(show_spinner="将数据转换为Excel文件..." if st.session_state["language"] == "English" else "Converting data to Excel file...")
def convert_dataframe_to_excel(dataframe: pd.DataFrame):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer) as writer:
        dataframe.to_excel(
            writer
        )
    return excel_buffer.getvalue()