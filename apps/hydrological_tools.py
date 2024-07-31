import streamlit as st
from hydrological_tools import (
    patting_a_potato,
    xinanjiang_model,
    AI_assistant,
    time_series_interpolation,
    bp_neural_network_regression
)


# 初始值
if "hydro_tools_index" not in st.session_state:
    st.session_state["hydro_tools_index"] = 2

# 语言和工具映射
language_options = {
    "中文": ["调戏土豆", "新安江模型", "AI助手", "时间序列插值", "bp神经网络回归"],
    "English": ["Patting Potatoes", "XinAnJiang Model", "AI Assistant", "Time Series Interpolation", "BP NN Regression"]
}
tools_function_map = {
    "调戏土豆": patting_a_potato.patting_a_potato_CN,
    "Patting Potatoes": patting_a_potato.patting_a_potato_EN,
    "新安江模型": xinanjiang_model.xinanjiang_model,
    "XinAnJiang Model": xinanjiang_model.xinanjiang_model,
    "AI助手": AI_assistant.AI_assistant,
    "AI Assistant": AI_assistant.AI_assistant,
    "时间序列插值": time_series_interpolation.time_series_interpolate,
    "Time Series Interpolation": time_series_interpolation.time_series_interpolate,
    "bp神经网络回归": bp_neural_network_regression.bp_neural_network_regression,
    "BP NN Regression": bp_neural_network_regression.bp_neural_network_regression,
}


# 侧边栏设置
def setup_sidebar():
    tool_list = language_options[
        st.radio("选择语言" if st.session_state.language == "中文" else "Select language",
                 list(language_options.keys()),
                 key="language",
                 label_visibility="collapsed"
                 )
    ]  # 选择语言对应的工具列表
    st.selectbox("水文工具", tool_list,
                 index=st.session_state["hydro_tools_index"],
                 key="hydrotools",
                 label_visibility="collapsed",
                 on_change=lambda: st.session_state.update(
                     hydro_tools_index=tool_list.index(st.session_state["hydrotools"])))


with st.sidebar:
    setup_sidebar()

# 设置页面标题
if st.session_state["language"] == "中文":
    st.markdown(
        '''&emsp;&emsp;以未来发展与新数据的获取以及新的实验研究密切相关的观测来结束对模拟的评述，
        也许显得奇怪，但以我们的观点来看，这门科学的现状就是如此。'''
    )
    st.markdown("<p style='text-align: right;'>——George Hornberger 和 Beth Boyer，1995</p>", unsafe_allow_html=True)
else:
    st.markdown(
        '''&emsp;&emsp;It may seem strange to end the review of simulations with observations closely related to 
        future development, acquisition of new data, and new experimental research, 
        but from our perspective, this is the current state of this science.'''
    )
    st.markdown("<p style='text-align: right;'>——George Hornberger, Beth Boyer, 1995</p>", unsafe_allow_html=True)
st.divider()

# 工具选择：
tools_function_map[st.session_state["hydrotools"]]()
