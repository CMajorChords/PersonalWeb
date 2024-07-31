"""这个文件专门用于放button点击时的状态"""
import streamlit as st
from modules import cache as stcache


def potato(number=0):
    st.session_state['potato'] += number


def change_language():
    if st.session_state['language'] == "中文":
        st.session_state['language'] = "English"
    elif st.session_state['language'] == "English":
        st.session_state['language'] = "中文"
