import streamlit as st


def clear_history_callback(add_number):
    """
    清除历史消息
    """
    st.session_state.messages_input = []
    st.session_state.messages_show = []
    st.session_state.text_uploader_key += add_number


def clear_chat_history():
    """
    清除历史对话记录

    :return: None
    """
    if st.session_state["language"] == "中文":
        st.button(
            "清除历史对话",
            help="历史图片也会被清除哦",
            on_click=clear_history_callback,
            args=(1,),
        )
    else:
        st.button(
            "Clear chat history",
            help="Historical images an texts will also be cleared",
            on_click=clear_history_callback,
            args=(1,),
        )
