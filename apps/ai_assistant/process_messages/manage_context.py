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
        st.button(label="清除历史对话",
                  help="历史图片也会被清除哦",
                  on_click=clear_history_callback,
                  use_container_width=True,
                  args=(1,),
                  )
    else:
        st.button(label="Clear chat history",
                  help="Historical images an texts will also be cleared",
                  on_click=clear_history_callback,
                  use_container_width=True,
                  args=(1,),
                  )


def clear_last_message_callback():
    """
    清除最后一条消息
    """
    if len(st.session_state.messages_input) > 0:
        st.session_state.messages_input.pop()
        st.session_state.messages_show.pop()


def clear_last_message():
    """
    清除最后一条消息

    :return: None
    """
    if st.session_state["language"] == "中文":
        st.button(label="清除最后一条消息",
                  help="清除最后一条消息",
                  on_click=clear_last_message_callback,
                  use_container_width=True,
                  )
    else:
        st.button(label="Clear last message",
                  help="Clear last message",
                  on_click=clear_last_message_callback,
                  use_container_width=True,
                  )
