import streamlit as st
from typing import List, Dict
from openai import OpenAI


def process_input_messages(model: str,
                           message_template: List[Dict[str, str]],
                           authentication: bool,
                           ):
    """处理输入消息"""
    client = OpenAI(
        api_key=st.secrets["OPENAI_API_KEY_PAYED"] if authentication else st.secrets["OPENAI_API_KEY_FREE"],
        base_url="https://api.chatanywhere.com.cn",  # https://api.chatanywhere.cn/v1https://api.chatanywhere.com.cn
    )
    if prompt := st.chat_input(
            "对AI说点什么..." if st.session_state["language"] == "中文" else "Say something to AI..."):
        st.session_state.messages_input.append({"role": "user", "content": prompt})
        st.session_state.messages_show.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)
        with st.chat_message("assistant"):
            response = client.chat.completions.create(model=model,
                                                      messages=message_template + st.session_state.messages_input,
                                                      stream=True,
                                                      )
            response_message = st.write_stream(response)
        st.session_state.messages_input.append({"role": "assistant", "content": response_message})
        st.session_state.messages_show.append({"role": "assistant", "content": response_message})
