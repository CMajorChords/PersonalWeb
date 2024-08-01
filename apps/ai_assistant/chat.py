import streamlit as st
from openai import OpenAI


@st.fragment
def chat_with_history(authentication: bool,
                      message_template: str,
                      model: str,
                      ):
    """
    和历史消息聊天

    :param authentication: 是否认证
    :param message_template: prompt模板
    :param model: 选择的模型
    """
    # 设置OpenAI客户端
    client = OpenAI(
        api_key=st.secrets["OPENAI_API_KEY_PAYED"] if authentication else st.secrets["OPENAI_API_KEY_FREE"],
        base_url="https://api.chatanywhere.com.cn",  # https://api.chatanywhere.cn/v1https://api.chatanywhere.com.cn
    )

    # 显示历史消息
    chat_container = st.container(height=400)
    for message in st.session_state.messages_show:
        with chat_container.chat_message(message["role"]):
            st.write(message["content"])

    # 聊天
    if prompt := st.chat_input(
            "对AI说点什么..." if st.session_state["language"] == "中文" else "Say something to AI..."):
        st.session_state.messages_input.append({"role": "user", "content": prompt})
        st.session_state.messages_show.append({"role": "user", "content": prompt})
        chat_container.chat_message("user").markdown(prompt)
        with chat_container.chat_message("assistant"):
            response = client.chat.completions.create(model=model,
                                                      messages=message_template + st.session_state.messages_input,
                                                      stream=True,
                                                      )
            response_message = st.write_stream(response)
        st.session_state.messages_input.append({"role": "assistant", "content": response_message})
        st.session_state.messages_show.append({"role": "assistant", "content": response_message})
