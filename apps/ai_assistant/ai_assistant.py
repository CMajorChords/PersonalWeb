import streamlit as st
from openai import OpenAI
from apps.ai_assistant.compute_token import compute_token_price
from apps.ai_assistant.set_ai import choose_model, get_template_message
from apps.ai_assistant.process_file_message import process_document_message, clear_chat_history, process_image_message
from utils import write_saying, check_password

if "messages_input" not in st.session_state:
    st.session_state.messages_input = []
if "messages_show" not in st.session_state:
    st.session_state.messages_show = []
if "text_upload_flag" not in st.session_state:
    st.session_state.text_upload_flag = False
if "image_upload_flag" not in st.session_state:
    st.session_state.image_upload_flag = False
if "text_uploader_key" not in st.session_state:
    st.session_state["text_uploader_key"] = 1000
if "image_uploader_key" not in st.session_state:
    st.session_state["image_uploader_key"] = -1000

# 创建OpenAI客户端
if check_password(text_label_zn="输入密码解锁更多功能", text_label_en="Enter password to unlock more features"):
    authentication = True
else:
    authentication = False
client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY_PAYED"] if authentication else st.secrets["OPENAI_API_KEY_FREE"],
    # api_key="sk-o09TBraPT6eTfvLVNewzXpGDxCyHfhBTrlpTz54lB4IBqKM8",
    base_url="https://api.chatanywhere.com.cn",  # https://api.chatanywhere.cn/v1https://api.chatanywhere.com.cn
)
# UI基础
write_saying("ai_assistant")
st.subheader("fun💤 AI", anchor=False)
col1, col2 = st.columns((7, 3))
chat_container = st.container(height=400)

# ai设置
with col2.container(border=True):
    # 选择模型
    model_chosen = choose_model(authenticated=authentication)
    # 提示词模板
    message_template = get_template_message()

# 聊天设置
# 设置文档上传
with col1.container(border=True):
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        process_document_message(authenticated=authentication)
        clear_chat_history()
    with sub_col2:
        process_image_message(authenticated=authentication)
        compute_token_price(messages=st.session_state.messages_input, model=model_chosen)

# 显示历史消息
for message in st.session_state.messages_show:
    with chat_container.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("对AI说点什么..." if st.session_state["language"] == "中文" else "Say something to AI..."):
    st.session_state.messages_input.append({"role": "user", "content": prompt})
    st.session_state.messages_show.append({"role": "user", "content": prompt})
    with chat_container.chat_message("user"):
        st.markdown(prompt)
    with chat_container.chat_message("assistant"):
        response = client.chat.completions.create(model=model_chosen,
                                                  messages=message_template + st.session_state.messages_input,
                                                  stream=True,
                                                  )
        response_message = st.write_stream(response)
    st.session_state.messages_input.append({"role": "assistant", "content": response_message})
    st.session_state.messages_show.append({"role": "assistant", "content": response_message})
