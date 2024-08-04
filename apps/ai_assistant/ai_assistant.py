import streamlit as st
from apps.ai_assistant.select_model import choose_model
from apps.ai_assistant.compute_token import compute_token_price
from apps.ai_assistant.process_messages import (process_template_message,
                                                process_document_message,
                                                process_image_message,
                                                clear_chat_history)
from utils import write_saying, check_password
from apps.ai_assistant.chat import chat_with_history

# 初始化session_state
# 消息管理
if "messages_input" not in st.session_state:
    st.session_state.messages_input = []
if "messages_show" not in st.session_state:
    st.session_state.messages_show = []
# 文件上传
if "text_upload_flag" not in st.session_state:
    st.session_state.text_upload_flag = False
if "image_upload_flag" not in st.session_state:
    st.session_state.image_upload_flag = False
if "text_uploader_key" not in st.session_state:
    st.session_state["text_uploader_key"] = 0
if "image_uploader_key" not in st.session_state:
    st.session_state["image_uploader_key"] = 0.1

# 密码设置
if check_password(text_label_zn="输入密码解锁更多功能", text_label_en="Enter password to unlock more features"):
    authentication = True
else:
    authentication = False

# UI基础
write_saying("ai_assistant")
st.subheader("fun💤 AI", anchor=False)
col1, col2 = st.columns((3, 1))

# ai设置
with col2.container(border=True):
    # 选择模型
    model_chosen = choose_model(authenticated=authentication)
    # 提示词模板
    message_template = process_template_message()

# 聊天设置
# 设置文档上传
with col1.container(border=True):
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        process_document_message(authenticated=authentication)
        clear_chat_history()
    with sub_col2:
        process_image_message(authenticated=authentication)
        compute_token_price(messages_input=st.session_state.messages_input,
                            messages_show=st.session_state.messages_show,
                            message_prompt_template=message_template,
                            model=model_chosen)

# 聊天
chat_with_history(authentication=authentication,
                  message_template=message_template,
                  model=model_chosen,
                  )
