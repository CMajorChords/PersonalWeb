import streamlit as st
from apps.ai_assistant.process_messages.embed_file import extract_text, extract_image


def text_uploader_callback(flag: bool):
    """
    每次上传文件后，更新flag.
    """
    st.session_state.text_upload_flag = flag


def image_uploader_callback(flag: bool):
    """
    每次上传文件后，更新flag.
    """
    st.session_state.image_upload_flag = flag


@st.fragment
def process_document_message(authenticated: bool):
    """
    获取上传文档的数据，并更新session_state中的messages_input和messages_show

    """
    # 上传文档
    if st.session_state["language"] == "中文":
        uploaded_file_label = "上传文档"
        help_str = "支持PDF、Word、Markdown、代码文件等。注意扫描版的pdf是图片不是文本哦，无法识别"
    else:
        uploaded_file_label = "Upload document"
        help_str = "Support PDF, Word, markdown, code files, etc. Note that the scanned PDF is images, not text"
    uploaded_document = st.file_uploader(label=uploaded_file_label,
                                         accept_multiple_files=False,
                                         on_change=text_uploader_callback,
                                         args=(True,),
                                         help=help_str,
                                         key=st.session_state.text_uploader_key,
                                         disabled=not authenticated,
                                         )
    # 提取文本并处理message
    if uploaded_document:
        text, file_name = extract_text(uploaded_document)
        text = {"role": "user", "content": text}
        file_name = {"role": "user", "content": f"`{file_name}`"}
        if st.session_state.text_upload_flag:
            # 在messages中添加文本或文件名
            st.session_state.messages_show.append(file_name)
            st.session_state.messages_input.append(text)
            # 防止反复添加文本
            st.session_state.text_upload_flag = False
            # 刷新text_uploader（用于删除上一个已经上传过文本的file_uploader）
            st.session_state.text_uploader_key += 1
            st.rerun()  # 重新运行以清除上一个已经上传过文本的file_uploader


@st.fragment
def process_image_message(authenticated: bool):
    """
    获取上传图片的数据，并更新session_state中的messages_input和messages_show

    """
    # 上传图片
    if st.session_state["language"] == "中文":
        uploaded_image_label = "上传图片"
        help_str = "支持jpg、png、bmp等格式。注意不支持动图哦"
    else:
        uploaded_image_label = "Upload image"
        help_str = "Support jpg, png, bmp and other formats. Note that GIF is not supported"
    uploaded_image = st.file_uploader(label=uploaded_image_label,
                                      accept_multiple_files=False,
                                      on_change=image_uploader_callback,
                                      args=(True,),
                                      help=help_str,
                                      key=st.session_state.image_uploader_key,
                                      disabled=not authenticated,
                                      )
    # 提取图片并处理message
    if uploaded_image:
        image_base64, image_pil = extract_image(uploaded_image)
        image = {"role": "user",
                 "content": [{"type": "image_url",
                              "image_url": {"url": f"data:image/png;base64,{image_base64}", }
                              }]
                 }
        image_show = {"role": "user", "content": image_pil}
        if st.session_state.image_upload_flag:
            # 在messages中添加图片
            st.session_state.messages_show.append(image_show)
            st.session_state.messages_input.append(image)
            # 防止反复添加图片
            st.session_state.image_upload_flag = False
            # 刷新image_uploader
            st.session_state.image_uploader_key += 1
            st.rerun()
