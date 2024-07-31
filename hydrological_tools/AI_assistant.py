import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
from io import BytesIO


@st.cache_data(
    show_spinner="读取PDF中...文件里的字数超过15k个token时，AI就读不下去了" if st.session_state.language == "中文" else "Reading PDF...When the number of words in the file exceeds 15k tokens, AI will not read it")
def read_pdf(uploaded_file):
    pdf_reader = PdfReader(BytesIO(uploaded_file.getvalue()))
    pdf_text = "这是一个pdf文件里的内容:" if st.session_state.language == "中文" else "This is the content in a PDF file:"
    for page in pdf_reader.pages:
        pdf_text += page.extract_text() or ""
    pdf_text = pdf_text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    pdf_text = ' '.join(pdf_text.split())
    return pdf_text


def select_model(model_name: str):
    model_options = {
        ":rainbow[gpt 3.5 turbo]": "gpt-3.5-turbo-0125",
        ":rainbow[gpt 4]": "gpt-4",
    }
    return model_options[model_name]


def AI_assistant():
    # 显示标题
    st.subheader("fun💤 AI", anchor=False)
    st.write(
        "使用基于transformer架构的大语言AI模型" if st.session_state.language == "中文" else "Use transformer-based large language "
                                                                                        "AI model")
    model_chosen = select_model(st.radio(
        "选择AI模型" if st.session_state.language == "中文" else "Select AI model",
        [":rainbow[gpt 3.5 turbo]", ":rainbow[gpt 4]"],
        captions=["最多60次提问/小时", "最多3次提问/天"] if st.session_state.language == "中文" else [
            "Up to 60 questions/hour",
            "Up to 3 questions/day"],
        horizontal=True,
        label_visibility="collapsed",
    ))
    # 设置openai的api_key
    client = OpenAI(
        api_key=st.secrets["OPENAI_API_KEY"],
        base_url="https://api.chatanywhere.cn/v1",  # https://api.chatanywhere.cn/v1https://api.chatanywhere.com.cn
    )
    # 初始化对话
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 设置标志，表示PDF已经上传
    if "pdf_uploaded" not in st.session_state:
        st.session_state.pdf_uploaded = False
    # 上传pdf文件
    uploaded_file = st.file_uploader(
        "可以上传PDF文件，根据文件内容进行对话" if st.session_state.language == "中文" else "You can upload PDF files and chat based on "
                                                                                           "the content of the file",
        type=["pdf"],
        help="可以上传文献、书籍等PDF文件" if st.session_state.language == "中文" else "You can upload PDF files such as literature "
                                                                                      "and books",
    )
    if uploaded_file and not st.session_state.pdf_uploaded:
        pdf_text = read_pdf(uploaded_file)
        st.session_state["pdf_text"] = {"role": "user", "content": pdf_text}
        st.session_state.pdf_uploaded = True  # 设置标志，表示PDF已经上传
    # 从session_state中获取用户输入的信息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # # 接收用户输入
    try:
        if prompt := st.chat_input(
                "对AI说点什么..." if st.session_state.language == "中文" else "Say something to AI..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                if "pdf_text" in st.session_state:
                    all_messages = [st.session_state.pdf_text] + st.session_state.messages
                else:
                    all_messages = st.session_state.messages
                stream = client.chat.completions.create(
                    model=model_chosen,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in all_messages
                    ],
                    stream=True,
                    prompt=st.session_state.messages[-1]["content"],
                )
                response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
    except:
        # 如果pdf存在的话，提示pdf文件太大了
        if st.session_state.pdf_uploaded:
            st.error(
                "pdf文件太大了，AI不想读了" if st.session_state.language == "中文" else "The PDF file is too large, AI "
                                                                                       "doesn't want to read it")
        else:
            st.error(
                "AI不想理你了，刷新页面重新开始吧" if st.session_state.language == "中文" else "AI doesn't want to talk to you, "
                                                                                              "refresh the apps "
                                                                                              "to start over")
