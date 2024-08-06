import streamlit as st


@st.cache_resource
def template2message(template_path: str) -> list:
    """
    读取模板文件，并转化为message格式

    :param template_path: 模板文件路径
    :return: message格式的模板
    """
    with open(template_path, "r", encoding="utf-8") as f:
        return [{"role": "system", "content": f.read()}]


@st.fragment
def process_template_message() -> list:
    """
    选择提示词模板

    :return: 提示词模板
    """
    path = "apps/ai_assistant/process_messages/prompt/"
    if st.session_state["language"] == "中文":
        template_options = {"逻辑推理": path + "Logical Reasoning.txt",
                            "文献阅读": path + "Literature Analysis.txt",
                            "科学研究": path + "Scientific Research.txt",
                            "Python编程": path + "Python Programming.txt",
                            }
        template_select_box_label = "加强AI能力"
    else:
        template_options = {"Logical Reasoning": path + "Logical Reasoning.txt",
                            "Literature Analysis": path + "Literature Analysis.txt",
                            "scientific research": path + "Scientific Research.txt",
                            "Python Programming": path + "Python Programming.txt",
                            }
        template_select_box_label = "Enhance AI capabilities"
    template_chosen = st.selectbox(label=template_select_box_label,
                                   options=template_options.keys(),
                                   key="template_chosen",
                                   index=0,
                                   )
    return template2message(template_options[template_chosen])
