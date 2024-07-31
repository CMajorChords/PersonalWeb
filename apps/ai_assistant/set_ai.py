import streamlit as st
import tiktoken


def get_template_message():
    """
    选择提示词模板

    :return: 提示词模板
    """
    if st.session_state["language"] == "中文":
        template_options = {"文献阅读": "apps/ai_assistant/prompt/Literature Analysis.txt",
                            "科学研究": "apps/ai_assistant/prompt/Scientific Research.txt",
                            "逻辑推理": "apps/ai_assistant/prompt/Logical Reasoning.txt"
                            }
        template_select_box_label = "加强AI能力"
    else:
        template_options = {"Literature Analysis": "apps/ai_assistant/prompt/Literature Analysis.txt",
                            "scientific research": "apps/ai_assistant/prompt/Scientific Research.txt",
                            "Logical Reasoning": "apps/ai_assistant/prompt/Logical Reasoning.txt"
                            }
        template_select_box_label = "Enhance AI capabilities"
    template_chosen = st.selectbox(
        label=template_select_box_label,
        options=template_options.keys(),
        key="template_chosen",
        index=2,
    )
    with open(template_options[template_chosen], "r", encoding="utf-8") as f:
        template = f.read()
    return [{"role": "system", "content": template}]


def choose_model(authenticated: bool):
    """
    将用户选择的AI模型转化为模型名称

    :param authenticated: 是否经过验证
    :return: 选择的AI模型
    """
    # 模型选择
    if authenticated:
        model_options = {"gpt 4o": "gpt-4o-ca",
                         "gpt 4o mini": "gpt-4o-mini",
                         "gpt 4 turbo": "gpt-4-turbo-ca",
                         "gpt 4": "gpt-4-ca",
                         "gpt 3.5 turbo": "gpt-3.5-turbo",  # "gpt-3.5-turbo-ca",
                         # "claude 3.5 sonnet": "claude-3-5-sonnet-20240620",
                         }
    else:
        model_options = {"gpt 3.5 turbo": "gpt-3.5-turbo",
                         "gpt 4o mini": "gpt-4o-mini",
                         }
    model_chosen = st.selectbox(label="选择AI模型" if st.session_state["language"] == "中文" else "Select AI model",
                                options=model_options.keys(),
                                index=1,
                                key="model_chosen",
                                )
    return model_options[model_chosen]


def compute_tokens(response) -> int:
    """
    计算response中的token数量

    :param response: AI返回的response
    :return: token数量
    """
    return len(response.choices[0].text.split())
