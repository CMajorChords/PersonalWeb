import streamlit as st


def choose_model(authenticated: bool):
    """
    将用户选择的AI模型转化为模型名称

    :param authenticated: 是否经过验证
    :return: 选择的AI模型
    """
    # 模型选择
    if authenticated:
        model_options = {"gpt 4o mini": "gpt-4o-mini",
                         "gpt 4o(long output)": "gpt-4o-2024-08-06",
                         "gpt 4o": "gpt-4o-ca",
                         "gpt 4 turbo": "gpt-4-turbo-ca",
                         "gpt 4": "gpt-4-ca",
                         # "gpt 3.5 turbo": "gpt-3.5-turbo",  # "gpt-3.5-turbo-ca",
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
