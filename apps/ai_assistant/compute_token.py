import tiktoken
import streamlit as st


def num_tokens_from_messages(messages, model):
    """根据消息计算token数量"""
    model_options = {"gpt 4o": "gpt-4o-ca",
                     "gpt 4 turbo": "gpt-4-turbo-ca",
                     "gpt 4": "gpt-4-ca",
                     "gpt 4o mini": "gpt-4o-mini",
                     "gpt 3.5 turbo": "gpt-3.5-turbo-ca",
                     }
    encoding = tiktoken.encoding_for_model(model_options[model])
    tokens_per_message = 3
    tokens_per_name = 3
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def count_prompt_output_token(messages, model):
    """将最后一条消息作为output，计算token数量"""
    if model == "gpt 4" or model == "gpt 4o":
        model += "-"
    encoding = tiktoken.encoding_for_model(model)
    tokens_per_message = 3
    tokens_per_name = 3
    # 计算prompt的token数量
    prompt_tokens = 0
    for message in messages[:-1]:
        prompt_tokens += tokens_per_message
        for key, value in message.items():
            try:
                prompt_tokens += len(encoding.encode(value))
                if key == "name":
                    prompt_tokens += tokens_per_name
            except:
                pass
    prompt_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    # 计算output的token数量
    output_tokens = 0
    output_tokens += tokens_per_message
    for key, value in messages[-1].items():
        try:
            output_tokens += len(encoding.encode(value))
        except:
            pass
    output_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return prompt_tokens, output_tokens


def compute_token_price(messages, model):
    """计算token价格"""
    # 判断message是否为空
    if st.session_state["language"] == "中文":
        label = "计算本次提问花费"
        str_prompt_token = "本次提问输入token数量："
        str_output_token = "本次提问输出token数量："
        str_token_price = "本次提问花费："
    else:
        label = "Cost of this question"
        str_prompt_token = "Input tokens:"
        str_output_token = "Output tokens:"
        str_token_price = "Cost:"
    if st.button(label=label):
        if messages:
            price_table = {
                "gpt-4o-ca": (0.02, 0.06),
                "gpt-4-turbo-ca": (0.04, 0.12),
                "gpt-4-ca": (0.12, 0.24),
                "gpt-4o-mini": (0.00105, 0.0042),
                "gpt-3.5-turbo-ca": (0.001, 0.003),
            }
            prompt_tokens, output_tokens = count_prompt_output_token(messages, model)
            price = price_table[model]
            token_price = (prompt_tokens * price[0] + output_tokens * price[1]) / 1000
            # token price应该保留两位小数
            st.toast(f"{str_prompt_token}{prompt_tokens}", icon="🪙")
            st.toast(f"{str_output_token}{output_tokens}", icon="🪙")
            st.toast(f"**{str_token_price}{token_price:.3f}¥**", icon="🪙")
        else:
            st.toast(f"{str_prompt_token} 0", icon="🪙")
            st.toast(f"{str_output_token} 0", icon="🪙")
            st.toast(f"{str_token_price} 0", icon="🪙")

    # gpt-3.5-turbo-ca	0.001 / 1K Tokens	0.003 / 1K Tokens	支持	Azure openai中转(也属于官方模型的一种)价格便宜, 但是回复的慢一些
    # gpt-3.5-turbo	0.0035 / 1K Tokens	0.0105 / 1K Tokens	支持	默认模型，等于gpt-3.5-turbo-0125
    # gpt-3.5-turbo-1106	0.007 / 1K Tokens	0.014 / 1K Tokens	支持	2023年11月6日更新的模型
    # gpt-3.5-turbo-0125	0.0035 / 1K Tokens	0.0105 / 1K Tokens	支持	2024年1月25日最新模型，数据最新，价格更更低，速度更快，修复了一些1106的bug。
    # gpt-3.5-turbo-0301	0.0105 / 1K Tokens	0.014 / 1K Tokens	支持	适合快速回答简单问题
    # gpt-3.5-turbo-0613	0.0105 / 1K Tokens	0.014 / 1K Tokens	支持	适合快速回答简单问题，支持Function
    # gpt-3.5-turbo-16k	0.021 / 1K Tokens	0.028 / 1K Tokens	支持	适合快速回答简单问题,字数更多
    # gpt-3.5-turbo-16k-0613	0.021 / 1K Tokens	0.028 / 1K Tokens	支持	适合快速回答简单问题，字数更多，支持Function
    # gpt-4	0.21 / 1K Tokens	0.42 / 1K Tokens	支持	默认模型，等于gpt-4-0613
    # gpt-4o	0.035/1K Tokens + 图片费用[2]	0.105/1K Tokens / 1K Tokens	支持	Openai 最新模型, 价格更低, 速度更快更聪明
    # gpt-4o-mini	0.00105/1K Tokens + 图片费用[2]	0.0042/1K Tokens / 1K Tokens	支持	Openai 最新模型, 价格更低, 输出质量在3.5之上4o之下, 并且支持读图
    # gpt-4-0613	0.21 / 1K Tokens	0.42 / 1K Tokens	支持	2023年6月13日更新的模型
    # gpt-4-turbo-preview	0.07 / 1K Tokens	0.21 / 1K Tokens	支持	最新模型，输入128K，输出最大4K，知识库最新2023年4月, 此模型始终指向最新的4的preview模型
    # gpt-4-0125-preview	0.07 / 1K Tokens	0.21 / 1K Tokens	支持	2024年1月25日更新的模型，输入128K，输出最大4K，知识库最新2023年4月, 修复了一些1106的bug
    # gpt-4-1106-preview	0.07 / 1K Tokens	0.21 / 1K Tokens	支持	2023年11月6日更新的模型，输入128K，输出最大4K，知识库最新2023年4月
    # gpt-4-vision-preview	0.07 / 1K Tokens + 图片费用[2]	0.21 / 1K Tokens	支持	多模态，支持图片识别
    # gpt-4-turbo	0.07 / 1K Tokens + 图片费用[2]	0.21 / 1K Tokens	支持	Openai 最新模型多模态，支持图片识别，支持函数tools
    # gpt-4-turbo-2024-04-09	0.07 / 1K Tokens + 0.10115*图片个数[3]	0.21 / 1K Tokens	支持	Openai 最新模型多模态，支持图片识别，支持函数tools
    # gpt-4-ca	0.12 / 1K Tokens	0.24 / 1K Tokens	支持	Azure openai中转 对标gpt-4(也属于官方模型的一种)价格便宜
    # gpt-4-turbo-ca	0.04 / 1K Tokens + 0.0578*图片个数[3]	0.12 / 1K Tokens	支持	Azure openai中转 对标gpt-4-turbo(也属于官方模型的一种)价格便宜, 但是回复的慢一些
    # gpt-4o-ca	0.02 / 1K Tokens + 0.0289*图片个数[2]	0.06 / 1K Tokens	支持	Azure openai中转 对标gpt-4o(也属于官方模型的一种)价格便宜
    # gpt-3.5-turbo-instruct	0.0105 / 1K Tokens	0.014 / 1K Tokens	支持	Completions模型 用于文本生成，提供准确的自然语言处理模型一般人用不上
