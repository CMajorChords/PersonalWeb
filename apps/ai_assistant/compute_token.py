import tiktoken
import streamlit as st
from tiktoken import Encoding
from math import ceil


def count_str_message_tokens(message: dict,
                             model_encoding: Encoding,
                             base_tokens: int = 6,
                             ) -> int:
    """
    计算单个文本消息的token数量

    :param message: 单个文本消息
    :param model_encoding: 模型编码
    :param base_tokens: 每个message的额外token数量
    :return: 单个文本消息的token数量
    """
    message_tokens = model_encoding.encode(message["content"])
    return len(message_tokens) + base_tokens


def count_image_message_tokens(message: dict,
                               base_tokens: int,
                               tile_tokens: int,
                               tile_size: int,
                               max_tokens: int,
                               ) -> int:
    """
    计算单个图片消息的token数量，图片格式应该为PIL.Image

    :param message: 单个图片消息
    :param base_tokens: 每个message的额外token数量
    :param tile_tokens: 每个tile的token数量
    :param tile_size: tile的大小
    :param max_tokens: 单个图片消息的最大token数量
    :return: 单个图片消息的token数量
    """
    width, height = message["content"].size
    num_tiles = ceil(width / tile_size) * ceil(height / tile_size)
    image_tokens = num_tiles * tile_tokens
    if image_tokens > max_tokens:
        image_tokens = max_tokens
    return base_tokens + image_tokens


def count_messages_tokens(messages_input: list,
                          messages_show: list,
                          message_prompt_template: list,
                          model: str,
                          base_tokens_str_message: int = 3,
                          base_tokens_image_message: int = 85 + 3,
                          tile_tokens: int = 170,
                          tile_size: int = 512,
                          image_max_tokens: int = 1445,
                          ) -> (int, int):
    """
    计算消息的token数量.
    从message_input中判断消息类型。
    如果是文本消息，将从messages_input中获取。
    如果是图片消息，将从messages_show中获取。

    :param messages_input: 输入进client的消息
    :param messages_show: 显示在UI上的消息
    :param message_prompt_template: 提示词模板
    :param model: 模型名称
    :param base_tokens_str_message: 每个文本message的额外token数量
    :param base_tokens_image_message: 每个图片message的额外token数量
    :param tile_tokens: 计算图片tokens时每个tile的token数量
    :param tile_size: 计算图片tokens时tile的大小
    :param image_max_tokens: 单个图片消息的最大token数量
    :return: 消息的token数量
    """
    # 配置模型编码
    if model == "gpt 4" or model == "gpt 4o":
        model += "-"
    model_encoding = tiktoken.encoding_for_model(model)
    # 计算input tokens
    messages_input_prompt = messages_input[:-1]
    messages_show_prompt = messages_show[:-1]
    prompt_str_tokens, prompt_image_tokens = 0, 0  # 3 tokens for start token, end token and separator token
    for idx, message in enumerate(messages_input_prompt):
        if isinstance(message["content"], str):
            prompt_str_tokens += count_str_message_tokens(message=message,
                                                          model_encoding=model_encoding,
                                                          base_tokens=base_tokens_str_message)
        else:
            message = messages_show_prompt[idx]
            prompt_image_tokens += count_image_message_tokens(message=message,
                                                              base_tokens=base_tokens_image_message,
                                                              tile_tokens=tile_tokens,
                                                              tile_size=tile_size,
                                                              max_tokens=image_max_tokens,
                                                              )
    # 计算output tokens
    message_input_output = messages_input[-1]
    message_show_output = messages_show[-1]
    if isinstance(message_input_output["content"], str):
        output_tokens = count_str_message_tokens(message=message_input_output,
                                                 model_encoding=model_encoding,
                                                 base_tokens=base_tokens_str_message)
    else:
        output_tokens = count_image_message_tokens(message=message_show_output,
                                                   base_tokens=base_tokens_image_message,
                                                   tile_tokens=tile_tokens,
                                                   tile_size=tile_size,
                                                   max_tokens=image_max_tokens,
                                                   )
    output_tokens += 3
    # 计算template tokens
    prompt_str_tokens += count_str_message_tokens(message=message_prompt_template[0],
                                                  model_encoding=model_encoding,
                                                  base_tokens=base_tokens_str_message)
    return prompt_str_tokens, prompt_image_tokens, output_tokens


def compute_token_price(messages_input: list,
                        messages_show: list,
                        message_prompt_template: list,
                        model: str,
                        ):
    """
    计算本次提问花费

    :param messages_input: 输入进client的消息
    :param messages_show: 显示在UI上的消息
    :param message_prompt_template: 提示词模板
    :param model: 模型名称
    """
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
        if messages_input:
            price_table = {
                "gpt-4o-ca": (0.02, 0.0289, 0.06),
                "gpt-4-turbo-ca": (0.04, 0.0578, 0.12),
                "gpt-4-ca": (0.12, 0.12, 0.24),
                "gpt-4o-mini": (0.00105, 0.035, 0.0042),
                "gpt-3.5-turbo-ca": (0.001, 0.001, 0.003),
            }
            prompt_str_tokens, prompt_image_tokens, output_tokens = count_messages_tokens(
                messages_input=messages_input,
                messages_show=messages_show,
                message_prompt_template=message_prompt_template,
                model=model)
            price = price_table[model]
            token_price = (prompt_str_tokens * price[0] +
                           prompt_image_tokens * price[1] +
                           output_tokens * price[2]) / 1000
            # token price应该保留三位小数
            st.toast(f"{str_prompt_token}{prompt_str_tokens + prompt_image_tokens}", icon="🪙")
            st.toast(f"{str_output_token}{output_tokens}", icon="🪙")
            st.toast(f"**{str_token_price}{token_price:.3f}￥**", icon="🪙")
        else:
            st.toast(f"{str_prompt_token}0", icon="🪙")
            st.toast(f"{str_output_token} 0", icon="🪙")
            st.toast(f"**{str_token_price} 0￥**", icon="🪙")

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
