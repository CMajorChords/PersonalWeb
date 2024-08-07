import streamlit as st
from PIL import Image
from io import BytesIO
import base64
import copy


def to_save_format(messages_input: list, messages_show: list) -> (list, list):
    """
    将消息转换为可保存的格式。搜查消息中的字典，若字典content对应的是图片，则将图片转换为base64编码。
    :param messages_input: 用户输入的消息列表
    :param messages_show: 显示的消息列表
    :return: 可保存的消息列表
    """

    def transfer_message_save(message_show_dict) -> dict:
        """
        将消息中的图片转换为base64编码，注意，如果已经是base64编码，则不再转换。
        :param message_show_dict: 显示的消息字典
        :return:
        """
        if isinstance(message_show_dict["content"], Image.Image):
            messages_show_dict_copy = copy.deepcopy(message_show_dict)
            messages_show_dict_copy["content"] = None
        return message_show_dict

    messages_show = [transfer_message_save(message_show_dict) for message_show_dict in messages_show]
    return messages_input, messages_show


def to_load_format(messages_input: list, messages_show: list) -> (list, list):
    """
    将可保存的消息转换为可加载的格式。搜查消息中的字典，若字典content对应的是图片的base64编码，则将base64编码转换为图片。
    :param messages_input: 用户输入的消息列表
    :param messages_show: 显示的消息列表
    :return: 可加载的消息列表
    """

    def transfer_message_load(message_show_dict, message_input_dict) -> dict:
        """
        根据message_input_dict中的信息，将message_show_dict中的None转化为Image.Image。
        :param message_show_dict: 显示的消息字典
        :param message_input_dict: 输入的消息字典
        :return: 转换后的显示的消息字典
        """
        if message_show_dict["content"] is None:
            # 首先截取base64编码
            image_base64 = message_input_dict["content"][0]["image_url"]["url"].split(",")[-1]
            # 将base64编码转化为Image.Image
            image = Image.open(BytesIO(base64.b64decode(image_base64)))
            message_show_dict_copy = copy.deepcopy(message_show_dict)
            message_show_dict_copy["content"] = image
        return message_show_dict

    messages_show = list(map(transfer_message_load, messages_show, messages_input))
    return messages_input, messages_show
