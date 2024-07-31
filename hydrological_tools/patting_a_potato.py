import json
import datetime
import time
import streamlit as st
import numpy as np
from streamlit_lottie import st_lottie


def patting_a_potato_CN():
    st.subheader("调戏土豆", anchor=False)
    st.write("从前有一个土豆在森林里迷了路")
    prompt = st.chat_input(
        placeholder="回答土豆：",
        key="回答土豆",
    )
    # 获取今天的日期
    today = datetime.date.today()
    # 设置2023年10月13号为基准日期
    base_date = datetime.date(2023, 10, 8)
    # 计算距离基准日期的天数
    days = (today - base_date).days
    with st.chat_message("🥔"):
        st.write("你知道我的家在哪里吗？")
    if prompt is not None:
        user = st.chat_message("user")
        user.write(prompt)
        if prompt == "你想我一下我就告诉你" or prompt == "你想我我就告诉你":
            with st.chat_message("🥔"):
                st.write("我想你了")
            with st.chat_message("🥔"):
                st.write(f"已经想你{days}天了，一直都很想你")
        elif prompt == "我不想你":
            with st.chat_message("🥔"):
                st.write("我也不想你🤯")
        elif prompt == "我想你" or prompt == "我想你了":
            with st.chat_message("🥔"):
                st.write("我也想你")
        elif prompt == "我爱你" or prompt == "我喜欢你":
            with st.chat_message("🥔"):
                st.write("真的假的😮")
            # 停顿五秒
            with st.spinner("土豆正在思考"):
                time.sleep(6)
            with st.chat_message("🥔"):
                st.write("我爱你❤")
            with st.chat_message("🥔"):
                st.write("我知道我的家在哪里了")
        elif prompt == "你喜欢我我就告诉你" or prompt == "你爱我我就告诉你":
            with st.chat_message("🥔"):
                st.write("我爱你（不带犹豫的）")
            with st.chat_message("🥔"):
                st.write("我知道我的家在哪里了")
        elif prompt == "你睡一觉我就告诉你":
            with st.chat_message("🥔"):
                st.write("好吧，晚安💤")
        elif prompt == "你让我爽一下我就告诉你":
            with st.chat_message("🥔"):
                st.write("咦，不可以色色")
        else:
            with st.chat_message("🥔"):
                st.write("骗子，鬼才信你")
        with st.chat_message("🥔"):
            st.write("快拍拍我")
    # 创建按钮
    st.button(
        "拍一拍土豆",
    )
    st_lottie(
        json.load(open("lottie/土豆.json")),
        height=200,
        speed=np.random.randint(1, 10)
    )


def patting_a_potato_EN():
    st.subheader("Patting Potatoes", anchor=False)
    st.write(
        "Once upon a time, there was a potato who got lost in the forest. The potato asked, 'Do you know where my home is?'")
    prompt = st.chat_input(
        placeholder="Answer the potato:",
        key="Answer the potato",
    )
    # 获取今天的日期
    today = datetime.date.today()
    # 设置2023年10月13号为基准日期
    base_date = datetime.date(2023, 10, 13)
    # 计算距离基准日期的天数
    days = (today - base_date).days
    # 如果距离基准日期小于0，就显示已经过去了多少天
    with st.chat_message("🥔"):
        st.write("Do you know where my home is?")
    if prompt is not None:
        user = st.chat_message("user")
        user.write(prompt)
        if prompt == "你想我一下我就告诉你" or prompt == "你想我我就告诉你":
            with st.chat_message("🥔"):
                st.write("I miss you")
            with st.chat_message("🥔"):
                st.write(f"I've missed you for {days} days, and I've always missed you")
        elif prompt == "我不想你":
            with st.chat_message("🥔"):
                st.write("I don't miss you either")
        elif prompt == "我想你" or prompt == "我想你了":
            with st.chat_message("🥔"):
                st.write("I miss you too")
        elif prompt == "我爱你" or prompt == "我喜欢你":
            with st.chat_message("🥔"):
                st.write("Really??")
            # 停顿五秒
            with st.spinner("The potato is thinking"):
                time.sleep(6)
            with st.chat_message("🥔"):
                st.write("I love you❤")
            with st.chat_message("🥔"):
                st.write("I know where my home is")
        elif prompt == "你喜欢我我就告诉你" or prompt == "你爱我我就告诉你":
            with st.chat_message("🥔"):
                st.write("I love you")
            with st.chat_message("🥔"):
                st.write("I know where my home is")
        elif prompt == "你睡一觉我就告诉你":
            with st.chat_message("🥔"):
                st.write("Okay, good night💤")
        elif prompt == "你让我爽一下我就告诉你":
            with st.chat_message("🥔"):
                st.write("Hey, you can't do that")
        else:
            with st.chat_message("🥔"):
                st.write("Liar, who would believe you")
        with st.chat_message("🥔"):
            st.write("Pat me")
    # 创建按钮
    st.button(
        "拍一拍土豆",
    )
    st_lottie(
        json.load(open("lottie/土豆.json")),
        height=200,
        speed=np.random.randint(1, 20)
    )
