渐变团 = 'https://lottie.host/d5d80b45-e0eb-4881-9b1c-8dbeb346c086/bPTVc2rJSH.json'
地球 = 'https://lottie.host/6d7210d8-5272-487e-8374-4c1aa95dd9d7/NJZOFBK4yl.json'
小熊 = 'https://lottie.host/df61a08c-e9b7-4e5f-9f8e-33d05e8bc087/LomgmMZ9TL.json'
花 = 'https://lottie.host/942b0239-7458-4fd8-ad2e-656536884a69/AntJ8l9wDO.json'

import streamlit as st
import json


# 缓存从本地加载的lottie json
@st.cache_resource()
def 像素地球():
    with open('lottie/像素地球.json', 'r') as f:
        json_data = json.load(f)
    return json_data


@st.cache_resource()
def 土豆():
    with open('lottie/土豆.json', 'r') as f:
        json_data = json.load(f)
    return json_data


@st.cache_resource()
def 蝴蝶花():
    with open('lottie/蝴蝶花.json', 'r') as f:
        json_data = json.load(f)
    return json_data


@st.cache_resource()
def 渐变团():
    with open('lottie/渐变团.json', 'r') as f:
        json_data = json.load(f)
    return json_data


@st.cache_resource()
def 小熊():
    with open('lottie/小熊.json', 'r') as f:
        json_data = json.load(f)
    return json_data


@st.cache_resource()
def 斑块地球():
    with open('lottie/斑块地球.json', 'r') as f:
        json_data = json.load(f)
    return json_data


@st.cache_resource()
def 摇摇晃晃的地球():
    with open('lottie/摇摇晃晃的地球.json', 'r') as f:
        json_data = json.load(f)
    return json_data


@st.cache_resource()
def 落叶花():
    with open('lottie/落叶花.json', 'r') as f:
        json_data = json.load(f)
    return json_data
