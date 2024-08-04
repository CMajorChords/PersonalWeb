import streamlit as st

st.balloons()

# 头像
col = st.columns(7)
col[3].image("data/image/头像无背景.png", use_column_width=True)


@st.cache_resource
def self_introduction(language: str):
    if language == "中文":
        st.header("About me", anchor=False)
        with open("data/markdown/about_me_ZN.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.header("关于我", anchor=False)
        with open("data/markdown/about_me_EN.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())


self_introduction(language=st.session_state["language"])

# 联系我、反馈和建议、请我喝咖啡
col1, col2, col3 = st.columns(3)
if st.session_state["language"] == "中文":
    with col1:
        popover = st.popover(label="📬联系我", use_container_width=True)
        popover.markdown("邮箱：1831372118＠qq.com")
        popover.markdown("QQ：1831372118")
        popover.markdown("微信：fz18671111056")
    with col2:
        popover = st.popover(label="🗣️反馈和建议", use_container_width=True)
        popover.markdown("对网站的建议可以在[反馈文档](https://docs.qq.com/doc/DU21SRXFYbXVBeWtF)中提出")
        popover.markdown("注意腾讯文档需要登录才能编辑哦🫠")
    with col3:
        with st.popover(label="🍵请我喝咖啡", use_container_width=True):
            st.markdown("蟹蟹🥰，一分钱就好，其它自己留着攒老婆本")
            st.image("data/image/收款码.png")
else:
    with col1:
        with st.popover(label="📬Contact me", use_container_width=True):
            st.markdown("Email: 1831372118＠qq.com")
            st.markdown("QQ: 1831372118")
            st.markdown("WeChat: fz18671111056")
    with col2:
        with st.popover(label="🗣️Feedback & Suggestions", use_container_width=True):
            st.markdown("Suggestions for the website can be made in the "
                        "[feedback document](https://docs.qq.com/doc/DU21SRXFYbXVBeWtF)")
            st.markdown("Note that you need to log in to Tencent documents to edit🫠")
    with col3:
        with st.popover(label="🍵Buy me a coffee", use_container_width=True):
            st.markdown("A penny is enough,thank you🥰")
            st.image("data/image/收款码.png")


# 一些与研究方向相关的图片
@st.cache_resource
def show_image(language: str):
    with st.container(border=True):
        col_1, col_2, col_3 = st.columns(3)
        col_1.image("data/image/集成学习.png", use_column_width=True)
        col_2.image("data/image/复合图.png", use_column_width=True)
        col_3.image("data/image/地图.png", use_column_width=True)
    with st.popover("水循环" if language == "中文" else "Water Cycle"):
        st.image("data/image/水循环英文版.png", use_column_width=True)


show_image(language=st.session_state["language"])
