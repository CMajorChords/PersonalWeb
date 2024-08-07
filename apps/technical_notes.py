from utils import write_saying
from apps.ai_assistant.process_messages.sl_messages import to_save_format, to_load_format
import pickle
import streamlit as st

# 名言
write_saying("technical_notes")

# 加载测试的messages
st.session_state["messages_input"] = pickle.load(open("messages_input.pkl", "rb"))
st.session_state["messages_show"] = pickle.load(open("messages_show.pkl", "rb"))

# 保存messages
messages_input, messages_show = to_save_format(st.session_state["messages_input"][:],
                                               st.session_state["messages_show"][:])
pickle.dump((messages_input, messages_show), open("messages.pkl", "wb"))

# 加载messages
messages_input, messages_show = pickle.load(open("messages.pkl", "rb"))
messages_input, messages_show = to_load_format(messages_input, messages_show)
st.session_state["messages_input"] = messages_input
st.session_state["messages_show"] = messages_show
