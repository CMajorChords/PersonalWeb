import streamlit as st
import hmac


def check_password(text_label_zn: str = "密码", text_label_en: str = "Password") -> bool:
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    def quit_authentication(del_password: bool):
        """Quits the authentication process."""
        if del_password:
            del st.session_state["password_correct"]

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        st.sidebar.button("🔒退出验证", on_click=quit_authentication, args=(True,))
        return True

    # Show input for password.
    with st.sidebar.container(border=True):
        if st.session_state["language"] == "中文":
            st.text_input(text_label_zn, type="password", on_change=password_entered, key="password")
            if "password_correct" in st.session_state:
                st.warning("😕密码错误")
        else:
            st.text_input(text_label_en, type="password", on_change=password_entered, key="password")
            if "password_correct" in st.session_state:
                st.warning("😕Incorrect password")
        return False


def quit_authentication():
    """Quits the authentication process."""

    def quit():
        st.session_state["password_correct"] = False
