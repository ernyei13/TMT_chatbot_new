import sys
import os
import streamlit as st

# Make sure agents/ is importable
current = os.path.dirname(os.path.abspath(__file__))
root    = os.path.abspath(os.path.join(current, ".."))
if root not in sys.path:
    sys.path.append(root)



from chat_tab import render_chat_tab
from eval_tab import render_eval_tab
# Adjust the CSS to align the title and tab selector horizontally and reduce their height
import streamlit as st

# --- Login screen ---
# load credentials from Streamlit secrets
TMT_USERNAME = st.secrets["TMT_USERNAME"]
TMT_PASSWORD = st.secrets["TMT_PASSWORD"]

if "logged_in" not in st.session_state:
    st.title("Please log in")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == TMT_USERNAME and password == TMT_PASSWORD:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("❌ Invalid credentials")
    st.stop()

st.set_page_config(page_title="TMT Toolkit", layout="centered", initial_sidebar_state="collapsed")

st.title("TMT Toolkit")

# Add custom CSS to fix the title at the top of the screen
st.markdown("""
    <style>
        .css-18e3th9 {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1000;
            background-color: white;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs {
            margin-top: 60px; /* Adjust to avoid overlap with the fixed title */
        }
    </style>
""", unsafe_allow_html=True)

tab_chat, tab_eval = st.tabs(["Chat", "Evaluation"])



# Adjust the sidebar width using custom CSS
st.markdown("""
    <style>
        .css-1d391kg .css-1d391kg {
            width: 200px; /* Adjust the width as needed */
        }
    </style>
""", unsafe_allow_html=True)


# Sidebar toggles
with st.sidebar:
    # Additional Options
    st.header("Agent Settings")
    st.checkbox("Enable Planner Agent", 
                key="enable_planner_agent")
    st.checkbox("Enable Reviewer Agent",
                key="enable_reviewer_agent")
    st.selectbox("Max Retry of Reviewer", range(6), key="max_retry_reviewer", index=1)  # Default to 1

    st.selectbox("Max Retry of SysML Filter", range(41), key="max_retry_sysml_filter", index=8)  # Default to 8
    st.selectbox("Max Documents retrieved by RAG", range(41), key="max_rag", index=8)  # Default to 8

    st.header("View Settings")
    st.checkbox("Show Thinking Steps",
                key="show_thinking_steps")
    st.checkbox("Show Context",
                key="show_context")
    st.checkbox("Show Elements",
                key="show_elements")
    st.checkbox("Show Diagrams",
                key="show_diagrams")


# Tabs
with tab_chat:
    render_chat_tab()

with tab_eval:
    render_eval_tab()
