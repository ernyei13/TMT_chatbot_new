
from __future__ import annotations

import sys
import os
import streamlit as st

import sys
from pathlib import Path

# Make sure agents/ is importable
ROOT_DIR: Path = Path(__file__).resolve().parents[1]  # …/TMT_chatbot
if str(ROOT_DIR) not in sys.path:                     # keep it idempotent
    sys.path.insert(0, str(ROOT_DIR))


    

from chat_tab import render_chat_tab
from eval_tab import render_eval_tab
# Adjust the CSS to align the title and tab selector horizontally and reduce their height

st.set_page_config(page_title="TMT Toolkit", layout="centered", initial_sidebar_state="collapsed")

# --- Login screen ---
# load credentials from Streamlit secrets
TMT_USERNAME = st.secrets["TMT_USERNAME"]
TMT_PASSWORD = st.secrets["TMT_PASSWORD"]
st.session_state.logged_in = True

from dotenv import load_dotenv
load_dotenv(override=True)
EVAL_ENABLED = False



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


st.title("TMT Engineering Assistant")
st.expander(":red[IMPORTANT DISCLAMER]", expanded=False).markdown(
    """
    - The chatbot can answer questions about the TMT documentation and SysML model elements.
    - NO GUARDRAILS ARE RESTRICTING THE SYSTEM. PLEASE ACT ACCORDINGLY AND NOT TRY TO BREAK IT.
    - There is no memory of the conversation, so each question is treated **independently**.
    - v0.0.6

    """
)

# Add custom CSS to fix the title at the top of the screen
st.markdown(
    """
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
            margin-top: 0px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if EVAL_ENABLED:
    tab_chat, tab_eval = st.tabs(["Chat", "Evaluation"])
else:
    tab_chat, = st.tabs(["Chat"])

# Adjust the sidebar width using custom CSS
st.markdown("""
    <style>
        .css-1d391kg .css-1d391kg {
            width: 200px; /* Adjust the width as needed */
        }
    </style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.warning("ONLY USE THE SIDEBAR WHILE THE ASSISTANT IS NOT RUNNING.")
    st.header("Agent Settings")
    st.checkbox("Enable Reviewer Agent", key="enable_reviewer_agent")
    st.selectbox("Max Retry of Reviewer", range(6), key="max_retry_reviewer", index=1)
    st.selectbox("Max Retry of SysML Filter", range(41), key="max_retry_sysml_filter", index=8)
    st.selectbox("Max Documents retrieved by RAG", range(41), key="max_rag", index=8)

    st.header("View Settings")
    st.checkbox("Show Thinking Steps", key="show_thinking_steps", value=True)
    st.checkbox("Show Context", key="show_context")
    st.checkbox("Show Elements", key="show_elements")
    st.checkbox("Show Diagrams", key="show_diagrams")

    


# Tabs
with tab_chat:
    render_chat_tab()
    
if EVAL_ENABLED:
    with tab_eval:
        render_eval_tab()
