import streamlit as st
import sys
import os
from typing import List, Dict, Any

# --- Add parent directory to path to find the 'agents' module ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    # --- Import the core agent function and message types ---
    from agents.run_graph import execute_agent_query
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError as e:
    st.error(
        f"Failed to import necessary modules. Please ensure 'agents/run_graph.py' exists and all dependencies are installed. Error: {e}"
    )
    st.stop()

# --- Chat Interface ---

def render_chat_tab() -> None:
    """Renders the chat agent interface just like chat_app.py."""
    st.caption("Ask questions about the TMT SysML model and documentation.")

    # Add custom CSS to lock the chat input to the bottom of the screen
    st.markdown("""
        <style>
            .stChatInput {
                position: fixed;
                bottom: 20px;
                width: 60%;
                z-index: 1000;
                align: center;
            }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar settings
    show_thinking_steps = st.session_state.get("show_thinking_steps", False)
    show_context = st.session_state.get("show_context", False)
    show_elements = st.session_state.get("show_elements", False)
    show_diagrams = st.session_state.get("show_diagrams", False)

    # Session state initialization
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content":
             "Hello! How can I help you understand the TMT model today?"}
        ]

    # Display existing chat messages
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            content = message["content"]
            if isinstance(content, str):
                st.markdown(content)
            elif isinstance(content, dict):
                result_data = content
                # Final Answer
                st.markdown("**Final Answer:**")
                st.markdown(result_data.get("final_answer", "No answer generated."))
                st.write("---")
                # Context
                if show_context:
                    st.markdown("**📚 Documentation Context Used:**")
                    context = result_data.get("rag_context", [])
                    if context:
                        for i, chunk in enumerate(context):
                            title = chunk.get("title", f"Source {i+1}")
                            score = chunk.get("score", 0.0)
                            text = chunk.get("text", "No text available.")
                            with st.expander(f"Context: {title} (Score: {score:.3f})", expanded=False):
                                st.markdown(f"**Source:** `{title}`")
                                st.markdown(f"**Score:** `{score:.3f}`")
                                st.info(text)
                    else:
                        st.caption("No documentation context was retrieved.")
                    st.write("---")
                # Elements
                if show_elements:
                    st.markdown("**🧩 Relevant Model Elements:**")
                    elements = result_data.get("model_query_result", [])
                    if elements:
                        if isinstance(elements, list):
                            for i, element in enumerate(elements):
                                if isinstance(element, dict):
                                    name = element.get("name", f"Element {i+1}")
                                    eid = element.get("id", "No ID")
                                    with st.expander(f"Element: {name} (ID: {eid})", expanded=False):
                                        st.json(element)
                                else:
                                    break
                        else:
                            with st.expander(f"One element found", expanded=False):
                                st.json(elements)
                    else:
                        st.caption("No relevant model elements identified.")
                    st.write("---")
                # Diagrams
                if show_diagrams:
                    st.markdown("**🖼️ Associated Diagrams:**")
                    diagram_paths = result_data.get("diagrams", [])
                    if diagram_paths:
                        cols = st.columns(len(diagram_paths))
                        for idx, path in enumerate(diagram_paths):
                            with cols[idx]:
                                if path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".svg")) and os.path.exists(path):
                                    st.image(path, caption=os.path.basename(path), use_container_width=True)
                                else:
                                    st.warning(f"Diagram not found or invalid: {path}")
                    else:
                        st.caption("No associated diagrams found.")

    # Handle user input
    # Example questions as buttons above the input
    # Example questions as buttons
    
    if not st.session_state.get("followup_q"):
        example_questions =  [
            "What packages are in the model?",
            "What is the significance of the APS?",
            "What requirements are related to the APS?",
        ]
    else:
        example_questions = st.session_state["followup_q"]
        
        
    cols = st.columns(len(example_questions))
    q = None
    for col, question in zip(cols, example_questions):
        if col.button(question, key=question):
            q = question

    if prompt := st.chat_input("Ask about the TMT...") or q:

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
        if st.session_state.get("user_question"):
            st.session_state.messages.append({"role": "user", "content": st.session_state.get("user_question")})

        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Thinking...🧠")
            progress_log = st.container()
            settings = {
                "enable_planner_agent": st.session_state.get("enable_planner_agent", False),
                
                "enable_reviewer_agent": st.session_state.get("enable_reviewer_agent", False),
                
                "max_retry_reviewer": st.session_state.get("max_retry_reviewer", 1),

                "max_retry_sysml_filter": st.session_state.get("max_retry_sysml_filter", 8),

                "max_rag": st.session_state.get("max_rag", 8),

            }
            try:
                agent_input = {
                    "messages": [HumanMessage(content=prompt)],
                    "model_query_result": {},
                    "rag_context": {}
                }
                if show_thinking_steps:
                    agent_response = execute_agent_query(agent_input, settings, logger=progress_log)
                else:
                    agent_response = execute_agent_query(agent_input, settings)

                assistant_response_data = {
                    "final_answer": agent_response.get("final_answer", "Error: no answer."),
                    "rag_context": agent_response.get("rag_context", []),
                    "model_query_result": agent_response.get("model_query_result", []),
                    "diagrams": agent_response.get("diagrams", []),
                    "followup_q": agent_response.get("followup_q", None)
                }
                st.session_state.followup_q = agent_response.get("followup_q", None)
                print(f"Followup question: {st.session_state.followup_q}")
                

                placeholder.empty()
                # Display structured response
                st.markdown("**Final Answer:**")
                st.markdown(assistant_response_data["final_answer"])
                st.write("---")
                
                if show_context:
                    st.markdown("**📚 Documentation Context Used:**")
                    for i, chunk in enumerate(assistant_response_data["rag_context"]):
                        title = chunk.get("title", f"Source {i+1}")
                        score = chunk.get("score", 0.0)
                        text = chunk.get("text", "No text available.")
                        with st.expander(f"Context: {title} (Score: {score:.3f})", expanded=False):
                            st.markdown(f"**Source:** `{title}`")
                            st.markdown(f"**Score:** `{score:.3f}`")
                            st.info(text)
                    st.write("---")
                if show_elements:
                    st.markdown("**🧩 Relevant Model Elements:**")
                    for i, element in enumerate(assistant_response_data["model_query_result"]):
                        if isinstance(element, dict):
                            name = element.get("name", f"Element {i+1}")
                            eid = element.get("id", "No ID")
                            with st.expander(f"Element: {name} (ID: {eid})", expanded=False):
                                st.json(element)
                        else:
                            st.warning(f"Unexpected element format: {element}")
                    st.write("---")
                if show_diagrams:
                    st.markdown("**🖼️ Associated Diagrams:**")
                    num = len(assistant_response_data["diagrams"])
                    cols = st.columns(num if num>0 else 1)
                    for idx, path in enumerate(assistant_response_data["diagrams"]):
                        with cols[idx]:
                            if path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".svg")) and os.path.exists(path):
                                st.image(path, use_container_width=True)
                            else:
                                st.warning(f"Not found: {path}")
                st.session_state.messages.append({"role": "assistant", "content": assistant_response_data})
                st.rerun()

            except Exception as e:
                placeholder.empty()
                st.error(f"An error occurred: {e}")
                st.exception(e)
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})