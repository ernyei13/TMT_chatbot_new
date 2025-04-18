# chat_app.py
import streamlit as st
import sys
import os
from typing import List, Dict, Any



# --- Add parent directory to path to find the 'agents' module ---
# Adjust this path if your 'agents' directory is located elsewhere
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..")) # Assuming agents is one level up
# Or if agents is in the same directory:
# parent_dir = current_dir
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    # --- Import the core agent function and message types ---
    from agents.run_graph import execute_agent_query # Make sure this path is correct
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError as e:
    st.error(f"Failed to import necessary modules. Please ensure 'agents/run_graph.py' exists and all dependencies (like langchain_core) are installed. Error: {e}")
    st.stop()

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="TMT Chat Agent", layout="wide")
st.title("TMT Chat Agent")
st.caption("Ask questions about the TMT sysml model and documentation.")

with st.sidebar:
    st.header("View Settings")
    show_thinking_steps = st.checkbox("Show Thinking Steps", value=False, key="show_thinking_steps")
    show_context = st.checkbox("Show Context", value=False, key="show_context")
    show_elements = st.checkbox("Show Elements", value=False, key="show_elements")
    show_diagrams = st.checkbox("Show Diagrams", value=False, key="show_diagrams")
    #st.header("model settings")


# --- Session State Initialization ---
if "messages" not in st.session_state:
    # Start with a system message or welcome message
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you understand the SysML model today?"}
    ]



# --- Display existing chat messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Simple text content for user messages and initial assistant message
        if isinstance(message["content"], str):
            st.markdown(message["content"])
        # Structured content for assistant responses from the agent
        elif isinstance(message["content"], dict):
            result_data = message["content"]

            # 1. Final Answer
            st.markdown("**Final Answer:**")
            st.markdown(result_data.get("final_answer", "No answer generated."))
            st.write("---") # Separator

            if show_context:
                # 2. Context
                st.markdown("**📚 Documentation Context Used:**")
                context = result_data.get("rag_context", [])
                if context and isinstance(context, list):
                    for i, chunk in enumerate(context):
                        title = chunk.get("title", f"Source {i+1}")
                        score = chunk.get("score", 0.0)
                        text = chunk.get("text", "No text available.")
                        with st.expander(f"Context: {title} (Score: {score:.3f})", expanded=False):
                            st.markdown(f"**Source:** `{title}`")
                            st.markdown(f"**Score:** `{score:.3f}`")
                            st.markdown("**Text:**")
                            st.info(text)
                else:
                    st.caption("No specific documentation context was retrieved.")
                st.write("---") # Separator

            if show_elements:
                # 3. Elements
                st.markdown("**🧩 Relevant Model Elements:**")
                elements = result_data.get("model_query_result", [])
                if elements and isinstance(elements, list):
                    for i, element in enumerate(elements):
                        name = element.get("name", f"Element {i+1}")
                        element_id = element.get("id", "No ID")
                        with st.expander(f"Element: {name} (ID: {element_id})", expanded=False):
                            # Display element details as JSON
                            st.json(element)
                else:
                    st.caption("No specific model elements were identified as relevant.")
                st.write("---") # Separator


            if show_diagrams:
                # 4. Diagrams
                st.markdown("**🖼️ Associated Diagrams:**")
                diagram_paths = result_data.get("diagrams", [])
                if diagram_paths and isinstance(diagram_paths, list):
                     # five diagrams per line (Create columns for side-by-side display)
                    cols = st.columns(len(diagram_paths))

                    for idx, path in enumerate(diagram_paths):
                        try:
                            # Basic check if path seems like an image file
                            if path and isinstance(path, str) and path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg')):
                                if os.path.exists(path):
                                    with cols[idx]:
                                        st.image(path, caption=f"Diagram: {os.path.basename(path)}", use_container_width=True)
                                else:
                                    with cols[idx]:
                                        st.warning(f"Diagram file not found at path: `{path}`")
                            else:
                                with cols[idx]:
                                    st.warning(f"Invalid or non-image path provided for diagram: `{path}`")
                        except Exception as e:
                            with cols[idx]:
                                st.error(f"Error loading diagram {path}: {e}")
                else:
                    st.caption("No associated diagrams were found for this query.")


# --- Handle User Input ---

if prompt := st.chat_input("Ask about the SysML model..."):

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display thinking indicator and call the agent
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...🧠")
        progress_log = st.container() # Container to show agent progress logs

        try:
            # Prepare the input for the agent function
            # Wrapping the raw prompt string into the expected input structure
            agent_input = {
                "messages": [HumanMessage(content=prompt)],
                 # You might need to initialize other keys if your agent expects them
                "model_query_result": {},
                "rag_context": {}
            }

            # Call the agent execution function
            # Pass the progress_log container to the logger argument if supported
            if show_thinking_steps:
                agent_response = execute_agent_query(agent_input, logger=progress_log)
            else:
                agent_response = execute_agent_query(agent_input)
            


            # --- Process and Store the Structured Response ---
            # Ensure the response structure is as expected
            assistant_response_data = {
                "final_answer": agent_response.get("final_answer", "Error: Could not extract final answer."),
                "rag_context": agent_response.get("rag_context", []),
                "model_query_result": agent_response.get("model_query_result", []),
                "diagrams": agent_response.get("diagrams", [])
            }
            # Clear "Thinking..." and display the actual response structure
            message_placeholder.empty() # Remove the thinking message

            # Display the structured response (will also be added to history below)
            # 1. Final Answer
            st.markdown("**Final Answer:**")
            st.markdown(assistant_response_data["final_answer"])
            st.write("---")

            # 2. Context
            st.markdown("**📚 Documentation Context Used:**")
            context = assistant_response_data["rag_context"]
            if context:
                for i, chunk in enumerate(context):
                    title = chunk.get("title", f"Source {i+1}")
                    score = chunk.get("score", 0.0)
                    text = chunk.get("text", "No text available.")
                    with st.expander(f"Context: {title} (Score: {score:.3f})", expanded=False):
                        st.markdown(f"**Source:** `{title}`")
                        st.markdown(f"**Score:** `{score:.3f}`")
                        st.markdown("**Text:**")
                        st.info(text)
            else:
                st.caption("No specific documentation context was retrieved.")
            st.write("---")

            # 3. Elements
            st.markdown("**🧩 Relevant Model Elements:**")
            elements = assistant_response_data["model_query_result"]
            if elements:
                for i, element in enumerate(elements):
                    name = element.get("name", f"Element {i+1}")
                    element_id = element.get("id", "No ID")
                    with st.expander(f"Element: {name} (ID: {element_id})", expanded=False):
                        st.json(element)
            else:
                st.caption("No specific model elements were identified as relevant.")
            st.write("---")

            # 4. Diagrams
            st.markdown("**🖼️ Associated Diagrams:**")
            diagram_paths = assistant_response_data["diagrams"]
            if diagram_paths:
                # Adjust layout if many diagrams - maybe fewer columns or scrollable area
                num_diagrams = len(diagram_paths)
                cols_per_row = 3 # Adjust as needed
                rows = (num_diagrams + cols_per_row - 1) // cols_per_row

                for r in range(rows):
                    cols = st.columns(cols_per_row)
                    for c in range(cols_per_row):
                        idx = r * cols_per_row + c
                        if idx < num_diagrams:
                            path = diagram_paths[idx]
                            try:
                                if path and isinstance(path, str) and path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg')):
                                    if os.path.exists(path):
                                        with cols[c]:
                                            st.image(path, caption=f"{os.path.basename(path)}", use_container_width=True)
                                    else:
                                        with cols[c]:
                                            st.warning(f"Not found: `{path}`")
                                else:
                                    with cols[c]:
                                        st.warning(f"Invalid path: `{path}`")
                            except Exception as e:
                                with cols[c]:
                                    st.error(f"Error loading diagram: {e}")
            else:
                st.caption("No associated diagrams were found.")

            # Add the *structured* response to session state
            st.session_state.messages.append({"role": "assistant", "content": assistant_response_data})


        except Exception as e:
            message_placeholder.empty() # Remove thinking message on error
            st.error(f"An error occurred while processing your request: {e}")
            st.exception(e) # Show traceback for debugging in the app
            # Add error message to chat history
            st.session_state.messages.append({"role": "assistant", "content": f"Sorry, an error occurred: {e}"})

