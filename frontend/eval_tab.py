# frontend/eval_tab.py

import json
import os

import streamlit as st

from agents.run_graph import execute_agent_query
from eval_utils.eval_utils import get_azure_openai_client, load_data
from eval_utils.eval_utils import (
    compute_vector_similarity,
    compute_llm_judge,
    compute_keyword_count,
)
from langchain_core.messages import HumanMessage


def render_eval_tab() -> None:
    st.title("LLM Agent Evaluation")

    client, deployment = get_azure_openai_client()
    if not client:
        st.error("Azure OpenAI client not configured. Check secrets.")
        st.stop()

    examples_path = os.path.join(os.path.dirname(__file__), "examples_2.json")
    default_data = []
    if os.path.exists(examples_path):
        with open(examples_path, "r") as f:
            default_data = json.load(f)

    uploaded = st.file_uploader("Upload JSON", type="json")
    if uploaded:
        data, err = load_data(uploaded)
    else:
        data, err = default_data, None

    if err:
        st.error(err)
        st.stop()

    st.success(f"Loaded {len(data)} examples.")
    idx = st.selectbox(
        "Question:",
        range(len(data)),
        format_func=lambda i: data[i]["question"],
    )
    item = data[idx]

    if st.button("Run Evaluation"):
        settings = {
            "enable_planner_agent": st.session_state.get(
                "enable_planner_agent", False
            ),
            "enable_reviewer_agent": st.session_state.get(
                "enable_reviewer_agent", False
            ),
            "max_retry_reviewer": st.session_state.get(
                "max_retry_reviewer", 1
            ),
            "max_retry_sysml_filter": st.session_state.get(
                "max_retry_sysml_filter", 8
            ),
            "max_rag": st.session_state.get("max_rag", 8),
        }
        progress_log = st.container()

        # run the agent
        agent_input = {
                    "messages": [HumanMessage(content=item["question"])],
                    "model_query_result": {},
                    "rag_context": {}
                }
        ans = execute_agent_query(agent_input, settings, logger = progress_log)

        ans = ans.get("final_answer", "")
        # compute evaluation metrics
        sim = compute_vector_similarity(
            item["expected_answer"], ans, client, deployment
        )
        print(f"Similarity: {sim}")
        st.metric("Semantic Similarity", f"{sim:.3f}")
        judge = compute_llm_judge(item["question"], ans, item["expected_answer"], item.get("keywords", []), client)
        print(f"Judge: {judge}")
        st.metric("Judge Score", judge["score"])
        st.write("Judge Comment:", judge["comment"])
        
        kw = compute_keyword_count(ans, item.get("keywords", []))
        print(f"Keyword Count: {kw}")
        st.metric("Keyword Count", kw)

        keywords = item.get("keywords", [])
        coverage = (compute_keyword_count(ans, keywords) / len(keywords) * 100) if keywords else 0
        st.metric("Keyword Coverage", f"{coverage:.0f}%")
        
        st.markdown("**Answer:**")
        st.write(ans)
