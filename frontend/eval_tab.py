# frontend/eval_tab.py

import streamlit as st
import os
import json
from eval_utils.eval_utils import (
    get_azure_openai_client,
    load_data,
    run_mock_llm_agent,
    calculate_similarity,
    llm_judge_evaluation,
    keyword_match_score,
)

# Load examples.json by default
examples_path = os.path.join(os.path.dirname(__file__), "examples.json")
if os.path.exists(examples_path):
    with open(examples_path, "r") as f:
        default_data = json.load(f)
else:
    default_data = []


def render_eval_tab() -> None:
    st.title("LLM Agent Evaluation")
    

    client, deployment = get_azure_openai_client()
    if not client:
        st.error("Azure OpenAI client not configured. Check secrets.")
        st.stop()

    # Use default_data if no file is uploaded
    uploaded = st.file_uploader("Upload JSON", type="json")
    if uploaded:
        data, err = load_data(uploaded)
    else:
        data, err = default_data, None

    if err:
        st.error(err)
        st.stop()

    st.success(f"Loaded {len(data)} examples.")
    idx = st.selectbox("Question:", range(len(data)),
                    format_func=lambda i: data[i]["question"])
    item = data[idx]


    if st.button("Run Evaluation"):
        settings = {
                "enable_planner_agent": st.session_state.get("enable_planner_agent", False),
                
                "enable_reviewer_agent": st.session_state.get("enable_reviewer_agent", False),
                
                "max_retry_reviewer": st.session_state.get("max_retry_reviewer", 1),

                "max_retry_sysml_filter": st.session_state.get("max_retry_sysml_filter", 8),

                "max_rag": st.session_state.get("max_rag", 8),
            }
        ans = run_mock_llm_agent(item["question"], settings = settings)["answer"]
        sim = calculate_similarity(item["expected_answer"], ans,
                                client, deployment)
        judge = llm_judge_evaluation(item["question"], ans, client)
        kw = keyword_match_score(ans, item.get("keywords", []))

        st.metric("Semantic Similarity", f"{sim:.3f}")
        st.metric("Judge Score", judge["score"])
        st.write("Judge Comment:", judge["comment"])
        st.metric("Keyword Score", kw)
        st.markdown("**Answer:**")
        st.write(ans)

    if st.button("Run All Examples"):
        results = []
        for example in data:
            settings = {
                "enable_planner_agent": st.session_state.get("enable_planner_agent", False),
                
                "enable_reviewer_agent": st.session_state.get("enable_reviewer_agent", False),
                
                "max_retry_reviewer": st.session_state.get("max_retry_reviewer", 1),

                "max_retry_sysml_filter": st.session_state.get("max_retry_sysml_filter", 8),

                "max_rag": st.session_state.get("max_rag", 8),
            }
            ans = run_mock_llm_agent(example["question"], settings=settings)["answer"]
            sim = calculate_similarity(example["expected_answer"], ans,
                                    client, deployment)
            judge = llm_judge_evaluation(example["question"], ans, client)
            kw = keyword_match_score(ans, example.get("keywords", []))

            results.append({
                "question": example["question"],
                "semantic_similarity": sim,
                "judge_score": judge["score"],
                "judge_comment": judge["comment"],
                "keyword_score": kw,
                "answer": ans
            })

        for result in results:
            st.subheader(f"Question: {result['question']}")
            st.metric("Semantic Similarity", f"{result['semantic_similarity']:.3f}")
            st.metric("Judge Score", result["judge_score"])
            st.write("Judge Comment:", result["judge_comment"])
            st.metric("Keyword Score", result["keyword_score"])
            st.markdown("**Answer:**")
            st.write(result["answer"])
