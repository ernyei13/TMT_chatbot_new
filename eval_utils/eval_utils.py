# eval_utils.py

import json
import numpy as np
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from agents.run_graph import execute_agent_query
from langchain_core.messages import HumanMessage
import streamlit as st
import os
from dotenv import load_dotenv
import streamlit as st
from openai import AzureOpenAI
load_dotenv()

def get_azure_openai_client():
    """Initialize AzureOpenAI client using .env variables."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    if not all((endpoint, api_key, embedding_deployment)):
        st.error(
            "Azure OpenAI credentials missing. Please define "
            "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
            "and AZURE_OPENAI_EMBEDDING_DEPLOYMENT in your .env file."
        )
        return None, None

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )
    return client, embedding_deployment


def load_data(uploaded_file):
    """Load and validate JSON list of {question, expected_answer, keywords}."""
    if uploaded_file is None:
        return None, "Please upload a JSON file."
    try:
        data = json.load(uploaded_file)
        if not isinstance(data, list):
            return None, "JSON must be a list of objects."
        for i, itm in enumerate(data):
            if not isinstance(itm, dict):
                return None, f"Item {i} is not a dict."
            if "question" not in itm or "expected_answer" not in itm:
                return None, f"Item {i} missing question or expected_answer."
        return data, None
    except json.JSONDecodeError:
        return None, "Invalid JSON format."


def run_mock_llm_agent(question: str, settings) -> dict:
    """Invoke the agent and return its answer and context."""
    resp = execute_agent_query(
        {"messages": [HumanMessage(question)]},
        logger=st.container(),
        settings=settings,
    )
    return {
        "answer": resp["final_answer"],
        "context": resp.get("rag_context", []),
        "model_query_result": resp.get("model_query_result", []),
    }


def _get_embedding(text, client, deployment):
    if not text:
        return None
    out = client.embeddings.create(model=deployment, input=text)
    if out.data:
        return np.array(out.data[0].embedding)
    return None


def calculate_similarity(text1, text2, client, deployment):
    """Cosine similarity of Azure embeddings."""
    if not (client and deployment and text1 and text2):
        return 0.0
    e1 = _get_embedding(text1, client, deployment)
    e2 = _get_embedding(text2, client, deployment)
    if e1 is None or e2 is None:
        return 0.0
    return float(cosine_similarity(e1.reshape(1, -1), e2.reshape(1, -1))[0][0])


def llm_judge_evaluation(question, answer, client):
    """Structured JSON score/comment from Azure chat model."""
    sys = """
        You are a strict evaluator. Rate from 1–10 on relevance,
        clarity, specificity, informativeness. Reply as JSON:
        {"score":<int>,"comment":"<text>"}
    """
    user = f"Question: {question}\nAnswer: {answer}"
    out = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": sys.strip()},
                  {"role": "user", "content": user}],
        temperature=0,
    )
    try:
        return out.choices[0].message.content and json.loads(
            out.choices[0].message.content
        )
    except Exception:
        return {"score": 0, "comment": "Invalid JSON from judge."}


def keyword_match_score(answer, keywords):
    """Average partial_ratio(keyword, answer) 0–100."""
    if not (answer and keywords):
        return 0
    scores = [fuzz.partial_ratio(answer, kw) for kw in keywords]
    return sum(scores) // len(scores)
