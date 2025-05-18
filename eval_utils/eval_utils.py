"""
eval_utils.py

Provides three evaluation functions:
- compute_vector_similarity: cosine similarity between embeddings
- compute_llm_judge: LLM-based correctness/completeness/clarity scoring
- compute_keyword_count: count of domain keywords in response
"""

import os
import json
import numpy as np
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
import csv
import re


import pandas as pd
from agents.run_graph import execute_agent_query


load_dotenv()

def _get_embedding(text: str,
                   client: AzureOpenAI,
                   deployment: str) -> np.ndarray | None:
    """Return embedding vector for a given text, or None on failure."""
    if not isinstance(text, str) or not text.strip():
        return None

    try:
        resp = client.embeddings.create(
            model=deployment,
            input=[text],
        )
        emb = resp.data[0].embedding
        return np.array(emb, dtype=float) if emb else None
    except Exception as e:
        raise RuntimeError(f"Embedding error for '{text[:30]}...': {e}")


def compute_vector_similarity(reference: str,
                              response: str,
                              client: AzureOpenAI,
                              deployment: str) -> float:
    """
    Compute cosine similarity between the embeddings of reference and response.
    Returns a float in [-1.0, 1.0], or 0.0 if embeddings cannot be computed.
    """
    e_ref = _get_embedding(reference, client, deployment)
    e_res = _get_embedding(response, client, deployment)
    if e_ref is None or e_res is None:
        return 0.0
    # reshape for sklearn
    return float(cosine_similarity(e_ref.reshape(1, -1),
                                   e_res.reshape(1, -1))[0][0])


def compute_llm_judge(question: str,
                      response: str,
                      reference: str,
                      keywords: list[str],
                      client: AzureOpenAI) -> dict:
    """
    Use an Azure OpenAI chat model to judge the response.
    Returns a dict: {"score": int, "comment": str}.
    Requires env-var AZURE_OPENAI_CHAT_DEPLOYMENT.
    """
    system_prompt = (
         "You are a strict evaluator. Given a question, a reference answer, the assistant's response, and a list of expected keywords, rate from 1–100 on correctness, completeness, clarity, and keyword coverage. The expected keywords and answer is provided if missing provide a score without them. If the length of the expected answer and the assistant answer differs signifficantly check if the long answer contains all the information from the short one. Do not decrease the score based on the length."
         " Reply as JSON: "
        '{"score":<int>,"comment":"<text>"} and nothing else.'
    )
    chat_model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
    if not chat_model:
        raise RuntimeError("AZURE_OPENAI_CHAT_DEPLOYMENT not set")



    try:
        payload = {
                "question": question,
                "reference": reference,
                "response": response,
                "keywords": keywords,
            }

        out = client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(payload)
                },
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = out.choices[0].message.content
        if isinstance(content, str):
            return json.loads(content)
        return content  # already a dict
    
    except Exception as e:
        raise RuntimeError(f"LLM judge error: {e}")



def compute_keyword_count(response: str, keywords: list[str]) -> int:
    """
    Count how many of the specified keywords appear in the response.
    Case-insensitive full word or phrase match.
    """
    if not isinstance(response, str) or not keywords:
        return 0

    resp_lower = " ".join(response.lower().split())
    count = 0
    for kw in set(keywords):
        if isinstance(kw, str):
            pattern = r'\b' + re.escape(kw.lower()) + r'\b'
            if re.search(pattern, resp_lower):
                count += 1
    return count



def get_azure_openai_client():
    """
    Initialize and return an AzureOpenAI client and embedding deployment name.
    Requires AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT in environment or .env.
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15")
    embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    if not all((endpoint, api_key, embedding_deployment)):
        st.error(
            "Azure OpenAI credentials missing. Please set "
            "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
            "and AZURE_OPENAI_EMBEDDING_DEPLOYMENT."
        )
        return None, None

    try:
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        return client, embedding_deployment
    except Exception as exc:
        st.error(f"Failed to initialize Azure OpenAI client: {exc}")
        return None, None


def load_data(uploaded_file):
    """
    Load and validate JSON list of {question, expected_answer, keywords}.
    Returns (data_list, error_message) where error_message is None on success.
    """
    if uploaded_file is None:
        return None, "Please upload a JSON file."

    try:
        raw = uploaded_file.getvalue().decode("utf-8")
        data = json.loads(raw)
    except Exception:
        return None, "Invalid JSON format."

    if not isinstance(data, list):
        return None, "JSON must be an array of objects."

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            return None, f"Item {idx} is not an object."
        if "question" not in item or "expected_answer" not in item:
            return None, f"Item {idx} missing 'question' or 'expected_answer'."
        if "keywords" in item and not isinstance(item["keywords"], list):
            st.warning(
                f"Item {idx} 'keywords' is not a list, resetting to empty."
            )
            item["keywords"] = []

    return data, None


def run_batch_evaluation(data: list, client, deployment, settings: dict, output_csv_path: str) -> None:
    results = []

    for idx, item in enumerate(data):
        question = item["question"]
        expected_answer = item["expected_answer"]
        keywords = item.get("keywords", [])

        agent_input = {
            "messages": [HumanMessage(content=question)],
            "model_query_result": {},
            "rag_context": {}
        }
        try:
            result = execute_agent_query(agent_input, settings, logger=None)
            answer = result.get("final_answer", "")
        except Exception as e:
            answer = ""
            print(f"Error running query on item {idx}: {e}")

        try:
            sim = compute_vector_similarity(expected_answer, answer, client, deployment)
        except Exception:
            sim = -1.0

        try:
            judge = compute_llm_judge(question, answer, expected_answer, keywords, client)
            judge_score = judge["score"]
            judge_comment = judge["comment"]
        except Exception:
            judge_score = -1
            judge_comment = "error"

        try:
            kw_count = compute_keyword_count(answer, keywords)
            kw_coverage = (kw_count / len(keywords)) * 100 if keywords else 0
        except Exception:
            kw_count = -1
            kw_coverage = 0

        results.append({
            "id": item.get("id", idx),
            "question": question,
            "similarity": sim,
            "judge_score": judge_score,
            "judge_comment": judge_comment,
            "keyword_count": kw_count,
            "keyword_coverage": f"{kw_coverage:.0f}%",
        })

        row = {
            "id": item.get("id", idx),
            "question": question,
            "similarity": sim,
            "judge_score": judge_score,
            "judge_comment": judge_comment,
            "keyword_count": kw_count,
            "keyword_coverage": f"{kw_coverage:.0f}%",
            "result": result.get("final_answer", ""),
        }

        header = [
            "id", "question", "similarity",
            "judge_score", "judge_comment",
            "keyword_count", "keyword_coverage", "result",
        ]
                # Append result to CSV
        with open(output_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writerow(row)
