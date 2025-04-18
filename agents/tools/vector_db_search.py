from dotenv import load_dotenv
import os
import requests
import json

# Load environment variables from .env file
load_dotenv()

# Configuration
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "https://ezsearch.search.windows.net")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "vector-tmt-basic")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY", "")
API_VERSION = os.getenv("API_VERSION", "2024-11-01-preview")
TEXT_VECTOR_FIELD = os.getenv("TEXT_VECTOR_FIELD", "text_vector")
SEMANTIC_CONFIG = os.getenv("SEMANTIC_CONFIG", "vector-tmt-basic-semantic-configuration")
LANGUAGE = os.getenv("LANGUAGE", "en-us")


def retrieve_text_from_azure_search(
    search_term: str,
    relevance_threshold: float,
    max_documents: int,
    reranker: bool,
    exclude_table_of_contents: bool
) -> list[dict]:
    """
    Queries Azure AI Search and returns RAG-usable caption chunks
    with metadata including relevance scores.

    Args:
        search_term (str): The search query.
        relevance_threshold (float): Minimum score to keep a result.
        max_documents (int): Maximum number of results to return.
        reranker (bool): Whether to apply reranking (semantic).
        exclude_table_of_contents (bool): If true, filters TOC-like results.

    Returns:
        list[dict]: List of result dictionaries with score, text, and metadata.
    """
    url = (
        f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX}/docs/search"
        f"?api-version={API_VERSION}"
    )

    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_SEARCH_API_KEY
    }

    query_type = "semantic" if reranker else "simple"

    payload = {
        "search": search_term,
        "count": True,
        "vectorQueries": [
            {
                "kind": "text",
                "text": search_term,
                "fields": TEXT_VECTOR_FIELD,
                "queryRewrites": "generative"
            }
        ],
        "queryType": query_type,
        "semanticConfiguration": SEMANTIC_CONFIG,
        "captions": "extractive",
        "answers": "extractive|count-3",
        "queryLanguage": LANGUAGE,
        "queryRewrites": "generative"
    }

    response = requests.post(url, headers=headers, json=payload, timeout=10)
    if not response.ok:
        print("⚠️ Azure Search Error:", response.status_code, response.text)
    response.raise_for_status()

    data = response.json()
    results = []

    for result in data.get("value", []):
        score = result.get("@search.score", 0.0)
        text = result.get("@search.captions", [{}])[0].get("text", "")
        key = result.get("key", "")
        highlights = result.get("highlights", "")
        title = result.get("title", "")
        chunk_id = result.get("chunk_id", "")

        if score < relevance_threshold:
            continue

        if exclude_table_of_contents and _looks_like_table_of_contents(text):
            continue

        results.append({
            "text": text,
            "score": score,
            "key": key,
            "highlights": highlights,
            "title": title,
            "chunk_id": chunk_id
        })

        if len(results) >= max_documents:
            break

    return results


def _looks_like_table_of_contents(text: str) -> bool:
    """
    Heuristic to detect if a text looks like a table of contents.
    """
    has_many_dots = text.count(".") > 10
    has_many_numbers = sum(c.isdigit() for c in text) > 10
    return has_many_dots and has_many_numbers


results = retrieve_text_from_azure_search(
        search_term="test",
        relevance_threshold=0.5,
        max_documents=2,
        reranker=True,
        exclude_table_of_contents=True
    )
print(json.dumps(results, indent=2))