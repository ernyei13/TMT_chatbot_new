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
    keywords: str,
    relevance_threshold: float,
    max_documents: int,
    reranker: bool,
    exclude_table_of_contents: bool
) -> dict: # <-- Changed return type hint to dict
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
        dict: Dictionary of result dictionaries with chunk_id as key,
              and score, text, and metadata as value.
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

    combined_search = f"{search_term} {' '.join(keywords)}".strip()


    payload = {
        "search": combined_search,
        "count": True,
        "vectorQueries": [
            {
                "kind": "text",
                "text": combined_search,
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
    
    # Change: Initialize results as a dictionary
    results_dict = {} 
    
    # Counter for max_documents
    documents_added = 0

    for result in data.get("value", []):
        score = result.get("@search.score", 0.0)
        text = result.get("@search.captions", [{}])[0].get("text", "")
        key = result.get("key", "")
        highlights = result.get("highlights", "")
        title = result.get("title", "")
        chunk_id = result.get("chunk_id", "")

        # Crucial check: ensure chunk_id exists and is unique for keys
        if not chunk_id:
            # You might want to generate a unique ID or skip this result
            # if chunk_id is not guaranteed to be present or unique.
            # For simplicity, we'll use a fallback if chunk_id is missing/empty.
            # A more robust solution might hash the text or use a counter.
            chunk_id = f"no_id_{len(results_dict) + 1}" 


        if score < relevance_threshold:
            continue

        if exclude_table_of_contents and _looks_like_table_of_contents(text):
            continue

        # Add to dictionary using chunk_id as key
        results_dict[chunk_id] = {
            "text": text,
            "score": score,
            "key": key,
            "highlights": highlights,
            "title": title,
            # "chunk_id": chunk_id # No need to repeat chunk_id inside the value if it's the key
        }
        
        documents_added += 1

        if documents_added >= max_documents:
            break

    return results_dict # <-- Return the dictionary

def _looks_like_table_of_contents(text: str) -> bool:
    """
    Heuristic to detect if a text looks like a table of contents.
    """
    has_many_dots = text.count(".") > 10
    has_many_numbers = sum(c.isdigit() for c in text) > 10
    return has_many_dots and has_many_numbers