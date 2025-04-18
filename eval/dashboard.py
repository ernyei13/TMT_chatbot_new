import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import json
import os
import numpy as np
import pandas as pd
from openai import AzureOpenAI # Use the Azure-specific client
from sklearn.metrics.pairwise import cosine_similarity
import time # For potential retries or delays
from agents.run_graph import execute_agent_query
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from fuzzywuzzy import fuzz

# --- Azure OpenAI Configuration & Client Initialization ---

def get_azure_openai_client():
    """Initializes and returns the AzureOpenAI client using Streamlit secrets."""
    try:
        # Check if secrets are loaded
        required_secrets = [
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_KEY",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
        ]
        if not all(secret in st.secrets for secret in required_secrets):
            st.error("Azure OpenAI credentials not found in Streamlit secrets.")
            st.info("Please create a `.streamlit/secrets.toml` file with your Azure Endpoint, API Key, and Embedding Deployment Name.")
            return None, None # Return None for client and deployment name

        endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"]
        api_key = st.secrets["AZURE_OPENAI_KEY"]
        deployment_name = st.secrets["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]

        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-02-15-preview"  # Use an appropriate API version
        )
        return client, deployment_name
    except Exception as e:
        st.error(f"Failed to initialize Azure OpenAI client: {e}")
        return None, None

# Attempt to initialize client globally but handle potential errors
azure_client, azure_embedding_deployment = get_azure_openai_client()

# --- Core Logic Functions ---

def load_data(uploaded_file):
    """Loads and validates data from the uploaded JSON file. (Unchanged)"""
    if uploaded_file is None:
        return None, "Please upload a JSON file."
    try:
        data = json.load(uploaded_file)
        if not isinstance(data, list):
            return None, "Error: JSON file should contain a list of objects."
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                 return None, f"Error: Item at index {i} is not a dictionary."
            if "question" not in item or "expected_answer" not in item:
                return None, f"Error: Item at index {i} is missing 'question' or 'expected_answer'."
            item.setdefault("id", f"item_{i+1}")
            item.setdefault("category", "Uncategorized")
        return data, None
    except json.JSONDecodeError:
        return None, "Error: Invalid JSON format in the uploaded file."
    except Exception as e:
        return None, f"An unexpected error occurred while loading the file: {e}"

def run_mock_llm_agent(question: str) -> dict:
    # Example user input
    user_input = {
        "messages": [HumanMessage(question)],
        "model_query_result": {},  # Initialize as an empty dictionary
        "rag_context": {}  # Initialize as an empty dictionary
    }

    # Execute the query and get the response
    progress_log = st.container()
    response = execute_agent_query(user_input, logger=progress_log) 
    
    return {
        "answer": f"{response['final_answer']}",
        "context": response['rag_context'],
        "model_query_result": response['model_query_result'],
    }
    

def get_azure_embedding(text: str, client: AzureOpenAI, deployment_name: str) -> np.ndarray | None:
    """Gets embedding for a text using Azure OpenAI."""
    if not text: # Handle empty strings
        return None
    try:
        response = client.embeddings.create(
            model=deployment_name,
            input=text
        )
        # Assuming the response structure gives a list of embeddings, take the first
        if response.data and len(response.data) > 0:
             embedding = response.data[0].embedding
             return np.array(embedding)
        else:
            st.warning(f"Azure API did not return embedding data for: '{text[:50]}...'")
            return None
    except Exception as e:
        st.error(f"Azure API call failed for embedding: {e}")
        # Consider adding retries with backoff here for production robustness
        # time.sleep(1) # Simple delay example
        return None

def calculate_similarity(text1: str, text2: str, client: AzureOpenAI, deployment_name: str) -> float:
    """Calculates cosine similarity between two texts using Azure embeddings."""
    if not client or not deployment_name:
         st.error("Azure client not initialized. Cannot calculate similarity.")
         return 0.0

    if not text1 or not text2:
        return 0.0 # Handle empty strings

    try:
        # Get embeddings concurrently? For simplicity, doing sequentially here.
        embedding1 = get_azure_embedding(text1, client, deployment_name)
        embedding2 = get_azure_embedding(text2, client, deployment_name)

        if embedding1 is not None and embedding2 is not None:
            # Reshape for cosine_similarity function (expects 2D arrays)
            embedding1 = embedding1.reshape(1, -1)
            embedding2 = embedding2.reshape(1, -1)

            # Compute cosine-similarity
            score = cosine_similarity(embedding1, embedding2)[0][0]
            return float(score) # Ensure it's a standard float
        else:
            st.warning("Could not retrieve embeddings for one or both texts. Similarity set to 0.")
            return 0.0
    except Exception as e:
        st.error(f"Error calculating similarity with Azure embeddings: {e}")
        return 0.0 # Return a default value on error
    
def llm_judge_evaluation(question: str, agent_answer: str, client: AzureOpenAI) -> dict:
    """Evaluates the agent's response using Azure OpenAI chat model with structured output."""
    try:
        system_prompt = """
            You are a strict evaluator reviewing answers from an AI assistant.

            Your task is to judge how well the assistant's answer responds to the user query.

            Evaluate on:
            1. Relevance
            2. Clarity
            3. Specificity
            4. Informativeness

            Rate from 1 (poor) to 10 (excellent) and explain briefly.

            Respond in **this exact JSON format**:
            {
            "score": 1-10,
            "comment": "your short explanation here"
            }
        """ 

        user_prompt = f"""Question: {question}\nAnswer: {agent_answer}\n"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # replace with your Azure deployment name if needed
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()}
            ],
            temperature=0
        )

        content = response.choices[0].message.content.strip()
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            st.warning("⚠️ LLM returned invalid JSON. Raw output:")
            st.code(content)
            return {"score": 0, "comment": "Invalid JSON from LLM."}

        return result

    except Exception as e:
        st.warning(f"LLM judge evaluation failed: {e}")
        return {"score": 0, "comment": "Evaluation failed."}


def keyword_match_score(answer: str, keywords: list[str]) -> int:
    """Calculates keyword-based partial match score (0–100)."""
    if not answer or not keywords:
        return 0
    scores = [fuzz.partial_ratio(answer.lower(), kw.lower()) for kw in keywords]
    return sum(scores) // len(scores)
# --- Streamlit App UI (Largely Unchanged) ---

st.set_page_config(layout="wide")
st.title("LLM Agent Evaluation")

# Check if Azure client is ready before proceeding
if not azure_client or not azure_embedding_deployment:
    st.warning("Azure OpenAI client is not configured. Please check your secrets configuration.")
    st.stop() # Stop execution if client isn't ready

# --- Session State Initialization ---
if 'evaluation_data' not in st.session_state:
    st.session_state.evaluation_data = None
if 'selected_question_index' not in st.session_state:
    st.session_state.selected_question_index = 0
if 'last_run_result' not in st.session_state:
    st.session_state.last_run_result = None
if 'all_run_results' not in st.session_state:
    st.session_state.all_run_results = []

# --- Sidebar for Upload and Settings ---
with st.sidebar:
    st.header("Configuration (optional)")
    uploaded_file = st.file_uploader("Upload a custom JSON file", type=["json"])

# 👇 Auto-load examples.json if not already loaded
if st.session_state.evaluation_data is None:
    try:
        default_path = "/Users/zoltanernyei/Documents/BME/szakdoga/clean_code_python/eval/examples.json"
        with open(default_path, "r") as f:
            data = json.load(f)
        st.session_state.evaluation_data = data
        st.session_state.uploaded_file_key = "default_examples"
        st.success("Auto-loaded examples.json")
    except Exception as e:
        st.error(f"Failed to load default examples.json: {e}")
        st.stop()

# 👇 Handle optional override via file upload
if uploaded_file:
    try:
        uploaded_data = json.load(uploaded_file)
        st.session_state.evaluation_data = uploaded_data
        st.session_state.uploaded_file_key = uploaded_file.name
        st.success(f"Loaded {len(uploaded_data)} items from {uploaded_file.name}")
    except Exception as e:
        st.error(f"Failed to parse uploaded file: {e}")


# --- Main Evaluation Area ---
if st.session_state.evaluation_data:
    data = st.session_state.evaluation_data
    question_list = [item.get("question", f"Question {i+1}") for i, item in enumerate(data)]

    st.header("Select Question & Evaluate")

    selected_question_index = st.selectbox(
        "Choose a question to evaluate:",
        options=range(len(question_list)),
        format_func=lambda index: question_list[index],
        index=st.session_state.selected_question_index
    )

    st.session_state.selected_question_index = selected_question_index

    selected_item = data[selected_question_index]
    question = selected_item["question"]
    expected_answer = selected_item["expected_answer"]
    keywords = selected_item.get("keywords", [])
    item_id = selected_item.get("id", f"item_{selected_question_index+1}")
    item_category = selected_item.get("category", "Uncategorized")

    st.write("---")
    st.subheader("Ground Truth")
    st.markdown(f"**ID:** `{item_id}`")
    st.markdown(f"**Category:** `{item_category}`")
    st.markdown(f"**Question:**")
    st.info(question)
    st.markdown(f"**Expected Answer:**")
    st.success(expected_answer)
    st.markdown(f"**Keywords:**")
    st.info(", ".join(keywords) if keywords else "None")



    # --- Run Agent and Display Results ---
    if st.button("Run Agent & Evaluate Similarity", key=f"run_{selected_question_index}"):
        with st.spinner("Running LLM Agent and calling Azure for embeddings..."):
            # 1. Run the (mock) LLM Agent
            agent_output = run_mock_llm_agent(question)
            llm_answer = agent_output["answer"]
            llm_context = agent_output["context"]
            llm_elements = agent_output["model_query_result"]

            # 2. Calculate Similarity using Azure
            similarity_score = calculate_similarity(
                expected_answer,
                llm_answer,
                azure_client,          # Pass the initialized client
                azure_embedding_deployment # Pass the deployment name
            )

            judge_result = llm_judge_evaluation(
                question, llm_answer, azure_client
            )


            keywords = expected_answer.split(",")
            keyword_score = keyword_match_score(llm_answer, keywords)

            # Store results
            st.session_state.last_run_result = {
                "id": item_id,
                "category": item_category,
                "question": question,
                "expected_answer": expected_answer,
                "llm_answer": llm_answer,
                "context": llm_context,
                "relevant_elements": llm_elements,
                "similarity_score": similarity_score,
                "judge_result": judge_result,
                "keyword_score": keyword_score
            }
            # Update batch results
            found = False
            for i, res in enumerate(st.session_state.all_run_results):
                if res["id"] == item_id:
                    st.session_state.all_run_results[i] = st.session_state.last_run_result
                    found = True
                    break
            if not found:
                 st.session_state.all_run_results.append(st.session_state.last_run_result)

    # Display results (visuals unchanged)
    if st.session_state.last_run_result and st.session_state.last_run_result["question"] == question:
        st.write("---")
        st.subheader("LLM Agent Output & Evaluation")
        result = st.session_state.last_run_result

        score = result["similarity_score"]
        if score >= 0.75: delta_color, emoji, help_text = "normal", "✅", "High similarity"
        elif score >= 0.5: delta_color, emoji, help_text = "off", "🤔", "Moderate similarity"
        else: delta_color, emoji, help_text = "inverse", "❌", "Low similarity"

        st.metric(
            label=f"Semantic Similarity {emoji}", value=f"{score:.3f}",
            delta_color=delta_color, help=help_text
        )

        # LLM Judge Score
        st.metric(label="LLM Judge Score (1–10)", value=result.get("llm_judge_score", 0))
        st.markdown(f"**LLM Judge Comment:** {result.get('llm_judge_comment', '')}")

        # Keyword Score
        st.metric(label="Keyword Match Score", value=result.get("keyword_score", 0))


        st.markdown(f"**Expected answer:**")
        st.success(result["expected_answer"])
        
        st.markdown("**Generated Answer:**")
        st.warning(result["llm_answer"])
        st.markdown("**Documentation Context:**")
        if "context" in result and isinstance(result["context"], list):
            for i, chunk in enumerate(result["context"]):
                title = chunk.get("title", "Unknown Source")
                score = chunk.get("score", 0.0)
                with st.expander(f"{title} | Score: {score:.3f}"):
                    st.markdown(f"**Source:** `{title}`")
                    st.markdown(f"**Score:** `{score:.3f}`")
                    st.markdown("**Text:**")
                    st.info(chunk.get("text", "No text available."))

        st.markdown("**Relevant Elements:**")
        for i, element in enumerate(result["relevant_elements"]):
            name = element.get("name", f"Element {i+1}")
            element_id = element.get("id", "No ID")
            with st.expander(f"{name} (ID: {element_id})", expanded=False):
                st.json(element)

    st.write("---")

    # --- Stretch Goal 1: Downloadable CSV Report (Unchanged logic) ---
    st.header("Batch Results & Reporting")
    if st.session_state.all_run_results:
        st.write(f"Stored results for {len(st.session_state.all_run_results)} evaluations.")
        df_results = pd.DataFrame(st.session_state.all_run_results)
        report_cols = [
            'id', 'category', 'question', 'expected_answer', 'llm_answer',
            'similarity_score', 'llm_judge_score', 'llm_judge_comment', 'keyword_score'
        ]
        df_display = df_results.reindex(columns=report_cols, fill_value='') # Use reindex to handle potentially missing columns safely
        st.dataframe(df_display)
        csv_data = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
           label="Download Results as CSV", data=csv_data,
           file_name='llm_evaluation_report_azure.csv', mime='text/csv',
        )
    else:
        st.info("Run evaluations to generate a report.")

    # --- Stretch Goal 2: Batch Evaluation (Updated to use Azure) ---
    if st.button("Run Batch Evaluation (All Questions)"):
        st.session_state.all_run_results = []
        progress_bar = st.progress(0)
        total_items = len(data)
        results_list = []
        batch_errors = 0

        for i, item in enumerate(data):
            q = item["question"]
            exp_ans = item["expected_answer"]
            item_id = item.get("id", f"item_{i+1}")
            item_cat = item.get("category", "Uncategorized")

            # Run agent
            agent_output = run_mock_llm_agent(q)
            # Calculate similarity using Azure
            sim_score = calculate_similarity(
                exp_ans,
                agent_output["answer"],
                azure_client,
                azure_embedding_deployment
            )
            # Check if similarity calculation failed (returned 0.0 due to error)
            if sim_score == 0.0 and (exp_ans and agent_output["answer"]):
                 # Check if the score is 0 because of an actual error or just low similarity
                 # A more robust check might involve inspecting logs/errors captured during calculate_similarity
                 # For simplicity, we'll count it if score is 0 and texts weren't empty
                 # This isn't perfect, as 0.0 could be a valid low score.
                 # Consider adding specific error flags from calculate_similarity if needed.
                 # batch_errors += 1 # Optional: track errors
                 pass # Decide how to handle specific errors

            results_list.append({
                "id": item_id, "category": item_cat, "question": q,
                "expected_answer": exp_ans, "llm_answer": agent_output["answer"],
                "similarity_score": sim_score, "llm_context": agent_output["context"],
                "llm_elements": agent_output["model_elements"]
            })
            progress_bar.progress((i + 1) / total_items)

        st.session_state.all_run_results = results_list
        st.success(f"Batch evaluation completed for {total_items} items.")
        # if batch_errors > 0:
        #      st.warning(f"{batch_errors} items encountered errors during similarity calculation.")
        st.rerun()

    # Display Average Scores (Unchanged logic)
    if st.session_state.all_run_results:
        df_batch = pd.DataFrame(st.session_state.all_run_results)
        if not df_batch.empty and 'similarity_score' in df_batch.columns:
            st.subheader("Average Similarity Scores")
            # Exclude potential error results (e.g., where score is 0 due to API failure)
            # This is a simple filter; adjust if 0.0 is a possible valid score.
            valid_scores = df_batch[df_batch['similarity_score'] != 0.0]['similarity_score']
            if not valid_scores.empty:
                 overall_avg = valid_scores.mean()
                 st.metric("Overall Average Similarity (valid scores)", f"{overall_avg:.3f}")

                 if 'category' in df_batch.columns and df_batch['category'].nunique() > 1:
                    # Calculate category average only on valid scores
                    category_avg = df_batch[df_batch['similarity_score'] != 0.0].groupby('category')['similarity_score'].mean()
                    st.write("**Average Similarity per Category (valid scores):**")
                    st.dataframe(category_avg)
            else:
                 st.warning("No valid similarity scores found to calculate averages.")


else:
    # Initial message before file upload or if Azure client failed
    if azure_client:
        st.info("Please upload a JSON file using the sidebar to begin.")
    # Error message about missing secrets is handled where the client is initialized