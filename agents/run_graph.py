from typing import Callable, Dict, Any, List, TypedDict, Annotated, Sequence
from hashlib import md5
from datetime import timedelta
import json

import streamlit as st
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END, add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda

from agents.rag_agent import make_rag_agent
from agents.graph_agent import make_query_builder_agent
from agents.tools.vector_db_search import retrieve_text_from_azure_search
from agents.tools.diagram_retriever import diagram_retriever
from agents.summarizer_agent import Summarizer
from agents.reviewer_agent import make_reviewer_agent
from agents.question_agent import make_followup_agent
from loaders.json_loader import load_elements


@st.cache_resource(ttl=timedelta(hours=1))
def get_elements() -> Dict[str, Any]:
    """Load the full SysML element tree (≈ hundreds of MB) exactly once."""
    return load_elements()

@st.cache_resource(show_spinner=False)
def get_langgraph(settings: dict):
    elements = get_elements()          # heavy JSON is already cached
    return build_sysml_langgraph_agent(elements,
                                       retrieve_text_from_azure_search,
                                       settings)

def _settings_hash(settings: dict) -> str:
    """Deterministic hash so a new graph is compiled only when
    the settings actually change."""
    blob = json.dumps(settings, sort_keys=True).encode()
    return md5(blob).hexdigest()

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    rag_context: Dict[str, Any]
    question_for_rag: str
    question_for_model: str
    model_query_result: Dict[str, Any]
    rag_agent_result: str
    final_answer: str
    diagrams: List[str]
    complete: bool
    retry_count: int
    followup_q: List[str]


# ----------- BUILD GRAPH -----------
def build_sysml_langgraph_agent(elements: Dict[str, Any], retriever_fn: Callable, settings: dict):

    #set the settings from the frontend
    sysml_retry_num = settings.get("max_retry_sysml_filter", 8)

    sysml_query_agent = make_query_builder_agent(elements, sysml_retry_num)

    rag_max_documents = settings.get("max_rag", 5)
    print(f"[RAG] max documents: {rag_max_documents}")
    
    rag_agent = make_rag_agent(
        lambda q, kw: retrieve_text_from_azure_search(q, kw, 0, rag_max_documents, True, True) 
    )

    # Create agents
    reviewer_agent = make_reviewer_agent(max_reviews=settings.get("max_retry_reviewer", 1))
    #planner_agent = make_planner_agent()
    followup = make_followup_agent()
    diagram_retriever_node = RunnableLambda(diagram_retriever)
    summarizer = Summarizer()
    summarizer_runnable = RunnableLambda(summarizer)

    # Create the graph and link the nodes
    graph = StateGraph(AgentState)

    # define router fun
    def review_decision(state: Dict[str, Any]) -> str:
        if state.get("complete", False):
            return "final"
        return state.get("call")
    
    # define entry node
    def set_retry_count(state: Dict[str, Any]) -> Dict[str, Any]:
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                q = msg.content
                break
        return {
            **state,
            "retry_count": 0,
            "question": q
        }

    #define the nodes of the graph 
    
    graph.add_node("set_retry_count", set_retry_count)
    graph.set_entry_point("set_retry_count")

    graph.add_node("sysml_query_agent", sysml_query_agent)
    graph.add_node("rag_agent", rag_agent)
    graph.add_node("diagram_retriever", diagram_retriever_node)
    graph.add_node("summarizer", summarizer_runnable)
    graph.add_node("followup", followup)

    #define the edges
    graph.add_edge("set_retry_count", "rag_agent")
    graph.add_edge("rag_agent", "sysml_query_agent")
    graph.add_edge("sysml_query_agent", "diagram_retriever")
    graph.add_edge("diagram_retriever", "summarizer")

    #construct based on the settings
    if settings.get("enable_reviewer_agent", False):
        graph.add_node("reviewer", reviewer_agent)
        graph.add_edge("summarizer", "reviewer")
        graph.add_conditional_edges(
            "reviewer",
            review_decision,
            {
                "final": "followup",
                "sysml_query_agent": "sysml_query_agent",  # rerun query-builder
                "rag_agent": "rag_agent",                  # rerun RAG search
            },
        )
    else:
        graph.add_edge("summarizer", "followup")
    
    graph.add_edge("followup", END)

    return graph.compile()

# query execution
def execute_agent_query(user_input: str,  settings: dict, logger=None) -> dict:
    # Create a container to show progress
    progress_area = logger.container() if logger else None

    def log(msg, emoji="🔹"):
        if progress_area:
            progress_area.markdown(f"{emoji} {msg}")

    # Step 1: Load model elements
    try:
        log("Loading SysML model elements...")
        elements_local = load_elements()
    except Exception as e:
        log(f"Failed to load model elements: {e}")
        return {
            "final_answer": AIMessage("Error loading model elements."),
            "rag_context": [],
            "model_query_result": []
        }

    # Step 2: Build the workflow based on the settings
    graph = get_langgraph(settings)
    log(f"LangGraph workflow ready. Settings: {settings}")

    # Step 3: Execute the query
    log("Running agent...")
    config = {"configurable": {"thread_id": "1"}}

    final_state = {}
    print(f"Settings: {settings}")

    for event in graph.stream(input=user_input, config=config, stream_mode="updates"):
        print("------------------------")
        print(f"Event type: {list(event.keys())[0]}")
        if isinstance(event, dict):
            node_name = list(event.keys())[0]  # e.g. 'rag_agent', 'summarizer', etc.
            node_output = event[node_name]

        if node_name == "followup":
            log(f"✅ Follow‑up questions generated by `{node_name}`")
            final_state = node_output
            break #stop if final message gatherred
        else:
            log(f"Intermediate state from `{node_name}`")
            log(f"💬 {node_output['messages'][-1].content}")
            if node_name == "rag_agent":
                log(f"🔍 Found {len(node_output['rag_context'])} relevant chunks")
            if node_name == "sysml_query_agent":
                log(f"🔍 Found {len(node_output['model_query_result'])} relevant model elements")
            if node_name == "diagram_retriever":
                log(f"📊 Found {len(node_output['diagrams'])} diagrams")
            if node_name == "summarizer":
                log(f"📜 Summarizer input tokens")
            if node_name == "reviewer":
                log(f"🔄 Reviewer input tokens")

    log("✅ Workflow completed.")

    return final_state
