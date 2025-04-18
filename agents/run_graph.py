from agents.rag_agent import make_rag_agent
from agents.graph_agent import make_query_builder_agent
from agents.tools.vector_db_search import retrieve_text_from_azure_search
from agents.tools.diagram_retriever import diagram_retriever
from agents.summarizer_agent import Summarizer
from agents.reviewer_agent import make_reviewer_agent
from agents.planner_agent import make_planner_agent
from loaders.json_loader import load_elements

from typing import Callable, Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
import json
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()

# ----------- BUILD STATE GRAPH -----------
def build_sysml_langgraph_agent(elements: Dict[str, Any], retriever_fn: Callable, settings: dict):

    sysml_retry_num = settings.get("sysml_retry_num", 8)
    sysml_query_agent = make_query_builder_agent(elements, sysml_retry_num)

    rag_max_documents = settings.get("rag_max_documents", 5)
    
    rag_agent = make_rag_agent(
        lambda q: retrieve_text_from_azure_search(q, 0, rag_max_documents, True, True) 
    )

    reviewer_agent = make_reviewer_agent(max_reviews=settings.get("max_retry_reviewer", 1))

    diagram_retriever_node = RunnableLambda(diagram_retriever)
    summarizer = Summarizer()
    summarizer_runnable = RunnableLambda(summarizer)

    # Define agent state
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        rag_context: Dict[str, Any]
        question_for_rag: str
        question_for_model: str
        model_query_result: Dict[str, Any]
        rag_agent_result: str
        final_answer: str
        diagrams: List[str]
        complete: bool
        retry_count: int

    # Create the graph and link the nodes
    graph = StateGraph(AgentState)

    def review_decision(state: Dict[str, Any]) -> str:
        return state.get("call", "final")
    def set_retry_count(state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **state,
            "retry_count": 0
        }
    
    if settings.get("enable_planner_agent", False):
        planner_agent = make_planner_agent()
        graph.add_node("planner", planner_agent)
        graph.set_entry_point("planner")
        graph.add_edge("planner", "sysml_query_agent")

    else:
        graph.add_node("set_retry_count", set_retry_count)
        graph.set_entry_point("set_retry_count")
        graph.add_edge("set_retry_count", "sysml_query_agent")
        
    graph.add_node("sysml_query_agent", sysml_query_agent)
    graph.add_node("rag_agent", rag_agent)
    graph.add_node("diagram_retriever", diagram_retriever_node)
    graph.add_node("summarizer", summarizer_runnable)

    graph.add_edge("sysml_query_agent", "rag_agent")
    graph.add_edge("rag_agent", "diagram_retriever")
    graph.add_edge("diagram_retriever", "summarizer")

    if settings.get("enable_reviewer_agent", False):

        graph.add_node("reviewer", reviewer_agent)
        graph.add_edge("summarizer", "reviewer")
        graph.add_conditional_edges(
            "reviewer",
            review_decision,
            {
                "final": END,
                "sysml_query_agent": "sysml_query_agent",  # rerun query-builder
                "rag_agent": "rag_agent",                  # rerun RAG search
            },
        )
    else:
        graph.add_edge("summarizer", END)

    checkpointer = InMemorySaver()

    return graph.compile(checkpointer=checkpointer)



# ----------- EXECUTION OF QUERY -----------

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

    # Step 2: Build the agent
    log("Building LangGraph agent...")
    graph = build_sysml_langgraph_agent(elements_local, retrieve_text_from_azure_search, settings)
    log("LangGraph agent ready.")

    # Step 3: Execute the query
    log("Running agent on query...")
    config = {"configurable": {"thread_id": "1"}}

    final_state = {}

    for event in graph.stream(input=user_input, config=config, stream_mode="updates"):
        print("------------------------")
        # Safely access type using getattr (for robustness)
        print(f"Event type: {list(event.keys())[0]}")
        if isinstance(event, dict):
            node_name = list(event.keys())[0]  # e.g. 'rag_agent', 'summarizer', etc.
            node_output = event[node_name]

            if "final_answer" in node_output and node_output["final_answer"]:
                log(f"✅ Final answer received from `{node_name}`")
                final_state = node_output
                #break  # stop streaming if you're happy with final state
            else:
                log(f"🔄 Intermediate state from `{node_name}`")
                log(f"💬 {node_output['messages'][-1].content}")
                if node_name == "rag_agent":
                    log(f"🔍 Found {len(node_output['rag_context'])} relevant chunks")
                if node_name == "sysml_query_agent":
                    log(f"🔍 Found {len(node_output['model_query_result'])} relevant model elements")
                if node_name == "diagram_retriever":
                    log(f"📊 Found {len(node_output['diagrams'])} diagrams")
                if node_name == "summarizer":
                    log(f"📜 Summarizer input tokens")

    log("✅ Agent completed.")

    return final_state
