from typing import Any, Dict, Sequence, TypedDict
from typing_extensions import Annotated
from langchain_core.messages import BaseMessage, add_messages

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    rag_context: Dict[str, Any]
    model_elements: Dict[str, Any]

    rag_agent_result: str
    diagrams: str
    final_answer: str
