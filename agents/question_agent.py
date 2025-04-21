import os
import json
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage


def make_followup_agent() -> RunnableLambda:
    """
    Create an agent that generates three standalone follow-up questions
    based on the AI's final answer.
    Returns JSON with key:
      - questions: List[str]
    """
    prompt_text = (
        "You are a follow-up question generator. "
        "Given the AI's final answer, generate three clear, standalone "
        "follow-up questions that a user might ask next. "
        "Return a JSON object with key 'questions' containing a list of "
        "exactly three questions."
    )
    system_message = SystemMessage(content=prompt_text)

    def _followup(state: Dict[str, Any]) -> Dict[str, Any]:
        print(f"[FOLLOWUP] started")
        final_answer = state.get("final_answer", "")
        chat_prompt = ChatPromptTemplate.from_messages([
            system_message,
            HumanMessage(content=f"Final Answer: {final_answer}")
        ])
        chain = (
            chat_prompt
            | AzureChatOpenAI(
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                temperature=0
            )
            | StrOutputParser()
        )
        raw = chain.invoke({})
        print(f"[FOLLOWUP] raw: {raw}")
        text = raw.strip().lstrip("```json").rstrip("```").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = {"questions": []}
        print(f"[FOLLOWUP]text: {text}")
        print(f"[FOLLOWUP] JSON: {parsed}")
        return {
            **state,
            "followup_q": parsed.get("questions", [])
        }

    return RunnableLambda(_followup)
