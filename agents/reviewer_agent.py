from typing import Dict, Any
import json
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()


def make_reviewer_agent(max_reviews: int) -> RunnableLambda:
    """
    Returns JSON with keys:
    - complete: bool
    - answer: str
    - call: str | None
    """
    prompt_text = (
        "You are a reviewer. Evaluate if the AI-generated answer fully addresses the user's question. The questions are about a telescope project. Called TMT.\n"
        "Return a JSON object with keys:\n"
        "  - complete (true/false)\n"
        "  - answer (your feedback or final answer)\n"
        "  - call ( choose an agent to call next from ['sysml_query_agent', 'rag_agent']) only call final if the answer is flawless\n"
        "  - question_for_rag  if you want to call the rag agent specify a better question to retrieve context GIVE THE NEW QUESTION HERE\n"
        "  - question_for_model if you want to use the sysml_query_agent give it a better quesiton to retrieve the required elements GIVE THE NEW QUESTION HERE\n\n\n"
    )

    system_message = SystemMessage(content=prompt_text)

    def _reviewer(state: dict) -> dict:
       # print(f"[REVIEWER] State: {state}")
        print(f"[REVIEWER] Retry count: {state['retry_count']}")

        reviews = state['retry_count']
        if reviews >= max_reviews:
            return {
                **state,
                'complete': True,
                'call': None,
                'messages': state.get('messages', []),
            }

        # Extract last human question and AI answer
        question = state['question']
        final_answer = state['final_answer']

        # Build and run the LLM chain
        chat_prompt = ChatPromptTemplate.from_messages([
            system_message,
            HumanMessage(content=f"Question: {question}\nAnswer: {final_answer}"),
        ])
        
        print(f"[REVIEWER] PROMPT: {chat_prompt}")


        chain = (
            chat_prompt
            | AzureChatOpenAI(
                deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
                api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
                temperature=0,
            )
            | StrOutputParser()
        )
        raw = chain.invoke({})
        try:
            # strip and parse
            raw_txt = raw.strip().lstrip("```json").rstrip("```").strip()
            parsed = json.loads(raw_txt)
            print(f"[REVIEWER] parsed: {parsed}")
        except json.JSONDecodeError:
            parsed = {"complete": True, "answer": "Reviewer parsing error.", "call": None}
        call = parsed.get("call")


        new_msgs = state.get("messages", []) + [AIMessage(content=parsed["answer"])]

        print(f"[REVIEWER] Parsed next to call: {call}")

        return {
            **state,
            "complete": parsed.get("complete", False),
            "messages": new_msgs,
            "retry_count": state.get("retry_count", 0) + 1,
            "call": call,
            "question_for_model": parsed.get("question_for_model"),
            "question_for_rag": parsed.get("question_for_rag")
        }

    return RunnableLambda(_reviewer)

# In build_sysml_langgraph_agent, map 'model_query_agent' to 'sysml_query_agent'
# graph.add_conditional_edges(..., {'model_query_agent': 'sysml_query_agent', ...})
