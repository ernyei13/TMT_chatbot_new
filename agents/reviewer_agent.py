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
    Create a reviewer agent that checks if the AI answer is complete.

    Returns JSON with keys:
    - complete: bool
    - answer: str
    - call: str | None
    """
    prompt_text = (
        "You are a reviewer. Evaluate if the AI-generated answer fully addresses the user's question. "
        "Return a JSON object with keys:\n"
        "  - complete (true/false)\n"
        "  - answer (your feedback or final answer)\n"
        "  - call (\"sysml_query_agent\" if model data is missing, "
        " \"rag_agent\" if context is missing, or null)"
    )

    system_message = SystemMessage(content=prompt_text)
    def _reviewer(state: dict) -> dict:
        print(f"[REVIEWER] State: {state}")
        print(f"[REVIEWER] System message: {state['retry_count']}")

        reviews = state['retry_count']
        if reviews >= max_reviews:
            return {
                **state,
                'final_answer': 'Reviewer limit reached.',
                'complete': True,
                'call': None,
                'messages': state.get('messages', []),
            }

        # Extract last human question and AI answer
        msgs = state['messages']
        question = next((m.content for m in reversed(msgs)
                         if isinstance(m, HumanMessage)), '')
        answer = next((m.content for m in reversed(msgs)
                       if isinstance(m, AIMessage)), '')
        final_answer = state['final_answer']

        # Build and run the LLM chain
        chat_prompt = ChatPromptTemplate.from_messages([
            system_message,
            HumanMessage(content=f"Question: {question}\nAnswer: {final_answer}"),
        ])
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
            if raw.strip().startswith('```'):
                raw = raw.strip('`\n')
            parsed = json.loads(raw)
            print(f"[REVIEWER] Parsed JSON: {parsed}")
        except json.JSONDecodeError:
            parsed = {'complete': True,
                      'answer': 'Reviewer parsing error.',
                      'call': None}

        # Append reviewer feedback and increment counter
        new_msgs = state.get('messages', []) + [AIMessage(content=parsed['answer'])]
        return {
            **state,
            'final_answer': parsed['answer'],
            'complete': parsed.get('complete', False),
            'call': parsed.get('call'),
            'messages': new_msgs,
            'retry_count': reviews + 1,
        }

    return RunnableLambda(_reviewer)

# In build_sysml_langgraph_agent, map 'model_query_agent' to 'sysml_query_agent'
# graph.add_conditional_edges(..., {'model_query_agent': 'sysml_query_agent', ...})
