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

    # Base prompt shared by both modes
    base_prompt = (
        "You are a critical reviewer for AI-generated responses in the domain of telescope systems, specifically for the Thirty Meter Telescope (TMT).\n\n"
        "Your job is to decide if the AI's answer is complete, useful, and accurate.\n"
        "IF THE ANSWER IS SAYING THERE IS NO CONTEXT OR DID NOT FOUND IT THEN IT IS NOT COMPLETE!\n"
        "## Instructions:\n"
        "1. Evaluate whether the answer fully addresses the user's question.\n"
        "2. If the answer is incomplete, vague, or incorrect, suggest what should be improved.\n"
        "3. If the answer does not mention elements by ID in brackets it is INCOMPLETE.\n"
        "4. If there are IDs for elements in the answer ask about the most important one from the sysml_query_agent in this case include the ID in your question.\n"
        "5. Decide what agent to call next to improve the result:\n"
        "   - 'rag_agent' for better documentation context.\n"
        "   - 'sysml_query_agent' for more relevant SysML model elements.\n"
        "6. Rewrite the user question to help the next agent use knowledge from the already given answer (RAG or model).\n"
        "7. Return your evaluation as a JSON object with the following keys:\n"
        "```\n"
        "{\n"
        "  \"complete\": true | false,\n"
        "  \"answer\": \"your final feedback\",\n"
        "  \"call\": \"rag_agent\" | \"sysml_query_agent\" | null,\n"
        "  \"question_for_rag\": \"new refined question for Docuements Retriever Agent\",\n"
        "  \"question_for_model\": \"new question for the SysML model\"\n"
        "}\n"
        "```\n"
        "## Rules:\n"
    )

    # Prompt for first round: complete must be false
    first_round_prompt = base_prompt + (
        "- You are in the initial round. Never return 'complete': true, even if the answer seems decent.\n"
        "- Always suggest what to improve and which agent to call.\n"
        "- You are expected to drive iteration at this stage.\n"
    )

    # Prompt for later rounds: allow 'complete: true'
    final_prompt = base_prompt + (
        "- If the answer is complete, well-cited, and includes referenced IDs, you may return 'complete': true.\n"
        "- Otherwise, suggest improvements and reroute the query.\n"
    )

    def _reviewer(state: dict) -> dict:
        reviews = state.get('retry_count', 0)
        print(f"[REVIEWER] Retry count: {reviews}")

        # Use the correct prompt depending on stage
        system_prompt = SystemMessage(
            content=first_round_prompt if reviews == 0 else final_prompt
        )

        # Early stop
        if reviews >= max_reviews:
            return {
                **state,
                'complete': True,
                'call': None,
                'messages': state.get('messages', []),
            }

        question = state['question']
        final_answer = state['final_answer']
        elements = state['model_query_result']

        chat_prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            HumanMessage(content=f"Question: {question}\nRetry: {reviews}\nAnswer: {final_answer}\nRelevant model elements: {elements}")
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
            raw_txt = raw.strip().lstrip("```json").rstrip("```").strip()
            parsed = json.loads(raw_txt)
        except json.JSONDecodeError:
            parsed = {
                "complete": True,
                "answer": "Reviewer failed to parse output.",
                "call": None
            }

        print(f"[REVIEWER] parsed: {parsed}")
        call = parsed.get("call")
        new_msgs = state.get("messages", []) + [AIMessage(content=parsed["answer"])]

        return {
            **state,
            "complete": parsed.get("complete", False),
            "messages": new_msgs,
            "retry_count": reviews + 1,
            "call": call,
            "question_for_model": parsed.get("question_for_model"),
            "question_for_rag": parsed.get("question_for_rag")
        }

    return RunnableLambda(_reviewer)
