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
        "You are a critical reviewer for AI-generated responses in the domain of telescope systems, specifically for the Thirty Meter Telescope (TMT).\n\n"
        "Your job is to decide if the AI's answer is complete, useful, and accurate.\n"
        "IF THE ANSWER IS SAYING THERE IS NO CONTEXT OR DID NOT FOUND IT THEN IT IS NOT COMPLETE!\n"
        "## Instructions:\n"
        "1. Evaluate whether the answer fully addresses the user's question.\n"
        "2. If the answer is incomplete, vague, or incorrect, suggest what should be improved.\n"
        "3. If the answer is missing elements mentioned by ID in [] call the sysml_query_agent.\n"
        "4. Decide what agent to call next to improve the result:\n"
        "   - 'rag_agent' for better documentation context.\n"
        "   - 'sysml_query_agent' for more relevant SysML model elements.\n"
        "5. Optionally rewrite the user question to help the next agent (RAG or model).\n"
        "6. Return your evaluation as a JSON object with the following keys:\n\n"
        "```\n"
        "{\n"
        "  \"complete\": true | false,               // Is the answer sufficient and final?\n"
        "  \"answer\": \"your final feedback\",       // Explain your reasoning or return the improved answer\n"
        "  \"call\": \"rag_agent\" | \"sysml_query_agent\" | null,\n"
        "  \"question_for_rag\": \"...optional...\", // Rewrite if calling rag_agent\n"
        "  \"question_for_model\": \"...optional...\" // Rewrite if calling sysml_query_agent\n"
        "}\n"
        "```\n\n"
        "## Rules:\n"
        "- Always return 'complete': true when the answer is flawless.\n"
        "- Never call another agent if the answer already satisfies the user.\n"
        "- Use plain language, be constructive, and suggest improvements if needed.\n"
        "- Do not return Markdown or escape JSON.\n"
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
            HumanMessage(content=f"Question: {question}\n Number of tries before this {reviews} Answer: {final_answer}"),
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
