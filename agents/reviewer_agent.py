from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from typing import Dict, Any
import json
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()


def make_reviewer_agent() -> RunnableLambda:
    system_prompt = """
    You are a systems engineering assistant responsible for evaluating whether the latest AI-generated answer is sufficient.

    ### Your inputs:
    - A user's question about a system.
    - A proposed answer (from a summarizer agent).
    - The model elements are listed with their names and IDs.

    ### Your task:
    1. Evaluate if the answer fully addresses the question.
    2. If the answer is complete:
        - Respond in JSON: {"complete": true, "answer": "✔️ FINAL ANSWER"}
    3. If the answer is not complete:
         - If the issue is missing model data → call: "model_query_agent"
         - If the issue is lack of context → call: "rag_agent"
         - Always include a new query suggestion to rerun.

    Respond strictly in JSON:

    {
      "complete": false,
      "answer": "Explain what's missing.",
      "call": "model_query_agent" or "rag_agent",
      "new_query": "BE VERY DOMAIN SPECIFIC. INCLUDE THE NAMES AND IDs OF THE ELEMENTS IN YOUR QUERY."
    }
    """

    def _reviewer(state: Dict[str, Any]) -> Dict[str, Any]:
        question = next((m.content for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)), "")
        answer_to_evaluate = next((m.content for m in reversed(state.get("messages", [])) if isinstance(m, AIMessage)), "")

        model_result = state.get("model_query_result", [])
        rag_chunks = state.get("rag_context", {})
        diagrams = state.get("diagrams", [])

        # Extract element names and IDs from the model result
        model_element_summary = "\n".join(
            f"- {el.get('name', '<no-name>')} [ID: {el.get('id', 'unknown')}]" for el in model_result
        )

        input_text = {
            "question": question,
            "proposed_answer": answer_to_evaluate,
            "model_query_result": model_result,
            "model_element_summary": model_element_summary,
            "rag_context": [c.get("content", "") for c in rag_chunks],
            "diagrams": diagrams
        }

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content="""
                Question: {question}

                Proposed Answer:
                {proposed_answer}

                Model Element Summary:
                {model_element_summary}

                Full Model Data:
                {model_query_result}

                Retrieved Docs:
                {rag_context}

                Diagrams:
                {diagrams}
                """)
        ])

        chain = (
            prompt
            | AzureChatOpenAI(
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                temperature=0,
                seed=0,
            )
            | StrOutputParser()
        )

        response = chain.invoke(input_text)

        try:
            # Strip Markdown-style code fences if present
            if response.strip().startswith("```json"):
                response = response.strip("```").strip()
                if response.startswith("json"):
                    response = response[len("json"):].strip()
            
            parsed = json.loads(response)
            complete = parsed.get("complete", False)
            final_answer = parsed.get("answer", "")
            call = parsed.get("call", None)
            new_query = parsed.get("new_query", None)
        except Exception as e:
            print(f"[REVIEW] JSON parsing failed: {e}")
            complete = True
            final_answer = "Reviewer could not evaluate answer. Possibly malformed."
            call = None
            new_query = None

        new_messages = state.get("messages", []) + [AIMessage(content=final_answer)]
        if not complete and new_query:
            new_messages.append(HumanMessage(content=new_query))

        return {
            **state,
            "final_answer": final_answer,
            "complete": complete,
            "call": call,
            "messages": new_messages
        }

    return RunnableLambda(_reviewer)
