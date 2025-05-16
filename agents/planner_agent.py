from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from typing import Dict, Any, TypedDict, Sequence, List
from typing_extensions import Annotated
import json
from langgraph.graph import add_messages
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()

#not used
def make_planner_agent() -> RunnableLambda:
    system_prompt = """
    You are a planning agent specializing in model-based systems engineering using SysML.

    Your task is to generate a step-by-step plan to answer a user's question about a complex system modeled in SysML 1.x.

    ### Key Points:
    - SysML extends UML with systems engineering concepts like Blocks, Requirements, Activities, Sequence Diagrams, State Machines, Use Cases, and Allocations.
    - Determine if the answer requires:
      - **Model query**: For structure, relationships, dependencies, or allocations.
      - **RAG document retrieval**: For specifications, justifications, or rationale.
    - Provide precise SysML element names and IDs when mentioned in the question.

    ### Output Format (strict JSON):
    {
      "steps": [
        {
          "agent": "model_query_agent",
          "description": "What to extract from the SysML model",
          "query": "A SysML-specific structured query"
        },
        {
          "agent": "rag_agent",
          "description": "What to retrieve from documents",
          "query": "Domain-specific semantic query"
        }
      ],
      "notes": "Optional modeling hints or assumptions"
    }

    Always pass the user's question to both agents if unsure.
    """

    def _planner(state: Dict[str, Any]) -> Dict[str, Any]:
        question = next((m.content for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)), "")
        print(f"[PLANNER] User question: {question}")

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User Question:\n{question}")
            ]
        )
        print(f"[PLANNER] Prompt: {prompt}")

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

        response = chain.invoke({"question": question})
        print(f"[PLANNER] LLM JSON plan: {response}")

        try:
            if response.strip().startswith("```json"):
                response = response.strip("```").strip()
                if response.startswith("json"):
                    response = response[len("json"):].strip()

            plan_json = json.loads(response)

            # Safely populate questions for RAG and model agents
            steps = plan_json.get("steps", [])
            question_for_model = steps[0].get("query", question) if len(steps) > 0 else question
            question_for_rag = steps[1].get("query", question) if len(steps) > 1 else question
            print(f"[PLANNER] Model query question: {question_for_model}")
            print(f"[PLANNER] RAG query question: {question_for_rag}")

            return {
                **state,
                "question_for_model": question_for_model,
                "question_for_rag": question_for_rag
            }
        except Exception as e:
            print(f"[PLANNER] Failed to parse JSON: {e}")

            return {
                **state,
                "question_for_model": question,
                "question_for_rag": question
            }

    return RunnableLambda(_planner)
