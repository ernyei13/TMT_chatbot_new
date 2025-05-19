
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
from dotenv import load_dotenv
import os
import re

# Load environment variables from .env file
load_dotenv()

# Summarizer class that handles summarization tasks
class Summarizer:
    def __init__(self):
        print(f"[Summarizer] Summarizer model: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
        print(f"[Summarizer] Summarizer model: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}")
        self.llm = AzureChatOpenAI(
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                temperature=0,
                seed=0,
        )

        self.prompt_template = """
            You are an expert assistant who understands SysML models. You have access to information about elements in a SysML model and context from TMT project documentation.

            The relevant model elements (SysML) include:
            {model_query_result}

            Additional context from the documentation is:
            {rag_result}

            Diagrams related to the elements are:
            {diagrams}

            Based on these inputs, summarize the findings and provide a clear, concise answer about the question.
            The user is interested in understanding how these model elements and documentation relate to each other and what their significance is.
            If you mention an element allways include the id in square brackets, e.g. [ID].

            Provide a detailed yet understandable summary of the findings. With concretes examples and references to the model elements and documentation.

            Your ultimate goal is to provide an answer to the user's question, which is:
            {question}
        """

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_query_result = inputs.get("model_query_result")
        diagrams = inputs.get("diagrams", [])
        question = inputs.get("question")
        rag_result = inputs.get("rag_agent_result")
        print(f"[Summarizer] question: {question}")

        # Generate the final summary
        answer = self.summarize_responses(model_query_result, rag_result, diagrams, question)
        content = answer.content if hasattr(answer, "content") else answer
        ai_message = AIMessage(content=content)

        return {
            **inputs,
            "messages": inputs.get("messages", []) + [ai_message],
            "final_answer": content,
            "complete": False,
            "call": "reviewer",
        }

    #not used
    def generate_graph(self, elements: List[Dict[str, Any]]) -> str:
        # Create a simple graph of related elements
        # For simplicity, return just the names of related elements in a simple textual format
        graph = []
        for element in elements:
            graph.append(f"Element: {element['name']}, Type: {element['sysml_type']}")
        return "\n".join(graph)

    def summarize_responses(self, model_query_result: Dict, rag_result: str, diagrams, question: str) -> str:
        # Prepare the LLM prompt to summarize both results
        model_query_str = json.dumps(model_query_result, indent=2)

        diagram_svgs = []
        for d in diagrams:
            print(f"Diagram path: {d}")
            assert os.path.exists(d), "SVG file does not exist."

            with open(d, "r", encoding="utf-8") as file:
                svg = file.read()
                svg = re.sub(r'<style[\s\S]*?</style>', '', svg)
                svg = re.sub(r'style="[^"]*"', '', svg)
                svg = re.sub(r'fill="[^"]*"', '', svg)
                svg = re.sub(r'stroke="[^"]*"', '', svg)
                svg = re.sub(r'<metadata[\s\S]*?</metadata>', '', svg)
                svg = re.sub(r'<!--[\s\S]*?-->', '', svg)
                svg = re.sub(r'<image[\s\S]*?</image>', '', svg)  # Remove embedded images

                if len(svg) < 200000:
                    diagram_svgs.append(svg)
                else:

                    continue
                    
        print("lenght of diagrams")
        print(len("\n".join(diagram_svgs)))
        print("length of context")
        print(len("\n".join(rag_result)))
        print("length of model query result")
        print(len("\n".join(model_query_str)))

        prompt = self.prompt_template.format(
            model_query_result=model_query_str,
            rag_result=rag_result,
            diagrams=diagram_svgs,
            question=question
        )
        print(f'[SUMMARIZER] LENGTH OF INPUT: {len("".join(prompt))}')

        if len("".join(prompt)) > 10485760:
            prompt = self.prompt_template.format(
                model_query_result=model_query_str[:1000000],
                rag_result=rag_result,
                diagrams=diagram_svgs,
                question=question
            )

        # Call the LLM to summarize the response
        response = self.llm.invoke(prompt)
        return response
        