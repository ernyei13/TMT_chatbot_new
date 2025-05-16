from typing import List, Dict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import List, Dict, Callable
from dotenv import load_dotenv
import os
from openai import AzureOpenAI


# Load environment variables from .env file
load_dotenv()

# ----------- RAG AGENT DEFINITION -----------
def make_rag_agent(retriever_fn: Callable) -> RunnableLambda:
    system_prompt = """
            You are an expert assistant on the Thirty Meter Telescope (TMT) project. 
            Your role is to support systems engineers and scientists by providing accurate, 
            well-cited answers based on the TMT project documentation, including but not limited to:
            - Science Requirements Document (SRD)
            - Observatory Architecture Document (OAD)
            - Operations Requirements Document (OpsRD)
            - Detailed Science Case (DSC)
            
            When given a question, analyze the provided document context and respond:
            - Clearly and concisely
            - Using phrases from the source context whenever possible
            - With focus on TMT system capabilities, design rationale, and scientific goals
            - If you encounter IDs mark them in the response as [ID]

            If the context is insufficient to answer confidently, state that clearly.
            Do not fabricate information or guess.
        """
    def _agent(state: dict) -> dict:
        messages = state["messages"]
        # If not found, check the messages
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                q = msg.content
                break
        
        #extract the question from question_for_rag

        question = q
     
        # Retrieve context chunks for the question
        if state.get("question_for_rag") is not None:
            question = "original question:" + question + " new question from the reviewer agent: " + state["question_for_rag"]

        #rephase the question
        client = AzureOpenAI(
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
            api_version="2025-02-01-preview"
        )

        prompt_rephrase = f"Rephrase the following question for a RAG agent: {question}"
        
        messages = [{"role": "user", "content": prompt_rephrase}] # Single function call

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )

        question_refined = response.choices[0].message.content
        print(f"[RAG] rephrased question: {question_refined}")

        context_chunks = retriever_fn(question_refined)
        context = state.get("context")
        if context is not None:
            context.append(context_chunks)
        context = context_chunks

        # Build the prompt
        prompt = ChatPromptTemplate.from_template(
            "{system_prompt}\n\nContext:\n{context}\n\nQuestion: {question} "
        )

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

        # Get the response from the chain
        print(f"[RAG] question: {question} ")
        print(f"[RAG] length of context: {len(context)}")
        answer = chain.invoke({
            "system_prompt": system_prompt,
            "context": context,
            "question": question,
        })

        

        return {
            **state,  # Keep everything from the original state
            "rag_context": context ,
            "rag_agent_result": answer,
            "messages": messages + [AIMessage(content=answer)]
        }
    return RunnableLambda(_agent)

