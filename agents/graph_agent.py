from typing import Callable, Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
import json
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def clean_json_response(response: str) -> str:
    """Clean JSON string from code block formatting."""
    if response.strip().startswith("```"):
        response = response.strip("` \n")
        if response.startswith("json"):
            response = response[4:].strip()
    return response


# ----------- QUERY BUILDER NODE -----------
def make_query_builder_agent(elements, max_retry) -> RunnableLambda:
    print(f"[GRAPH AGENT] max retry: {max_retry}")
    system_prompt = """
        You are an expert assistant for querying SysML models, specifically for a system that manages TMT project documentation.  
        You are tasked with generating queries to search through SysML elements with a wide range of attributes, including relationships between elements.
        Always provide at least one filter.
        Given a question, follow these instructions:
        IMPORTANT: Only use the ID columns when explicitly given an ID in the question.

        1. **Filters:**
        - Identify the relevant filters based on the question. Filters can apply to:
            - sysml_type (a list of element types like "Block", "ValueType", "Requirement", etc.)
            - name (the name of the element)
            - type (SysML element types like "InstanceSpecification", "DataType", etc.)
            - documentation (textual content describing the element)
            - id (unique identifier for the element)
            - owner_id (id of the parent element, only use if ID is explicitly mentioned)
            - creator (who created the element)
            - modified (last modification time)
            - created (creation time)
        - You may also use **logical operators** like "contains", "equals", "startswith", and "endswith" for string matching.
        - Provide relevant **filter rules** to capture the intended query.

        2. **Relations:**
        - Identify if the question requires relations:
            - **Children**: Return a list of child elements.
            - **Owner**: Return the parent/owner of the element.
            - **Parent Chain**: Provide the full hierarchy of parent elements.
            - **Related Elements**: Return elements connected through children or owner relationships up to a depth of 2.
            - **Package Contents**: List elements within a package (recursively).
        - Include these relations in the output, if requested in the query.

        3. **Element Structure:**
        - Each element has attributes such as id, name, sysml_type, type, documentation, etc.
        - Relationships between elements include children (sub-elements) and owner (the parent element).
        - The system handles large, complex structures with multiple relations, so ensure that the output is formatted clearly, especially when dealing with parent-child hierarchies.

        4. **Query Structure:**
        - You will return a **JSON object** that includes:
            - **Filters**: A list of filter criteria (e.g., sysml_type, name, type).
            - **Relations**: Whether relations like children, owner, etc., should be included.
            - **Relation Types**: A list of relationship types to include (e.g., ["children", "owner"]).
        - Example of the query format:
            {
                "filters": [
                    {"field": "sysml_type", "operation": "contains", "value": "Block"},
                    {"field": "name", "operation": "contains", "value": "Controller"}
                ],
                "include_relations": true,
                "relation_types": ["children", "owner"]
            }

        5. **Handling Incomplete Context:**
        - If the context provided is insufficient to answer the question confidently, return a response indicating that the context is insufficient.

        ---

        ### Example Elements:
        You will be dealing with elements that have various attributes such as applied_stereotype_ids, documentation, type, id, owner_id, and slot_ids. Here is an example of a SysML element:

       

        ### Your Task:
        Based on the question you receive, output a JSON query to filter SysML elements by applying conditions to fields like sysml_type, type, name, documentation, creator, modified, created, and return related elements (such as children or owner) if needed.

        Ensure that the response is well-structured and uses logical operations such as "contains", "equals", or "startswith" where necessary.
        Your response must be ready for parsing in Python. NEVER return raw text or unstructured data. Never return a query that is not in JSON format. Never return an empty query.
        """

    def _agent(state: dict) -> dict:
        results = []
        messages = state["messages"]
        
        #use the question_for_model to get the question
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                q = msg.content
                break
        
        #extract the question from question_for_model

        question = state["question"]

        if state.get("question_for_model") is not None:
            question = "original question:" + question + " new question from the reviewer agent: " + state["question_for_model"]

        print("[GRAPH AGENT] Question to the Graph query agent:", question)
        
        prompt = ChatPromptTemplate.from_template(
            "{system_prompt}\n\nQuestion: {question}\n"
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

        # Use the system prompt and question to generate the query
        response = chain.invoke({
            "system_prompt": system_prompt,
            "question": question
        })
        print("Raw response from query builder:", response)

        response = clean_json_response(response)

        results = []
        attempts = 0
        max_attempts = max_retry
        max_results_length = 200000

        while attempts < max_attempts:
            try:
                query = json.loads(response)
                if not query.get("filters"):
                    return {
                        "messages": state["messages"] + [
                            AIMessage(content="No filters were generated. Please rephrase the question."),
                        ],
                        "model_query_result": []
                    }

                results = _executor({
                    "query": query,
                    "question": question,
                    "elements": elements
                })

                result_char_len = len("\n".join(json.dumps(results, indent=2)))
                print(f"Query to the Graph query agent (attempt {attempts + 1}):", query)
                print(f"Found {len(results)} elements.")
                
                if len(results) == 1 and result_char_len < max_results_length/5:
                    # Include relations with recursion
                    print("Found one element, including relations with recursion.")# Use get_basic_info instead of serialize
                    for element in elements.values():
                        if element.id == results[0]["id"]:
                            results = element.serialize(max_depth=1)
                            break

                if len(results) != 0 or result_char_len >= max_results_length:
                    break

                if len(results) == 0:
                    retry_reason = "No elements matched the query. Try again with a different, less restricting filter. Try searching for name. Try less filters. Do not try the same filter again. Try only one."
                    print("No elements matched the query.")
                else:
                    retry_reason = f"Too many elements matched the query. Try again with more specific filters."
                    print("Too many elements matched the query: ", result_char_len)

                retry_prompt = (
                    f"{retry_reason} Try making the filters more accurate.\n"
                    f"The previous filters were:\n{json.dumps(query, indent=2)}\n"
                    "Try using a different filter strategy.\n"
                )

                response = chain.invoke({
                    "system_prompt": system_prompt + "\nNOTE: " + retry_prompt,
                    "question": question
                })
                response = clean_json_response(response)

            except Exception as e:
                print("Failed to parse the query:", e)
                results = []
                break

            print("Retrying query... Attempt", attempts + 1, "of", max_attempts)
            attempts += 1


        print("number of retrieved model elements:", len(results))
        #print how many elements actually sent
        elements_new = state.get("model_query_result")
        for r in results:
            try:
                elements_new[r["id"]] = r
            except Exception as e:
                print("Failed to add element to the list, only including the newly found element", e)
                elements_new = results
                continue


        return {
            **state,
            "messages": state["messages"] + [
                AIMessage(content="model_query_result is filled"),
            ],
            "model_query_result": elements_new,
        }

    return RunnableLambda(_agent)

def _executor(state: dict) -> List[Dict[str, Any]]:
    query = state.get("query", {})
    elements = state.get("elements", {})
    max_results = 100  # Optional: limit number of main results
    max_relations = 5  # Optional: limit number of related items

    results = []

    def matches_filter(element, filter_rule):
        field = filter_rule.get("field", "")
        operation = filter_rule.get("operation", "")
        value = filter_rule.get("value", "")

        def string_match(source, match_value):
            source_str = str(source).lower()
            match_str = str(match_value).lower()

            if operation == "contains":
                return match_str in source_str
            elif operation == "equals":
                return match_str == source_str
            elif operation == "startswith":
                return source_str.startswith(match_str)
            elif operation == "endswith":
                return source_str.endswith(match_str)
            return False

        try:
            attr_value = getattr(element, field, None)
            if field == "sysml_type":
                attr_value = element.sysml_type or element.type

            if isinstance(attr_value, (str, int, float)):
                return string_match(attr_value, value)
            if isinstance(attr_value, list):
                return any(string_match(val, value) for val in attr_value)
            if '.' in field:
                # Handle nested fields like 'owner.name'
                current = element
                for attr in field.split('.'):
                    current = getattr(current, attr, None)
                    if current is None:
                        return False
                return string_match(current, value)
        except Exception as e:
            print(f"[Filter error] Field: {field}, Error: {e}")
            return False

        return False

    for element in elements.values():
        # Apply filters
        if query.get("filters") and not all(matches_filter(element, f) for f in query["filters"]):
            continue

        # Start building result
        result = element.get_basic_info(max_depth=0)

        if query.get("include_relations"):
            result["relations"] = {}

            if "children" in query.get("relation_types", []):
                result["relations"]["children"] = [
                    child.get_basic_info(max_depth=0)
                    for child in element.children[:max_relations]
                ]

            if "owner" in query.get("relation_types", []) and element.owner:
                result["relations"]["owner"] = element.owner.get_basic_info(max_depth=0)

            if "parent_chain" in query.get("relation_types", []):
                parent_chain = []
                current = element.owner
                while current:
                    parent_chain.append(current.get_basic_info(max_depth=0))
                    current = current.owner
                result["relations"]["parent_chain"] = parent_chain

            if "related" in query.get("relation_types", []):
                related_elements = []
                if element.owner:
                    related_elements.append(element.owner)
                related_elements.extend(element.children)
                result["relations"]["related_elements"] = [
                    e.get_basic_info(max_depth=0) for e in related_elements[:max_relations]
                ]

            if "package" in query.get("relation_types", []):
                # Flat package example
                result["relations"]["elements_in_package"] = [
                    c.get_basic_info(max_depth=0)
                    for c in element.children[:max_relations]
                ]

        results.append(result)
        

        if len(results) >= max_results:
            break
    print("number of retrieved model elements:", len(results))
    for r in results:
        print("retrieved element id:", r["id"])
        print("retrieved element name:", r["name"])
        print("retrieved element type:", r["sysml_type"])
    return results

