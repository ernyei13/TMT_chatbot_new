from typing import Any, Dict, List


def _executor(state: dict) -> List[Dict[str, Any]]:
    query = state.get("query", {})
    elements = state.get("elements", {})
    max_results = 10  # Optional: limit number of main results
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

    return results

