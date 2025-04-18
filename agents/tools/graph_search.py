from typing import Callable, Dict, Any, List

def _executor(state: dict) -> List[Dict[str, Any]]:
    query = state.get("query", {})
    elements = state.get("elements", {})
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
            # Handling sysml_type and type
            if field == "sysml_type":
                # If sysml_type is filled, use it, else use the 'type' field
                sysml_value = element.sysml_type if element.sysml_type else element.type
                return string_match(sysml_value, value)
            # Handle direct attribute matching
            if hasattr(element, field):
                attr_value = getattr(element, field)

                # Handle basic fields (like sysml_type, name, etc.)
                if isinstance(attr_value, (str, int, float)):
                    return string_match(attr_value, value)
                
                # Handle list-based fields (sysml_type, children, etc.)
                if isinstance(attr_value, list):
                    return any(string_match(val, value) for val in attr_value)

                # Handle nested attribute matching (e.g., "owner.name")
                if isinstance(attr_value, object):
                    return string_match(str(attr_value), value)

            # If the field is nested (e.g., "owner.name"), split the field
            if '.' in field:
                attrs = field.split('.')
                current = element
                for attr in attrs:
                    if hasattr(current, attr):
                        current = getattr(current, attr)
                    else:
                        return False
                return string_match(current, value)

        except Exception as e:
            print(f"Error matching field {field}: {e}")
            return False

        return False

    def traverse_relationships(element, depth=1):
        elements_in_hierarchy = [element]
        
        if "children" in query.get("relation_types", []):
            for child in element.children:
                elements_in_hierarchy.append(child)
                if depth > 1:
                    elements_in_hierarchy.extend(traverse_relationships(child, depth - 1))

        if "owner" in query.get("relation_types", []) and element.owner:
            elements_in_hierarchy.append(element.owner)
            if depth > 1:
                elements_in_hierarchy.extend(traverse_relationships(element.owner, depth - 1))

        return elements_in_hierarchy

    # Package-based retrieval: Find all elements under a given package (recursively)
    def retrieve_elements_in_package(package_element, depth=1):
        all_elements = [package_element]
        if "children" in query.get("relation_types", []):
            for child in package_element.children:
                all_elements.append(child)
                if depth > 1:
                    all_elements.extend(retrieve_elements_in_package(child, depth - 1))
        return all_elements

    # Logic to handle the query, based on filters and relations
    for element in elements.values():
        # Apply filters to each element
        if query.get("filters", []) and not all(matches_filter(element, f) for f in query["filters"]):
            continue
        
        result = {
            "id": element.id,
            "name": element.name,
            "type": element.type,
            "sysml_type": element.sysml_type,
            "documentation": element.documentation,
            "elastic_id": element.elastic_id,
            "ref_id": element.ref_id,
            "modifier": element.modifier,
            "modified": element.modified,
            "creator": element.creator,
            "created": element.created,
            "project_id": element.project_id,
            "commit_id": element.commit_id,
            "editable": element.editable
        }

        # Handling relations (e.g., owner, children, parent chain)
        if query.get("include_relations", False):
            result["relations"] = {}

            # Handle children relation
            if "children" in query.get("relation_types", []):
                result["relations"]["children"] = [
                    {
                        "id": child.id, 
                        "name": child.name, 
                        "type": child.type,
                        "sysml_type": child.sysml_type
                    } 
                    for child in element.children
                ]
            
            # Handle owner relation
            if "owner" in query.get("relation_types", []) and element.owner:
                result["relations"]["owner"] = {
                    "id": element.owner.id,
                    "name": element.owner.name,
                    "type": element.owner.type,
                    "sysml_type": element.owner.sysml_type
                }

            # Handle full parent-child chain (nested hierarchy)
            if "parent_chain" in query.get("relation_types", []):
                parent_chain = []
                current_parent = element.owner
                while current_parent:
                    parent_chain.append({
                        "id": current_parent.id, 
                        "name": current_parent.name, 
                        "type": current_parent.type, 
                        "sysml_type": current_parent.sysml_type
                    })
                    current_parent = current_parent.owner
                result["relations"]["parent_chain"] = parent_chain

            # Package-based retrieval: What elements are in a package?
            if "package" in query.get("relation_types", []):
                elements_in_package = retrieve_elements_in_package(element)
                result["relations"]["elements_in_package"] = [
                    {"id": el.id, "name": el.name} for el in elements_in_package
                ]

            # Related elements based on relationship distance
            if "related" in query.get("relation_types", []):
                related_elements = traverse_relationships(element, depth=2)
                result["relations"]["related_elements"] = [
                    {"id": el.id, "name": el.name, "type": el.type} for el in related_elements
                ]

        results.append(result)

    return results
