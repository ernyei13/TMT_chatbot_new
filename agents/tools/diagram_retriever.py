import os
from typing import Any, Dict, List


def diagram_retriever(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attach any .svg files whose filenames contain an element 'id'
    from state["model_query_result"].
    """
    raw = state.get('model_query_result', {})
    if isinstance(raw, dict):
        # handle dict-of-id->element or single element dict
        if 'id' in raw:
            model_elements = [raw]
            print("[DIAGRAM RETRIEVER] Found single element")
            print(model_elements)
        else:
            model_elements = list(raw.values())
            print("[DIAGRAM RETRIEVER] Found multiple elements")
            print(model_elements)
    elif isinstance(raw, list):
        model_elements = raw
        print("[DIAGRAM RETRIEVER] Found list of elements")
        print(model_elements)
    else:
        model_elements = []

    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )
    diagrams_dir = os.path.join(base_dir, "diagrams")
    if not os.path.isdir(diagrams_dir):
        raise FileNotFoundError(f"Directory not found: {diagrams_dir}")

    svg_files = [
        fname for fname in os.listdir(diagrams_dir)
        if fname.lower().endswith(".svg")
    ]

    related: List[str] = []
    for element in model_elements:
        print(f"Element: {element}")
        if not isinstance(element, dict):
            continue
        el_id = element.get("id")
        if not isinstance(el_id, str):
            continue
        for fname in svg_files:
            name, _ = os.path.splitext(fname)
            if el_id in name:
                related.append(os.path.join(diagrams_dir, fname))
                break

    return {**state, "diagrams": related}
