import os

def diagram_retriever(state: dict) -> dict:
    model_elements = state.get("model_query_result", [])
    diagrams_path = "/Users/zoltanernyei/Documents/BME/szakdoga/clean_code_python/diagrams"
    existing_diagrams = os.listdir(diagrams_path)
    try:
        related_diagrams = []
        for el in model_elements:
            el_id = el.get("id")
            if any(f"{el_id}{ext}" in existing_diagrams for ext in ["", ".svg"]):
                related_diagrams.append(el_id)
        print(f"Related diagrams: {related_diagrams}")
    #catch all exceptions:
    except AttributeError:
        print("KeyError: 'id' not found in model elements")
        related_diagrams = []
        return {
            **state,
            "diagrams": [],
        }
    

    return {
        **state,
        "diagrams": [f"{diagrams_path}/{el_id}.svg" for el_id in related_diagrams if f"{el_id}.svg" in existing_diagrams],
    }
