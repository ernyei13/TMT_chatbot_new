from loaders.json_loader import load_elements

elements = load_elements()

first_element = next(iter(elements.values()))
print(first_element.serialize())


for value in list(elements.values())[:20]:
    print(value.serialize())
