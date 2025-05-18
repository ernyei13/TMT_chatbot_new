
import json
from pathlib import Path

# Find and open the only JSON file in the current directory
json_file = next(Path(".").glob("examples_2.json"))
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Count keywords per item
keyword_counts = [{"keyword_count": len(item.get("keywords", []))} for item in data]


print(keyword_counts)