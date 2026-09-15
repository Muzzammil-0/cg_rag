import json
from pathlib import Path

with open("data/processed/extracted_text.json", "r", encoding="utf-8") as f:
    d = json.load(f)

judgments = [(k, v["text"]) for k, v in d.items() if v.get("doc_type") == "judgment"]
print(f"Total judgments: {len(judgments)}")
print(f"First judgment: {judgments[0][0]}")
print(f"Length: {len(judgments[0][1])} chars")
print("\n--- First 3000 chars ---")
print(judgments[0][1][:3000])