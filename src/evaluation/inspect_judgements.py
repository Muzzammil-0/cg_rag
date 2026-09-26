import json
from collections import Counter
from pathlib import Path

CHUNKS_PATH = Path("data/processed/chunks_v2.jsonl")

chunks = []
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

judgment_chunks = [c for c in chunks if c["metadata"].get("doc_type") == "judgment"]

print(f"Total chunks: {len(chunks)}")
print(f"Judgment chunks: {len(judgment_chunks)}")
print(f"Unique judgments: {len(set(c['metadata']['source'] for c in judgment_chunks))}")

per_doc = Counter(c["metadata"]["source"] for c in judgment_chunks)
values = list(per_doc.values())
print(f"Chunks per judgment - min: {min(values)}, max: {max(values)}, avg: {sum(values) // len(values)}")

print("\nSample chunks:")
for c in judgment_chunks[:3]:
    print("---")
    print("Metadata:", c["metadata"])
    print("Text:", c["text"][:250].replace("\n", " "))