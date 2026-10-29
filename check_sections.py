import json

with open("data/processed/chunks_v2.jsonl", encoding="utf-8") as f:
    lines = [json.loads(l) for l in f]

ipc = [c for c in lines if "IPC_eng" in c["metadata"].get("source", "")]

for sec in ["166", "342", "350", "352", "354"]:
    chunks = [c for c in ipc if c["metadata"].get("section") == sec]
    print(f"IPC Section {sec}: {len(chunks)} chunks")
    for c in chunks[:1]:
        snippet = c["text"][:140].replace("\n", " ")
        print(f"   -> {snippet}")