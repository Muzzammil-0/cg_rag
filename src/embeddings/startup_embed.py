import json
import os
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer


def get_chroma_path():
    if os.getenv("SPACE_ID"):
        return Path("/tmp/vector_db")
    return Path("data/vector_db")


def build_if_needed():
    db_path = get_chroma_path()
    chunks_file = Path("data/processed/chunks_v2.jsonl")

    client = chromadb.PersistentClient(path=str(db_path))

    try:
        collection = client.get_collection("legal_chunks")
        if collection.count() > 0:
            print(f"ChromaDB already built: {collection.count()} chunks")
            return
    except Exception:
        pass

    print("Building ChromaDB from chunks_v2.jsonl...")
    chunks = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} chunks. Generating embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [c["metadata"] for c in chunks]

    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    try:
        client.delete_collection("legal_chunks")
    except Exception:
        pass
    collection = client.create_collection("legal_chunks", metadata={"hnsw:space": "cosine"})

    batch_size = 5000
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )
    print(f"ChromaDB built: {len(chunks)} chunks at {db_path}")


if __name__ == "__main__":
    build_if_needed()