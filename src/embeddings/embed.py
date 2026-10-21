import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_chunks(jsonl_path):
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def get_contextual_text(chunk):
    """Prepends Act name and Section to the text for better retrieval."""
    meta = chunk["metadata"]
    source = meta.get("source", "").replace(".pdf", "").replace(".PDF", "")
    sec = meta.get("section", meta.get("article", meta.get("para", "")))
    
    # Clean up source name (e.g., "Protection_of_Women_from_Domestic_Violence_Act_2005")
    source = source.replace("_", " ")
    
    if sec:
        return f"[{source}] [Section {sec}] {chunk['text']}"
    return f"[{source}] {chunk['text']}"

def main():
    chunks_file = PROJECT_ROOT / "data/processed/chunks_v2.jsonl"
    db_path = PROJECT_ROOT / "data/vector_db"
    collection_name = "legal_chunks"
    batch_size = 5000

    chunks = load_chunks(chunks_file)
    print(f"Loaded {len(chunks)} chunks.")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # WE USE THE CONTEXTUAL TEXT HERE
    texts = [get_contextual_text(chunk) for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [chunk["metadata"] for chunk in chunks]

    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    client = chromadb.PersistentClient(path=str(db_path))
    try:
        client.delete_collection(collection_name)
    except:
        pass
    collection = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    total = len(chunks)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        print(f"Adding batch {start} to {end-1}...")
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=texts[start:end], # Store the contextual text in Chroma
            metadatas=metadatas[start:end]
        )

    print(f"Stored {total} chunks in Chroma at {db_path}")

if __name__ == "__main__":
    main()