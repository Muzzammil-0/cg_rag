import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

def load_chunks(jsonl_path):
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def main():
    chunks_file = Path("data/processed/chunks.jsonl")
    db_path = Path("data/vector_db")
    collection_name = "legal_chunks"

    # Load chunks
    chunks = load_chunks(chunks_file)
    print(f"Loaded {len(chunks)} chunks.")

    # Initialize embedding model (CPU-friendly)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded.")

    # Prepare texts, ids, metadatas
    texts = [chunk["text"] for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [chunk["metadata"] for chunk in chunks]

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    # Initialize Chroma
    client = chromadb.PersistentClient(path=str(db_path))
    # Delete existing collection if exists to avoid duplicates (optional)
    try:
        client.delete_collection(collection_name)
    except:
        pass
    collection = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    # Add to collection
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas
    )
    print(f"Stored {len(chunks)} chunks in Chroma at {db_path}")

if __name__ == "__main__":
    main()