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
    batch_size = 5000  # below Chroma's max batch size

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
    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection '{collection_name}'.")
    except:
        pass
    collection = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    # Add in batches
    total = len(chunks)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        print(f"Adding batch {start} to {end-1}...")
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=texts[start:end],
            metadatas=metadatas[start:end]
        )

    print(f"Stored {total} chunks in Chroma at {db_path}")

if __name__ == "__main__":
    main()