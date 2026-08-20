import json
from pathlib import Path
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import re

# ----------------------------
# Load all chunks into memory
# ----------------------------
def load_chunks(jsonl_path):
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def tokenize(text):
    return re.findall(r'\w+', text.lower())

# ----------------------------
# BM25 setup
# ----------------------------
chunks = load_chunks("data/processed/chunks.jsonl")
corpus = [chunk["text"] for chunk in chunks]
tokenized_corpus = [tokenize(doc) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# ----------------------------
# Dense retrieval (Chroma)
# ----------------------------
client = chromadb.PersistentClient(path="data/vector_db")
collection = client.get_collection("legal_chunks")
model = SentenceTransformer('all-MiniLM-L6-v2')

def dense_search(query, top_k=20):
    query_embedding = model.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    # results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
    return list(zip(results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]))

def sparse_search(query, top_k=20):
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for idx in top_indices:
        results.append((f"chunk_{idx}", chunks[idx]["text"], chunks[idx]["metadata"], scores[idx]))
    return results

# ----------------------------
# Reciprocal Rank Fusion
# ----------------------------
def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    fused_scores = {}
    for rank, (id_, text, meta, _) in enumerate(dense_results):
        fused_scores[id_] = fused_scores.get(id_, 0) + 1 / (k + rank + 1)
    for rank, (id_, text, meta, _) in enumerate(sparse_results):
        fused_scores[id_] = fused_scores.get(id_, 0) + 1 / (k + rank + 1)
    # Sort by fused score descending
    sorted_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return sorted_ids, fused_scores

def hybrid_search(query, top_k=5):
    dense_results = dense_search(query, top_k=20)
    sparse_results = sparse_search(query, top_k=20)
    sorted_ids, fused_scores = reciprocal_rank_fusion(dense_results, sparse_results)
    # Map id back to chunk
    id_to_chunk = {f"chunk_{i}": chunks[i] for i in range(len(chunks))}
    final_results = []
    for id_ in sorted_ids[:top_k]:
        chunk = id_to_chunk[id_]
        final_results.append({
            "id": id_,
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "fused_score": fused_scores[id_]
        })
    return final_results

# ----------------------------
# Test
# ----------------------------
if __name__ == "__main__":
    query = "What is the punishment for theft under IPC?"
    results = hybrid_search(query, top_k=5)
    for i, res in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Score: {res['fused_score']:.4f}")
        print(f"Source: {res['metadata'].get('source')}")
        print(f"Text: {res['text'][:300]}...")