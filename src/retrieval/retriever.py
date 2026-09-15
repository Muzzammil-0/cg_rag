import json
import os as _os
import re
from pathlib import Path
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import nltk
from nltk.stem import PorterStemmer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# ----------------------------
# Load chunks
# ----------------------------
def load_chunks(jsonl_path):
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

_stemmer = PorterStemmer()

def tokenize(text):
    tokens = re.findall(r'\w+', text.lower())
    return [_stemmer.stem(t) for t in tokens]

# ----------------------------
# Source (act) detection
# ----------------------------
SOURCE_PATTERNS = [
    (["bnss", "bharatiya nagarik suraksha", "nagarik suraksha"], "Bharatiya_Nagarik"),
    (["bsa", "bharatiya sakshya", "sakshya"], "BSA"),
    (["bns", "bharatiya nyaya sanhita", "nyaya sanhita"], "BNS"),
    (["ipc", "indian penal"], "IPC_eng"),
    (["crpc", "criminal procedure"], "criminal_procedure"),
    (["cpc", "civil procedure"], "civil_procedure"),
    (["evidence act", "indian evidence", "iea"], "iea_1872"),
    (["constitution"], "constitution"),
    (["pesa"], "pesa"),
    (["forest rights", "fra rules", "fra rule"], "FRA"),
    (["land revenue", "cg land revenue"], "Land_Revenue"),
    (["excise"], "Excise"),
    (["rent control"], "Rent_Control"),
    (["municipalit"], "Municipalities"),
    (["panchayat"], "Panchayat"),
    (["tendu"], "tendu"),
    (["vanopaj"], "vanopaj"),
    (["lok sewa", "lok-sewa"], "Lok-Sewa"),
    (["special court", "special courts"], "Special Courts"),
    (["cattle preservation", "agricultural cattle"], "Cattle"),
]

def detect_source_filter(query):
    q = query.lower()
    for keywords, source_fragment in SOURCE_PATTERNS:
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', q):
                return source_fragment
    return None

def matches_source(metadata, source_fragment):
    src = metadata.get("source", "")
    return source_fragment.lower() in src.lower()

# ----------------------------
# Judgment-query detection
# ----------------------------
JUDGMENT_PATTERNS = [
    "what have courts said",
    "what have cg courts",
    "what have the courts",
    "what has the court said",
    "case law",
    "precedent",
    "judgment",
    "judgements",
    "judgments",
    "high court held",
    "high court said",
    "high court observed",
    "courts have dealt",
    "courts have interpreted",
    "case law on",
    "ruling",
    "rulings",
    "has the court held",
]

def query_wants_judgments(query):
    q = query.lower()
    return any(p in q for p in JUDGMENT_PATTERNS)

# ----------------------------
# Explicit section detection
# ----------------------------
def extract_section_from_query(query):
    q = query.lower()
    m = re.search(r'\barticle\s+(\d+[a-z]?)\b', q)
    if m:
        return m.group(1).upper()
    m = re.search(r'\b(?:section|sec\.?)\s+(\d+[a-z]?)\b', q)
    if m:
        return m.group(1).upper()
    return None

# ----------------------------
# Synonym hints
# ----------------------------
SECTION_HINTS = [
    # IPC
    ("definition of theft", "IPC", "378"),
    ("punishment for theft", "IPC", "379"),
    ("punishment for cheating", "IPC", "420"),
    ("punishment for murder", "IPC", "302"),
    ("attempt to murder", "IPC", "307"),
    ("punishment for dacoity", "IPC", "395"),
    ("punishment for defamation", "IPC", "500"),
    ("culpable homicide", "IPC", "299"),
    # BNS
    ("definition of theft", "BNS", "303"),
    ("punishment for murder", "BNS", "103"),
    ("attempt to murder", "BNS", "109"),
    ("culpable homicide", "BNS", "105"),
    # CrPC
    ("fir", "criminal_procedure", "154"),
    ("first information report", "criminal_procedure", "154"),
    ("bail", "criminal_procedure", "436"),
    ("arrest without warrant", "criminal_procedure", "41"),
    # BNSS
    ("fir", "Bharatiya_Nagarik", "173"),
    ("arrest without warrant", "Bharatiya_Nagarik", "35"),
    # CPC
    ("place of suing", "civil_procedure", "15"),
    ("where can a civil suit be filed", "civil_procedure", "15"),
    ("res judicata", "civil_procedure", "11"),
    ("res sub judice", "civil_procedure", "10"),
    # Evidence Act
    ("primary evidence", "iea_1872", "62"),
    ("secondary evidence", "iea_1872", "63"),
    ("electronic evidence", "iea_1872", "65B"),
    ("burden of proof", "iea_1872", "101"),
    # Constitution
    ("right to education", "constitution", "21A"),
    ("right to life", "constitution", "21"),
    ("equality before law", "constitution", "14"),
    ("freedom of speech", "constitution", "19"),
    ("constitutional remedies", "constitution", "32"),
    ("uniform civil code", "constitution", "44"),
    # PESA
    ("where does pesa apply", "pesa", "3"),
    ("pesa applicability", "pesa", "3"),
    # CG Rent Control
    ("eviction", "Rent_Control", "12"),
    # CG Land Revenue
    ("mutation", "Land_Revenue", "109"),
    ("partition", "Land_Revenue", "178"),
]

def detect_section_hint(query, source_filter):
    if not source_filter:
        return None
    q = query.lower()
    sf = source_filter.lower()
    for phrase, src_hint, section in SECTION_HINTS:
        if phrase in q and src_hint.lower() in sf:
            return section
    return None

# ----------------------------
# BM25 setup
# ----------------------------
chunks = load_chunks("data/processed/chunks.jsonl")
corpus = [chunk["text"] for chunk in chunks]
tokenized_corpus = [tokenize(doc) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

_bm25_cache = {}

def _bm25_key(source_fragment, doc_type_filter):
    return (source_fragment or "", doc_type_filter or "")

def get_filtered_bm25(source_fragment, doc_type_filter=None):
    key = _bm25_key(source_fragment, doc_type_filter)
    if key in _bm25_cache:
        return _bm25_cache[key]

    indices = []
    for i, c in enumerate(chunks):
        meta = c["metadata"]
        if source_fragment and not matches_source(meta, source_fragment):
            continue
        if doc_type_filter and meta.get("doc_type") != doc_type_filter:
            continue
        indices.append(i)

    if not indices:
        _bm25_cache[key] = (None, [], [])
        return _bm25_cache[key]

    filtered_corpus = [tokenized_corpus[i] for i in indices]
    filtered_bm25 = BM25Okapi(filtered_corpus)
    result = (filtered_bm25, [chunks[i] for i in indices], indices)
    _bm25_cache[key] = result
    return result

# ----------------------------
# Chroma + models
# ----------------------------
def _chroma_path():
    if _os.getenv("SPACE_ID"):
        return "/tmp/vector_db"
    return "data/vector_db"

client = chromadb.PersistentClient(path=_chroma_path())
model = SentenceTransformer('all-MiniLM-L6-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

_collection = None

def get_collection():
    global _collection
    if _collection is not None:
        return _collection
    try:
        _collection = client.get_collection("legal_chunks")
    except Exception:
        from src.embeddings.startup_embed import build_if_needed
        build_if_needed()
        _collection = client.get_collection("legal_chunks")
    return _collection

ALL_SOURCES = sorted(set(c["metadata"].get("source", "") for c in chunks))

SECTION_MAP = {}
for i, c in enumerate(chunks):
    src = c["metadata"].get("source", "").lower()
    sec = str(
        c["metadata"].get("section",
                          c["metadata"].get("article",
                                            c["metadata"].get("para", "")))
    ).upper().rstrip('.').strip()
    if sec:
        SECTION_MAP.setdefault((src, sec), []).append(f"chunk_{i}")

def find_chunks_by_section(source_filter, target_section, doc_type_filter=None):
    if not target_section:
        return []
    target = target_section.upper().rstrip('.').strip()
    matching_ids = []
    for (src, sec), ids in SECTION_MAP.items():
        if sec != target:
            continue
        if source_filter and source_filter.lower() not in src:
            continue
        for cid in ids:
            if doc_type_filter:
                try:
                    idx = int(cid.replace("chunk_", ""))
                    if chunks[idx]["metadata"].get("doc_type") != doc_type_filter:
                        continue
                except (ValueError, IndexError):
                    continue
            matching_ids.append(cid)
    return matching_ids

def dense_search(query, top_k=20, source_filter=None, doc_type_filter=None):
    query_embedding = model.encode([query])[0]
    kwargs = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }

    where_conditions = []
    if source_filter:
        matching_sources = [s for s in ALL_SOURCES if source_filter.lower() in s.lower()]
        if not matching_sources:
            return []
        if len(matching_sources) == 1:
            where_conditions.append({"source": matching_sources[0]})
        else:
            where_conditions.append({"source": {"$in": matching_sources}})

    if doc_type_filter:
        where_conditions.append({"doc_type": doc_type_filter})

    if len(where_conditions) == 1:
        kwargs["where"] = where_conditions[0]
    elif len(where_conditions) > 1:
        kwargs["where"] = {"$and": where_conditions}

    results = get_collection().query(**kwargs)
    rows = list(zip(
        results["ids"][0], results["documents"][0],
        results["metadatas"][0], results["distances"][0],
    ))
    return rows[:top_k]

def sparse_search(query, top_k=20, source_filter=None, doc_type_filter=None):
    tokenized_query = tokenize(query)

    if source_filter or doc_type_filter:
        filtered_bm25, filtered_chunks, original_indices = get_filtered_bm25(
            source_filter, doc_type_filter
        )
        if filtered_bm25 is None:
            return []
        scores = filtered_bm25.get_scores(tokenized_query)
        top_local = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            (f"chunk_{original_indices[i]}",
             filtered_chunks[i]["text"],
             filtered_chunks[i]["metadata"],
             scores[i])
            for i in top_local
        ]
    else:
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            (f"chunk_{i}", chunks[i]["text"], chunks[i]["metadata"], scores[i])
            for i in top_indices
        ]

# ----------------------------
# RRF
# ----------------------------
def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    fused_scores = {}
    for rank, (id_, _, _, _) in enumerate(dense_results):
        fused_scores[id_] = fused_scores.get(id_, 0) + 1 / (k + rank + 1)
    for rank, (id_, _, _, _) in enumerate(sparse_results):
        fused_scores[id_] = fused_scores.get(id_, 0) + 1 / (k + rank + 1)
    sorted_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return sorted_ids, fused_scores

# ----------------------------
# Hybrid search
# ----------------------------
def hybrid_search(query, top_k=5, source_filter=None, doc_type_filter=None, rerank=True):
    if source_filter is None:
        source_filter = detect_source_filter(query)
    if doc_type_filter is None and query_wants_judgments(query):
        doc_type_filter = "judgment"

    target_section = extract_section_from_query(query)
    if not target_section:
        target_section = detect_section_hint(query, source_filter)

    final_results = []
    seen_ids = set()

    if target_section:
        section_ids = find_chunks_by_section(source_filter, target_section, doc_type_filter)
        for cid in section_ids:
            if cid in seen_ids:
                continue
            try:
                idx = int(cid.replace("chunk_", ""))
                c = chunks[idx]
            except (ValueError, IndexError):
                continue
            final_results.append({
                "id": cid,
                "text": c["text"],
                "metadata": c["metadata"],
                "fused_score": 1.0,
                "match_type": "section_exact",
            })
            seen_ids.add(cid)
            if len(final_results) >= top_k:
                break

    if len(final_results) < top_k:
        dense_results = dense_search(
            query, top_k=40, source_filter=source_filter, doc_type_filter=doc_type_filter
        )
        sparse_results = sparse_search(
            query, top_k=40, source_filter=source_filter, doc_type_filter=doc_type_filter
        )

        sorted_ids, fused_scores = reciprocal_rank_fusion(dense_results, sparse_results)
        id_to_chunk = {f"chunk_{i}": chunks[i] for i in range(len(chunks))}

        general_candidates = []
        for cid in sorted_ids:
            if cid in seen_ids:
                continue
            c = id_to_chunk.get(cid)
            if c:
                general_candidates.append({"id": cid, **c})
            if len(general_candidates) >= 30:
                break

        remaining = top_k - len(final_results)
        if rerank and general_candidates:
            pairs = [(query, c["text"]) for c in general_candidates]
            rerank_scores = reranker.predict(pairs)
            reranked = sorted(zip(general_candidates, rerank_scores), key=lambda x: x[1], reverse=True)
            for c, s in reranked[:remaining]:
                final_results.append({
                    "id": c["id"],
                    "text": c["text"],
                    "metadata": c["metadata"],
                    "fused_score": float(s),
                    "match_type": "reranked",
                })
        else:
            for c in general_candidates[:remaining]:
                final_results.append({
                    "id": c["id"],
                    "text": c["text"],
                    "metadata": c["metadata"],
                    "fused_score": fused_scores.get(c["id"], 0.0),
                    "match_type": "rrf",
                })

    return final_results[:top_k]


# ----------------------------
# Test
# ----------------------------
if __name__ == "__main__":
    test_queries = [
        "What is the punishment for murder under BNS?",
        "What is the definition of theft under IPC?",
        "What is the procedure for registering an FIR under CrPC?",
        "What are the grounds for eviction under the Chhattisgarh Rent Control Act?",
        "What have CG courts said about land acquisition compensation?",
        "What have the courts held about electricity theft?",
    ]
    for q in test_queries:
        sf = detect_source_filter(q)
        wants_j = query_wants_judgments(q)
        print(f"\n=== {q} ===")
        print(f"  source_filter={sf}  judgment_query={wants_j}")
        results = hybrid_search(q, top_k=3)
        for i, r in enumerate(results):
            m = r["metadata"]
            sec = m.get("section", m.get("article", m.get("para", "")))
            print(f"  [{i+1}] {m.get('source')} sec/para={sec} ({r.get('match_type')})")
            print(f"      {r['text'][:140].replace(chr(10), ' ')}...")