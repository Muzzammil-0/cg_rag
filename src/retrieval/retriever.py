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
# Source (act) detection — HINTS TIER
# ----------------------------
SOURCE_PATTERNS = [
    # New criminal law triad
    (["bnss", "bharatiya nagarik suraksha", "nagarik suraksha"], "Bharatiya_Nagarik"),
    (["bsa", "bharatiya sakshya", "sakshya"], "BSA"),
    (["bns", "bharatiya nyaya sanhita", "nyaya sanhita"], "BNS"),
    # Old criminal law
    (["ipc", "indian penal"], "IPC_eng"),
    (["crpc", "criminal procedure"], "criminal_procedure"),
    (["evidence act", "indian evidence", "iea"], "iea_1872"),
    # Civil / Constitutional
    (["cpc", "civil procedure"], "civil_procedure"),
    (["constitution"], "constitution"),
    # Central acts
    (["hindu marriage"], "Hindu_Marriage"),
    (["special marriage"], "Special_Marriage"),
    (["dowry prohibition", "dowry"], "Dowry_Prohibition"),
    (["domestic violence", "dv act"], "Domestic_Violence"),
    (["child marriage", "pcma", "prohibition of child marriage"], "Child_Marriage"),
    (["negotiable instrument", "ni act", "cheque bounce", "section 138",
      "cheque dishonour", "dishonour of cheque"], "Negotiable"),
    (["motor vehicle", "motor vehicles", "mv act", "motor accident", "motor claim"], "Motor"),
    (["ndps", "narcotic", "psychotropic"], "Narcotic"),
    (["pocso", "protection of children from sexual"], "pocso"),
    (["arms act"], "ArmsAct"),
    (["prevention of corruption", "pc act", "corruption act"], "Corruption"),
    (["sc st act", "sc/st", "scheduled castes", "atrocities act",
      "sc st atrocities", "sc/st atrocities"], "Scheduled"),
    (["juvenile justice", "jj act", "juvenile"], "Juvenile"),
    (["maintenance and welfare of parents", "parents act"], "Maintenance_and_Welfare"),
    (["pesa"], "pesa"),
    (["forest rights", "fra rules", "fra rule"], "FRA"),
    # CG acts
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


def matches_source(metadata, source_filter):
    """source_filter can be a string OR a list/tuple of strings."""
    src = metadata.get("source", "").lower()
    if not source_filter:
        return True
    if isinstance(source_filter, (list, tuple, set)):
        return any(str(f).lower() in src for f in source_filter)
    return str(source_filter).lower() in src


# ----------------------------
# Classifier fallback — CLASSIFIER TIER
# ----------------------------
CLASSIFIER_DIR = Path(__file__).resolve().parents[2] / "data" / "classifier" / "model"
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.30

# Maps classifier act names → source-fragment lists.
# Merged classes (Criminal Code, Criminal Procedure, etc.) map to multiple sources.
CLASSIFIER_ACT_TO_FRAGMENTS = {
    "Arms Act": ["ArmsAct"],
    "BNS": ["BNS"],
    "BNSS": ["Bharatiya_Nagarik"],
    "BSA": ["BSA"],
    "CG Cattle Preservation Rules": ["Cattle"],
    "CG Excise Act": ["Excise"],
    "CG Land Revenue Code": ["Land_Revenue"],
    "CG Lok Sewa Guarantee Act": ["Lok-Sewa"],
    "CG Municipalities Act": ["Municipalities"],
    "CG Panchayat Raj Act": ["Panchayat"],
    "CG Rent Control Act": ["Rent_Control"],
    "CG Special Courts Act": ["Special Courts"],
    "CG Tendu Patta Act": ["tendu"],
    "CG Vanopaj Act": ["vanopaj"],
    "CPC": ["civil_procedure"],
    "Criminal Code": ["IPC_eng", "BNS"],
    "Criminal Procedure": ["criminal_procedure", "Bharatiya_Nagarik"],
    "Domestic Violence Act": ["Domestic_Violence"],
    "Dowry Prohibition Act": ["Dowry_Prohibition"],
    "Evidence Law": ["iea_1872", "BSA"],
    "Forest Rights Act Rules": ["FRA"],
    "Hindu Marriage Act": ["Hindu_Marriage"],
    "Juvenile Justice Act": ["Juvenile"],
    "Marriage Law": ["Hindu_Marriage", "Special_Marriage"],
    "Motor Vehicles Act": ["Motor"],
    "Negotiable Instruments Act": ["Negotiable"],
    "PESA": ["pesa"],
    "POCSO Act": ["pocso"],
    "Prevention of Corruption Act": ["Corruption"],
    "SC/ST (PoA) Act": ["Scheduled"],
    "Special Marriage Act": ["Special_Marriage"],
}

_classifier_bundle = None   # None = not loaded yet, False = failed/unavailable


def _load_classifier():
    """Lazy-load the act classifier. Returns bundle dict or False."""
    global _classifier_bundle
    if _classifier_bundle is not None:
        return _classifier_bundle

    if not CLASSIFIER_DIR.exists():
        _classifier_bundle = False
        return False

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained(str(CLASSIFIER_DIR))
        model = AutoModelForSequenceClassification.from_pretrained(str(CLASSIFIER_DIR))
        model.eval()

        labels_path = CLASSIFIER_DIR / "act_labels.json"
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

        _classifier_bundle = {
            "tokenizer": tokenizer,
            "model": model,
            "idx_to_act": {int(k): v for k, v in labels["idx_to_act"].items()},
            "torch": torch,
        }
        print("[classifier] loaded")
    except Exception as e:
        print(f"[classifier] failed to load: {e}")
        _classifier_bundle = False

    return _classifier_bundle

def classifier_route(query, threshold=CLASSIFIER_CONFIDENCE_THRESHOLD):
    """
    Returns a list of source fragments if the classifier is confident
    AND the predicted act maps to at least one source. Otherwise None.
    """
    bundle = _load_classifier()
    if not bundle:
        return None

    try:
        tok = bundle["tokenizer"]
        model = bundle["model"]
        idx_to_act = bundle["idx_to_act"]
        torch = bundle["torch"]

        inputs = tok(query, return_tensors="pt", truncation=True,
                     padding="max_length", max_length=64)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            top_prob, top_idx = float(probs.max()), int(probs.argmax())

        if top_prob < threshold:
            return None

        act = idx_to_act.get(top_idx)
        if not act:
            return None

        fragments = CLASSIFIER_ACT_TO_FRAGMENTS.get(act)
        if not fragments:
            return None

        return fragments
    except Exception as e:
        print(f"[classifier] inference failed: {e}")
        return None


# ----------------------------
# Judgment-query detection
# ----------------------------
JUDGMENT_PATTERNS = [
    "what have courts said", "what have cg courts", "what have the courts",
    "what has the court said", "what have courts held", "what have the courts held",
    "what has the court held", "what have the high court", "what has the high court",
    "case law", "precedent", "judgment", "judgements", "judgments",
    "high court held", "high court said", "high court observed",
    "courts have dealt", "courts have interpreted", "courts have ruled",
    "court has ruled", "case law on", "judgments on", "ruling", "rulings",
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
    ("common intention", "IPC", "34"),
    ("criminal conspiracy", "IPC", "120B"),
    ("abetment", "IPC", "107"),
    ("definition of culpable homicide", "IPC", "299"),
    ("what is culpable homicide", "IPC", "299"),
    ("culpable homicide", "IPC", "299"),
    ("definition of murder", "IPC", "300"),
    ("what is murder", "IPC", "300"),
    ("elements of murder", "IPC", "300"),
    ("ingredients of murder", "IPC", "300"),
    ("murder definition", "IPC", "300"),
    ("murder", "IPC", "302"),
    ("dowry death", "IPC", "304B"),
    ("abetment of suicide", "IPC", "306"),
    ("definition of theft", "IPC", "378"),
    ("punishment for theft", "IPC", "379"),
    ("punishment for cheating", "IPC", "420"),
    ("punishment for murder", "IPC", "302"),
    ("attempt to murder", "IPC", "307"),
    ("punishment for dacoity", "IPC", "395"),
    ("punishment for defamation", "IPC", "500"),
    ("definition of rape", "IPC", "375"),
    ("what constitutes rape", "IPC", "375"),
    ("elements of rape", "IPC", "375"),
    ("ingredients of rape", "IPC", "375"),
    ("conditions for rape", "IPC", "375"),
    ("conditions for a case to be considered rape", "IPC", "375"),
    ("when is it rape", "IPC", "375"),
    ("punishment for rape", "IPC", "376"),
    ("rape", "IPC", "376"),
    ("dowry cruelty", "IPC", "498A"),
    # BNS
    ("definition of theft", "BNS", "303"),
    ("attempt to murder", "BNS", "109"),
    ("definition of murder", "BNS", "101"),
    ("what is murder", "BNS", "101"),
    ("elements of murder", "BNS", "101"),
    ("culpable homicide", "BNS", "105"),
    ("punishment for murder", "BNS", "103"),
    ("murder", "BNS", "103"),
    ("definition of rape", "BNS", "63"),
    ("what constitutes rape", "BNS", "63"),
    ("elements of rape", "BNS", "63"),
    ("ingredients of rape", "BNS", "63"),
    ("conditions for rape", "BNS", "63"),
    ("conditions for a case to be considered rape", "BNS", "63"),
    ("when is it rape", "BNS", "63"),
    ("punishment for rape", "BNS", "64"),
    ("rape", "BNS", "64"),
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
    # Hindu Marriage Act
    ("conditions for hindu marriage", "Hindu_Marriage", "5"),
    ("valid hindu marriage", "Hindu_Marriage", "5"),
    ("restitution of conjugal rights", "Hindu_Marriage", "9"),
    ("judicial separation", "Hindu_Marriage", "10"),
    ("grounds for divorce", "Hindu_Marriage", "13"),
    ("divorce", "Hindu_Marriage", "13"),
    ("maintenance pendente lite", "Hindu_Marriage", "24"),
    ("permanent alimony", "Hindu_Marriage", "25"),
    ("custody of children", "Hindu_Marriage", "26"),
    ("maintenance provision", "Hindu_Marriage", "24"),
    ("maintenance under hindu", "Hindu_Marriage", "24"),
    ("hindu marriage maintenance", "Hindu_Marriage", "24"),
    ("maintenance after divorce", "Hindu_Marriage", "25"),
    ("alimony", "Hindu_Marriage", "25"),
    # Special Marriage Act
    ("conditions for special marriage", "Special_Marriage", "4"),
    ("special marriage registration", "Special_Marriage", "4"),
    # Dowry Prohibition Act
    ("penalty for giving dowry", "Dowry_Prohibition", "3"),
    ("penalty for demanding dowry", "Dowry_Prohibition", "4"),
    ("dowry prohibition punishment", "Dowry_Prohibition", "3"),
    ("punishment for demanding dowry", "Dowry_Prohibition", "4"),
    ("punishment for giving dowry", "Dowry_Prohibition", "3"),
    # Domestic Violence Act
    ("application to magistrate", "Domestic_Violence", "12"),
    ("protection order", "Domestic_Violence", "18"),
    ("residence order", "Domestic_Violence", "19"),
    ("monetary relief", "Domestic_Violence", "20"),
    ("custody order", "Domestic_Violence", "21"),
    # Child Marriage Act
    ("punishment for child marriage", "Child_Marriage", "9"),
    ("child marriage punishment", "Child_Marriage", "9"),
    # NI Act
    ("cheque bounce", "Negotiable", "138"),
    ("dishonour of cheque", "Negotiable", "138"),
    ("section 138 ni act", "Negotiable", "138"),
    ("cheque dishonour punishment", "Negotiable", "138"),
    ("notice for cheque bounce", "Negotiable", "138"),
    # Motor Vehicles Act
    ("motor accident compensation", "Motor", "166"),
    ("accident claim", "Motor", "166"),
    ("claim petition motor", "Motor", "166"),
    ("motor accident appeal", "Motor", "173"),
    ("motor accident", "Motor", "166"),
    ("motor vehicle accident", "Motor", "166"),
    ("compensation for accident", "Motor", "166"),
    ("accident compensation", "Motor", "166"),
    ("compensation for motor accident", "Motor", "166"),
    ("motor accident claim", "Motor", "166"),
    # NDPS Act
    ("ndps offence", "Narcotic", "8"),
    ("punishment for cannabis", "Narcotic", "20"),
    ("punishment for heroin", "Narcotic", "21"),
    ("ndps consumption", "Narcotic", "27"),
    ("ndps bail", "Narcotic", "37"),
    ("possession of cannabis", "Narcotic", "20"),
    ("cannabis", "Narcotic", "20"),
    ("ganja", "Narcotic", "20"),
    ("heroin", "Narcotic", "21"),
    ("possession of drugs", "Narcotic", "20"),
    ("cannabis possession", "Narcotic", "20"),
    ("ndps cannabis", "Narcotic", "20"),
    # POCSO Act
    ("penetrative sexual assault", "pocso", "3"),
    ("pocso punishment", "pocso", "4"),
    ("aggravated penetrative", "pocso", "5"),
    ("sexual assault on child", "pocso", "7"),
    ("sexual harassment child", "pocso", "11"),
    # Arms Act
    ("illegal arms", "ArmsAct", "25"),
    ("arms act punishment", "ArmsAct", "25"),
    ("using arms", "ArmsAct", "27"),
    # PC Act
    ("bribery public servant", "Corruption", "7"),
    ("criminal misconduct", "Corruption", "13"),
    ("corruption punishment", "Corruption", "7"),
    # SC/ST Act
    ("atrocities on sc st", "Scheduled", "3"),
    ("sc st atrocities", "Scheduled", "3"),
    # Juvenile
    ("heinous offence juvenile", "Juvenile", "74"),
    ("juvenile punishment", "Juvenile", "75"),
    # Parents Act
    ("maintenance of parents", "Maintenance_and_Welfare", "4"),
    ("neglect of parents", "Maintenance_and_Welfare", "5"),
    # PESA
    ("where does pesa apply", "pesa", "3"),
    ("pesa applicability", "pesa", "3"),
    # Rent Control
    ("grounds for eviction", "Rent_Control", "11"),
    ("eviction grounds", "Rent_Control", "11"),
    ("evict tenant", "Rent_Control", "11"),
    ("habitual defaulter", "Rent_Control", "11"),
    ("eviction", "Rent_Control", "11"),
    # CG Land Revenue
    ("mutation", "Land_Revenue", "109"),
    ("partition", "Land_Revenue", "178"),
]

def detect_section_hint(query, source_filter):
    q = query.lower()
    sf = source_filter.lower() if isinstance(source_filter, str) else ""
    for phrase, src_hint, section in SECTION_HINTS:
        if phrase in q:
            if sf and src_hint.lower() not in sf:
                continue
            return section, src_hint
    return None, None

# ----------------------------
# BM25 setup
# ----------------------------
chunks = load_chunks("data/processed/chunks_v2.jsonl")
corpus = [chunk["text"] for chunk in chunks]
tokenized_corpus = [tokenize(doc) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

_bm25_cache = {}

def _bm25_key(source_filter, doc_type_filter):
    if isinstance(source_filter, (list, tuple, set)):
        sf_key = tuple(sorted(str(x) for x in source_filter))
    else:
        sf_key = str(source_filter) if source_filter else ""
    return (sf_key, doc_type_filter or "")

def get_filtered_bm25(source_filter, doc_type_filter=None):
    key = _bm25_key(source_filter, doc_type_filter)
    if key in _bm25_cache:
        return _bm25_cache[key]

    indices = []
    for i, c in enumerate(chunks):
        meta = c["metadata"]
        if source_filter and not matches_source(meta, source_filter):
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
        if source_filter and not matches_source({"source": src}, source_filter):
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

def _matching_sources_for(source_filter):
    """Return list of ALL_SOURCES strings matching source_filter (string or list)."""
    if isinstance(source_filter, (list, tuple, set)):
        out = set()
        for fragment in source_filter:
            for s in ALL_SOURCES:
                if str(fragment).lower() in s.lower():
                    out.add(s)
        return sorted(out)
    return [s for s in ALL_SOURCES if str(source_filter).lower() in s.lower()]

def dense_search(query, top_k=20, source_filter=None, doc_type_filter=None):
    query_embedding = model.encode([query])[0]
    kwargs = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }

    where_conditions = []
    if source_filter:
        matching_sources = _matching_sources_for(source_filter)
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
    # TIER 1: hints (explicit match in query)
    if source_filter is None:
        source_filter = detect_source_filter(query)

    # Judgment-mode override
    if doc_type_filter is None and query_wants_judgments(query):
        doc_type_filter = "judgment"
        source_filter = None

    target_section = extract_section_from_query(query)
    hint_source = None
    if not target_section and doc_type_filter != "judgment":
        target_section, hint_source = detect_section_hint(query, source_filter)
        if target_section and hint_source and not source_filter:
            source_filter = hint_source

    # TIER 2: classifier fallback (only if no hints fired and not judgment mode)
    if not source_filter and not target_section and doc_type_filter != "judgment":
        classifier_fragments = classifier_route(query)
        if classifier_fragments:
            source_filter = classifier_fragments  # list of fragments

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
        # Hint tier
        "What is the punishment for murder under BNS?",
        "What are the grounds for eviction under the Chhattisgarh Rent Control Act?",
        "What is the punishment for cheque bounce under Section 138 NI Act?",
        "What is the grounds for divorce under Hindu Marriage Act?",
        # Classifier tier (no act name in query)
        "How do I get a protection order against my abusive husband?",
        "What happens if a cheque I gave gets dishonoured?",
        "My landlord is threatening to throw me out without going to court.",
        "Can I file for divorce because my spouse is cruel?",
        "What are the penalties for carrying an unlicensed gun?",
        # Semantic only (no hint, no confident classifier)
        "What is the punishment for jaywalking in Chhattisgarh?",
    ]
    for q in test_queries:
        print(f"\n=== {q} ===")
        sf_hint = detect_source_filter(q)
        wants_j = query_wants_judgments(q)
        cls_frags = None if sf_hint else classifier_route(q)
        print(f"  hint_filter={sf_hint}  judgment_query={wants_j}  classifier={cls_frags}")
        results = hybrid_search(q, top_k=3)
        if not results:
            print("  [NO RESULTS]")
        for i, r in enumerate(results):
            m = r["metadata"]
            sec = m.get("section", m.get("article", m.get("para", "")))
            print(f"  [{i+1}] {m.get('source')} sec/para={sec} ({r.get('match_type')})")
            print(f"      {r['text'][:140].replace(chr(10), ' ')}...")