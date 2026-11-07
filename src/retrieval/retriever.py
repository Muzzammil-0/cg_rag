import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import chromadb
import nltk
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from src.retrieval.query_analyzer import (
    extract_act_section,
    get_fragments_for_act,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ----------------------------
# Constants
# ----------------------------
SOURCE_PATTERNS: List[Tuple[List[str], str]] = [
    (["domestic violence", "dv act"], "Domestic_Violence"),
    (["bnss", "bharatiya nagarik suraksha", "nagarik suraksha"], "Bharatiya_Nagarik"),
    (["bsa", "bharatiya sakshya", "sakshya"], "BSA"),
    (["bns", "bharatiya nyaya sanhita", "nyaya sanhita"], "BNS"),
    (["ipc", "indian penal"], "IPC_eng"),
    (["crpc", "criminal procedure"], "criminal_procedure"),
    (["cpc", "civil procedure"], "civil_procedure"),
    (["evidence act", "indian evidence", "iea"], "iea_1872"),
    (["constitution"], "constitution"),
    (["hindu marriage"], "Hindu_Marriage"),
    (["special marriage"], "Special_Marriage"),
    (["dowry prohibition", "dowry"], "Dowry_Prohibition"),
    (["child marriage", "pcma", "prohibition of child marriage"], "Child_Marriage"),
    (["maintenance and welfare of parents", "parents act"], "Maintenance_and_Welfare"),
    (["juvenile justice", "jj act", "juvenile"], "Juvenile"),
    (["contract act"], "Contract"),
    (["transfer of property", "tpa"], "Transfer_Property"),
    (["specific relief"], "Specific_Relief"),
    (["arbitration", "conciliation"], "Arbitration"),
    (["real estate", "rera"], "RERA"),
    (["consumer protection", "consumer act"], "Consumer"),
    (["insolvency", "ibc", "bankruptcy"], "IBC"),
    (["negotiable instrument", "ni act", "cheque bounce", "section 138", "cheque dishonour", "dishonour of cheque"], "Negotiable"),
    (["motor vehicle", "motor vehicles", "mv act", "motor accident", "motor claim"], "Motor"),
    (["code on wages", "minimum wages"], "Labour_Wages"),
    (["industrial relations", "industrial disputes"], "Labour_Relations"),
    (["social security"], "Labour_Security"),
    (["occupational safety", "osh code"], "Labour_OSH"),
    (["posh", "sexual harassment"], "POSH"),
    (["dpdp", "data protection"], "DPDP"),
    (["information technology", "it act"], "IT_Act"),
    (["money laundering", "pmla"], "PMLA"),
    (["prevention of corruption", "pc act", "corruption act"], "Corruption"),
    (["arms act"], "ArmsAct"),
    (["ndps", "narcotic", "psychotropic"], "Narcotic"),
    (["pocso", "protection of children from sexual"], "pocso"),
    (["sc st act", "sc/st", "scheduled castes", "atrocities act", "sc st atrocities", "sc/st atrocities"], "Scheduled"),
    (["environment protection", "epa"], "Environment"),
    (["right to information", "rti"], "RTI"),
    (["forest rights", "fra rules", "fra rule"], "FRA"),
    (["pesa"], "pesa"),
    (["land revenue", "cg land revenue"], "Land_Revenue"),
    (["excise"], "Excise"),
    (["rent control"], "Rent_Control"),
    (["municipal corporation"], "Municipal_Corp"),
    (["municipalit"], "Municipalities"),
    (["panchayat"], "Panchayat"),
    (["tendu"], "tendu"),
    (["vanopaj"], "vanopaj"),
    (["lok sewa", "lok-sewa"], "Lok-Sewa"),
    (["lokayukt", "lokayukta"], "Lokayukt"),
    (["special court", "special courts"], "Special Courts"),
    (["cattle preservation", "agricultural cattle"], "Cattle"),
    (["police act"], "CG_Police"),
    (["co-operative", "cooperative societies"], "Cooperative"),
    (["krishi upaj mandi", "mandi act"], "Mandi"),
    (["vishwavidyalaya", "university act"], "Vishwavidyalaya"),
    (["nagar tatha gram", "town planning"], "Town_Planning"),
    (["shops and establishments", "shops act"], "Shops"),
    (["industrial employment", "standing orders"], "CG_Industrial"),
]

JUDGMENT_PATTERNS: List[str] = [
    "what have courts said", "what have cg courts", "what have the courts",
    "what has the court said", "what have courts held", "what have the courts held",
    "what has the court held", "what have the high court", "what has the high court",
    "case law", "precedent", "judgment", "judgements", "judgments",
    "high court held", "high court said", "high court observed",
    "courts have dealt", "courts have interpreted", "courts have ruled",
    "court has ruled", "case law on", "judgments on", "ruling", "rulings",
    "has the court held",
]


CLASSIFIER_DIR = PROJECT_ROOT / "data" / "classifier" / "model"
CLASSIFIER_CONFIDENCE_THRESHOLD: float = 0.30

CLASSIFIER_ACT_TO_FRAGMENTS: Dict[str, List[str]] = {
    "Arms Act": ["ArmsAct"], "BNS": ["BNS"], "BNSS": ["Bharatiya_Nagarik"], "BSA": ["BSA"],
    "CG Cattle Preservation Rules": ["Cattle"], "CG Excise Act": ["Excise"], "CG Land Revenue Code": ["Land_Revenue"],
    "CG Lok Sewa Guarantee Act": ["Lok-Sewa"], "CG Municipalities Act": ["Municipalities"], "CG Panchayat Raj Act": ["Panchayat"],
    "CG Rent Control Act": ["Rent_Control"], "CG Special Courts Act": ["Special Courts"], "CG Tendu Patta Act": ["tendu"],
    "CG Vanopaj Act": ["vanopaj"], "CPC": ["civil_procedure"], "Criminal Code": ["IPC_eng", "BNS"],
    "Criminal Procedure": ["criminal_procedure", "Bharatiya_Nagarik"], "Domestic Violence Act": ["Domestic_Violence"],
    "Dowry Prohibition Act": ["Dowry_Prohibition"], "Evidence Law": ["iea_1872", "BSA"], "Forest Rights Act Rules": ["FRA"],
    "Hindu Marriage Act": ["Hindu_Marriage"], "Juvenile Justice Act": ["Juvenile"], "Marriage Law": ["Hindu_Marriage", "Special_Marriage"],
    "Motor Vehicles Act": ["Motor"], "Negotiable Instruments Act": ["Negotiable"], "PESA": ["pesa"], "POCSO Act": ["pocso"],
    "Prevention of Corruption Act": ["Corruption"], "SC/ST (PoA) Act": ["Scheduled"], "Special Marriage Act": ["Special_Marriage"],
}

_stemmer = PorterStemmer()

def load_chunks(jsonl_path: str) -> List[Dict[str, Any]]:
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def tokenize(text: str) -> List[str]:
    tokens = re.findall(r'\w+', text.lower())
    return [_stemmer.stem(t) for t in tokens]

def detect_source_filter(query: str) -> Optional[str]:
    q = query.lower()
    for keywords, source_fragment in SOURCE_PATTERNS:
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', q):
                return source_fragment
    return None

def matches_source(metadata: Dict[str, Any], source_filter: Optional[Union[str, List[str], Tuple[str, ...], set]]) -> bool:
    src = metadata.get("source", "").lower()
    if not source_filter:
        return True
    if isinstance(source_filter, (list, tuple, set)):
        return any(str(f).lower() in src for f in source_filter)
    return str(source_filter).lower() in src

def query_wants_judgments(query: str) -> bool:
    q = query.lower()
    return any(p in q for p in JUDGMENT_PATTERNS)

def extract_section_from_query(query: str) -> Optional[str]:
    q = query.lower()
    m = re.search(r'\barticle\s+(\d+[a-z]?)\b', q)
    if m: return m.group(1).upper()
    m = re.search(r'\b(?:section|sec\.?)\s+(\d+[a-z]?)\b', q)
    if m: return m.group(1).upper()
    return None


def reciprocal_rank_fusion(dense_results: List[Tuple], sparse_results: List[Tuple], k: int = 60) -> Tuple[List[str], Dict[str, float]]:
    fused_scores: Dict[str, float] = {}
    for rank, (id_, _, _, _) in enumerate(dense_results):
        fused_scores[id_] = fused_scores.get(id_, 0) + 1 / (k + rank + 1)
    for rank, (id_, _, _, _) in enumerate(sparse_results):
        fused_scores[id_] = fused_scores.get(id_, 0) + 1 / (k + rank + 1)
    sorted_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return sorted_ids, fused_scores

class ActClassifier:
    def __init__(self) -> None:
        self._bundle: Union[None, bool, Dict[str, Any]] = None

    def _load(self) -> Union[bool, Dict[str, Any]]:
        if self._bundle is not None: return self._bundle
        if not CLASSIFIER_DIR.exists():
            self._bundle = False
            return False
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            tokenizer = AutoTokenizer.from_pretrained(str(CLASSIFIER_DIR))
            model = AutoModelForSequenceClassification.from_pretrained(str(CLASSIFIER_DIR))
            model.eval()
            with open(CLASSIFIER_DIR / "act_labels.json", "r", encoding="utf-8") as f:
                labels = json.load(f)
            self._bundle = {"tokenizer": tokenizer, "model": model, "idx_to_act": {int(k): v for k, v in labels["idx_to_act"].items()}, "torch": torch}
            print("[classifier] loaded")
        except Exception as e:
            print(f"[classifier] failed to load: {e}")
            self._bundle = False
        return self._bundle

    def classify(self, query: str, threshold: float = CLASSIFIER_CONFIDENCE_THRESHOLD) -> Optional[List[str]]:
        bundle = self._load()
        if not bundle: return None
        try:
            tok, model, idx_to_act, torch = bundle["tokenizer"], bundle["model"], bundle["idx_to_act"], bundle["torch"]
            inputs = tok(query, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
                top_prob, top_idx = float(probs.max()), int(probs.argmax())
            if top_prob < threshold: return None
            act = idx_to_act.get(top_idx)
            return CLASSIFIER_ACT_TO_FRAGMENTS.get(act) if act else None
        except Exception as e:
            print(f"[classifier] inference failed: {e}")
            return None

class LegalRetriever:
    def __init__(self, chunks_data: List[Dict[str, Any]]) -> None:
        self.chunks = chunks_data
        
        # BM25 now uses the contextual text
        self._tokenized_corpus = [tokenize(self._get_contextual_text(doc)) for doc in self.chunks]
        self.bm25 = BM25Okapi(self._tokenized_corpus)
        
        self._bm25_cache: Dict[Tuple[Any, str], Tuple[Optional[BM25Okapi], List, List[int]]] = {}
        self.client = chromadb.PersistentClient(path=self._chroma_path())
        self._collection = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')  # or "cuda" if available
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')  # or "cuda" if available
        self.classifier = ActClassifier()
        self.ALL_SOURCES: List[str] = []
        self.SECTION_MAP: Dict[Tuple[str, str], List[str]] = {}
        self._initialize_metadata_maps()

    def _get_contextual_text(self, chunk: Dict[str, Any]) -> str:
        """Matches the logic used in embed.py so BM25 sees the same text."""
        meta = chunk["metadata"]
        source = meta.get("source", "").replace(".pdf", "").replace(".PDF", "").replace("_", " ")
        sec = meta.get("section", meta.get("article", meta.get("para", "")))
        if sec:
            return f"[{source}] [Section {sec}] {chunk['text']}"
        return f"[{source}] {chunk['text']}"

    @staticmethod
    def _chroma_path() -> str:
        if os.getenv("SPACE_ID"): return "/tmp/vector_db"
        return str(PROJECT_ROOT / "data" / "vector_db")

    def get_collection(self):
        if self._collection is not None: return self._collection
        try:
            self._collection = self.client.get_collection("legal_chunks")
        except Exception:
            from src.embeddings.startup_embed import build_if_needed
            build_if_needed()
            self._collection = self.client.get_collection("legal_chunks")
        return self._collection

    def _initialize_metadata_maps(self) -> None:
        self.ALL_SOURCES = sorted(set(c["metadata"].get("source", "") for c in self.chunks))
        self.SECTION_MAP = {}
        for i, c in enumerate(self.chunks):
            src = c["metadata"].get("source", "").lower()
            sec = str(c["metadata"].get("section", c["metadata"].get("article", c["metadata"].get("para", "")))).upper().rstrip('.').strip()
            if sec:
                self.SECTION_MAP.setdefault((src, sec), []).append(f"chunk_{i}")

    def _matching_sources_for(self, source_filter: Union[str, List[str], Tuple[str, ...], set]) -> List[str]:
        if isinstance(source_filter, (list, tuple, set)):
            out = set()
            for fragment in source_filter:
                for s in self.ALL_SOURCES:
                    if str(fragment).lower() in s.lower(): out.add(s)
            return sorted(out)
        return [s for s in self.ALL_SOURCES if str(source_filter).lower() in s.lower()]

    @staticmethod
    def _bm25_key(source_filter, doc_type_filter) -> Tuple[Any, str]:
        if isinstance(source_filter, (list, tuple, set)): sf_key = tuple(sorted(str(x) for x in source_filter))
        else: sf_key = str(source_filter) if source_filter else ""
        return (sf_key, doc_type_filter or "")

    def _get_filtered_bm25(self, source_filter, doc_type_filter=None):
        key = self._bm25_key(source_filter, doc_type_filter)
        if key in self._bm25_cache: return self._bm25_cache[key]
        indices = []
        for i, c in enumerate(self.chunks):
            meta = c["metadata"]
            if source_filter and not matches_source(meta, source_filter): continue
            if doc_type_filter and meta.get("doc_type") != doc_type_filter: continue
            indices.append(i)
        if not indices:
            self._bm25_cache[key] = (None, [], [])
            return self._bm25_cache[key]
        filtered_corpus = [self._tokenized_corpus[i] for i in indices]
        filtered_bm25 = BM25Okapi(filtered_corpus)
        result = (filtered_bm25, [self.chunks[i] for i in indices], indices)
        if len(self._bm25_cache) >= 64:
            self._bm25_cache.clear()
        self._bm25_cache[key] = result
        return result

    def find_chunks_by_section(self, source_filter, target_section: Optional[str], doc_type_filter: Optional[str] = None) -> List[str]:
        if not target_section: return []
        target = target_section.upper().rstrip('.').strip()
        matching_ids: List[str] = []
        for (src, sec), ids in self.SECTION_MAP.items():
            if sec != target: continue
            if source_filter and not matches_source({"source": src}, source_filter): continue
            for cid in ids:
                if doc_type_filter:
                    try:
                        idx = int(cid.replace("chunk_", ""))
                        if self.chunks[idx]["metadata"].get("doc_type") != doc_type_filter: continue
                    except (ValueError, IndexError): continue
                matching_ids.append(cid)
        return matching_ids

    def dense_search(self, query: str, top_k: int = 20, source_filter=None, doc_type_filter: Optional[str] = None) -> List[Tuple]:
        query_embedding = self.model.encode([query])[0]
        kwargs: Dict[str, Any] = {"query_embeddings": [query_embedding.tolist()], "n_results": top_k, "include": ["documents", "metadatas", "distances"]}
        where_conditions: List[Dict[str, Any]] = []
        if source_filter:
            matching_sources = self._matching_sources_for(source_filter)
            if not matching_sources: return []
            if len(matching_sources) == 1: where_conditions.append({"source": matching_sources[0]})
            else: where_conditions.append({"source": {"$in": matching_sources}})
        if doc_type_filter: where_conditions.append({"doc_type": doc_type_filter})
        if len(where_conditions) == 1: kwargs["where"] = where_conditions[0]
        elif len(where_conditions) > 1: kwargs["where"] = {"$and": where_conditions}
        results = self.get_collection().query(**kwargs)
        return list(zip(results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]))[:top_k]

    def sparse_search(self, query: str, top_k: int = 20, source_filter=None, doc_type_filter: Optional[str] = None) -> List[Tuple]:
        tokenized_query = tokenize(query)
        if source_filter or doc_type_filter:
            filtered_bm25, filtered_chunks, original_indices = self._get_filtered_bm25(source_filter, doc_type_filter)
            if filtered_bm25 is None: return []
            scores = filtered_bm25.get_scores(tokenized_query)
            top_local = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            return [(f"chunk_{original_indices[i]}", filtered_chunks[i]["text"], filtered_chunks[i]["metadata"], scores[i]) for i in top_local]
        else:
            scores = self.bm25.get_scores(tokenized_query)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            return [(f"chunk_{i}", self.chunks[i]["text"], self.chunks[i]["metadata"], scores[i]) for i in top_indices]

    def hybrid_search(self, query: str, top_k: int = 5, source_filter=None, doc_type_filter: Optional[str] = None, rerank: bool = True, exhaustive: bool = False) -> List[Dict[str, Any]]:
        
        # NO HINTS. We rely purely on the neural network.
        if not exhaustive:
            if source_filter is None: source_filter = detect_source_filter(query)
            if doc_type_filter is None and query_wants_judgments(query):
                doc_type_filter = "judgment"
                source_filter = None
            
            if not source_filter and doc_type_filter != "judgment":
                classifier_fragments = self.classifier.classify(query)
                if classifier_fragments: source_filter = classifier_fragments

            if not source_filter and doc_type_filter != "judgment":
                llm_act, _ = extract_act_section(query)
                if llm_act:
                    frags = get_fragments_for_act(llm_act)
                    if frags: source_filter = frags
            
        return self._execute_search_pipeline(query, top_k, source_filter, doc_type_filter, rerank, exhaustive)

    def _execute_search_pipeline(self, query: str, top_k: int, source_filter, doc_type_filter: Optional[str], rerank: bool, exhaustive: bool) -> List[Dict[str, Any]]:
        final_results: List[Dict[str, Any]] = []
        seen_ids: set = set()

        # 1. FETCH CANDIDATES
        search_depth = 100 if exhaustive else 40
        dense_results = self.dense_search(query, top_k=search_depth, source_filter=source_filter, doc_type_filter=doc_type_filter)
        sparse_results = self.sparse_search(query, top_k=search_depth, source_filter=source_filter, doc_type_filter=doc_type_filter)
        
        sorted_ids, fused_scores = reciprocal_rank_fusion(dense_results, sparse_results)
        id_to_chunk = {f"chunk_{i}": self.chunks[i] for i in range(len(self.chunks))}
        
        general_candidates: List[Dict[str, Any]] = []
        candidate_limit = 100 if exhaustive else 30
        for cid in sorted_ids:
            if cid in seen_ids: continue
            c = id_to_chunk.get(cid)
            if c: general_candidates.append({"id": cid, **c})
            if len(general_candidates) >= candidate_limit: break

        # 2. STRATIFIED RERANKING (50/50 Split)
        if rerank and general_candidates:
            acts = [c for c in general_candidates if c["metadata"].get("doc_type") != "judgment"]
            judgments = [c for c in general_candidates if c["metadata"].get("doc_type") == "judgment"]

            reranked_acts = []
            if acts:
                pairs = [(query, self._get_contextual_text(c)) for c in acts]
                scores = self.reranker.predict(pairs)
                reranked_acts = sorted(zip(acts, scores), key=lambda x: x[1], reverse=True)

            reranked_judgments = []
            if judgments:
                pairs = [(query, self._get_contextual_text(c)) for c in judgments]
                scores = self.reranker.predict(pairs)
                reranked_judgments = sorted(zip(judgments, scores), key=lambda x: x[1], reverse=True)

            # 50% ACTS, 50% JUDGMENTS
            act_slots = min(top_k, max(2, int(top_k * 0.5)))
            judgment_slots = max(0, top_k - act_slots)

            for c, s in reranked_acts[:act_slots]:
                final_results.append({"id": c["id"], "text": c["text"], "metadata": c["metadata"], "fused_score": float(s), "match_type": "act_reranked"})
            
            for c, s in reranked_judgments[:judgment_slots]:
                final_results.append({"id": c["id"], "text": c["text"], "metadata": c["metadata"], "fused_score": float(s), "match_type": "judgment_reranked"})

            if len(final_results) < top_k:
                for c, s in reranked_acts[act_slots:] + reranked_judgments[judgment_slots:]:
                    if len(final_results) >= top_k: break
                    final_results.append({"id": c["id"], "text": c["text"], "metadata": c["metadata"], "fused_score": float(s), "match_type": "fallback_reranked"})

        else:
            for c in general_candidates[:top_k]:
                final_results.append({"id": c["id"], "text": c["text"], "metadata": c["metadata"], "fused_score": fused_scores.get(c["id"], 0.0), "match_type": "rrf"})

        return final_results[:top_k]

# ------------------------------------------------------------------
# BRIDGE FUNCTION FOR generator.py
# ------------------------------------------------------------------
_global_retriever = None
_retriever_lock = threading.Lock()

def hybrid_search(query: str, top_k: int = 5, source_filter=None, doc_type_filter: Optional[str] = None, rerank: bool = True, exhaustive: bool = False):
    global _global_retriever
    if _global_retriever is None:
        with _retriever_lock:
            if _global_retriever is None:
                chunks_path = str(PROJECT_ROOT / "data" / "processed" / "chunks_v2.jsonl")
                _global_retriever = LegalRetriever(load_chunks(chunks_path))

    return _global_retriever.hybrid_search(query, top_k=top_k, source_filter=source_filter, doc_type_filter=doc_type_filter, rerank=rerank, exhaustive=exhaustive)