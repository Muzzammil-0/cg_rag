import os
import json
import re
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Dict, List, Tuple, Optional
from groq import Groq

# --- 1. PATH SETUP ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# --- 2. BULLETPROOF ENV LOADING ---
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

try:
    from src.retrieval.retriever import hybrid_search
except ImportError as e:
    print(f"[FATAL] Error importing required module: {e}")
    raise SystemExit("Could not load retriever module. Check project structure.")

client = None
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print(f"[FATAL ERROR] GROQ_API_KEY is still not loading!")
    sys.exit(1)

try:
    client = Groq(api_key=api_key)
    print("[INFO] Groq Client initialized successfully.")
except Exception as e:
    print(f"[FATAL ERROR] Failed to initialize Groq client: {e}")
    sys.exit(1)

CHUNK_CHAR_LIMIT: int = 1200
MAX_RETRIES: int = 3
BACKOFF_BASE: int = 5
DEFAULT_MODEL: str = "openai/gpt-oss-120b" 

ParsedData = Dict[str, Any]
RetrievalResults = List[Dict[str, Any]]

# ------------------------------------------------------------
# Helper Utilities
# ------------------------------------------------------------
def call_groq_with_retry(model: str, messages: List[Dict], temperature: float = 0.0, max_tokens: int = 1024) -> Optional[Any]:
    if client is None: 
        print("[ERROR] Cannot call Groq API: Client not initialized.")
        return None
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        except Exception as e:
            last_error = e
            err_str = str(e)
            is_rate_limit = ("rate_limit_exceeded" in err_str or "Request too large" in err_str or "413" in err_str or "429" in err_str)
            if is_rate_limit and attempt < MAX_RETRIES - 1:
                wait = BACKOFF_BASE * (2 ** attempt)
                print(f"[INFO] Rate limit hit. Attempt {attempt + 1}/{MAX_RETRIES} — waiting {wait}s...")
                time.sleep(wait)
                continue
            break
    print(f"[ERROR] Groq API call failed after {MAX_RETRIES} attempts: {last_error}")
    return None

def extract_json(text: str) -> Optional[Dict]:
    text = re.sub(r'<think>.*?(?:</think>|$)', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'```(?:json)?\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    matches = list(re.finditer(r'\{.*\}', text, re.DOTALL))
    if matches:
        for match in reversed(matches):
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
    return None

def validate_citations(parsed: ParsedData, results: RetrievalResults) -> ParsedData:
    valid_citations = []
    if parsed and "citations" in parsed and isinstance(parsed["citations"], list):
        for c in parsed["citations"]:
            try:
                c_int = int(c)
                if 1 <= c_int <= len(results): valid_citations.append(c_int)
            except ValueError: pass
    parsed["citations"] = valid_citations
    return parsed

def _normalize_for_match(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s) 
    return re.sub(r'\s+', ' ', s).strip()

def _quote_in_chunk(quote: str, chunk_text: str) -> bool:
    q, c = _normalize_for_match(quote), _normalize_for_match(chunk_text)
    if not q or not c: return False
    if len(q) < 30: return q in c
    q_words = q.split()
    ngram_len = max(3, min(4, len(q_words)))
    for i in range(len(q_words) - ngram_len + 1):
        if " ".join(q_words[i:i + ngram_len]) in c: return True
    return False

def _normalize_source(s: str) -> str:
    s = str(s).lower().replace(".pdf", "")
    return re.sub(r'[^a-z0-9]', '', s)

def verify_provision_quotes(parsed: ParsedData, results: RetrievalResults, strict: bool = True) -> Tuple[ParsedData, int]:
    if not parsed or not isinstance(parsed.get("provisions"), list): return parsed, 0
    # Map section number -> list of (normalized source, text) so a quote is checked
    # against the cited act, not just any act that shares the same section number.
    section_to_chunks: Dict[str, List[Tuple[str, str]]] = {}
    for r in results:
        meta = r["metadata"]
        raw_sec = meta.get("section") or meta.get("article") or meta.get("para") or ""
        m = re.match(r'(\d+[A-Za-z]*)', str(raw_sec).strip(), re.IGNORECASE)
        if m:
            section_key = m.group(1).lower()
            section_to_chunks.setdefault(section_key, []).append(
                (_normalize_source(meta.get("source", "")), r["text"])
            )

    kept, dropped = [], []
    for p in parsed.get("provisions", []):
        quote = str(p.get("quote", "")).strip()
        section_raw = str(p.get("section", "")).strip()
        m = re.match(r'(?:section\s+)?(\d+[A-Za-z]*)', section_raw, re.IGNORECASE)
        section_key = m.group(1).lower() if m else None

        if not quote or not section_key:
            dropped.append((p, "missing required data")); continue
        candidates = section_to_chunks.get(section_key)
        if not candidates:
            dropped.append((p, "no matching context chunk")); continue

        prov_src = _normalize_source(p.get("source", ""))
        same_act = [text for src, text in candidates
                    if prov_src and src and (prov_src in src or src in prov_src)]
        # Fall back to all same-section chunks when the cited source name matches no
        # retrieved source, preserving prior behavior for inconsistent naming.
        matching_texts = same_act or [text for _, text in candidates]
        chunk_text = "\n\n[CONTEXT APPENDED]\n".join(matching_texts)

        if _quote_in_chunk(quote, chunk_text):
            p["quote_verified"] = True
            kept.append(p)
        else:
            if strict: dropped.append((p, "fabricated quote"))
            else:
                p["quote_verified"] = False
                kept.append(p)

    parsed["provisions"] = kept
    return parsed, len(dropped)

def regenerate_narrative(parsed: ParsedData, fact_pattern: str, results: RetrievalResults, model: str) -> ParsedData:
    if not parsed or not parsed.get("provisions"): return parsed
    verified_lines = [f"[{p.get('citations')} ({p.get('source')} {p.get('section')})]: \"{p.get('quote')}\"" for p in parsed["provisions"]]
    verified_block = "\n".join(verified_lines)

    system_msg = (
        "You are a legal reasoning editor. Rewrite the 'likely_reasoning' and 'possible_outcomes' "
        "sections based ONLY on provided verified provisions. Do NOT invent statutes.\n\n"
        f"VERIFIED PROVISIONS:\n---\n{verified_block}\n---"
    )
    user_msg = (
        f"Fact Pattern: {fact_pattern}\n\n"
        f"Draft: {parsed.get('likely_reasoning', '')}\n\n"
        "Output ONLY a JSON object with keys 'likely_reasoning' (string) and 'possible_outcomes' (list of objects with 'scenario', 'likelihood', 'rationale')."
    )

    response = call_groq_with_retry(model=model, messages=[
        {"role": "system", "content": system_msg}, 
        {"role": "user", "content": user_msg}
    ], temperature=0.1, max_tokens=2000)
    
    if response:
        rewritten = extract_json(response.choices[0].message.content)
        if rewritten:
            parsed["likely_reasoning"] = rewritten.get("likely_reasoning", "")
            parsed["possible_outcomes"] = rewritten.get("possible_outcomes", [])
    return parsed

def validate_reasoning_provisions(parsed: ParsedData, results: RetrievalResults) -> ParsedData:
    if not parsed or not isinstance(parsed.get("provisions"), list): return parsed
    present_section_keys = set()
    for r in results:
        raw_sec = r["metadata"].get("section") or r["metadata"].get("article") or r["metadata"].get("para") or ""
        m = re.match(r'(\d+[A-Za-z]*)', str(raw_sec).strip(), re.IGNORECASE)
        if m: present_section_keys.add(m.group(1).lower())

    kept = []
    for p in parsed["provisions"]:
        sec_raw = str(p.get("section", "")).strip()
        if not sec_raw: kept.append(p); continue
        m = re.match(r'(?:section\s+)?(\d+[A-Za-z]*)', sec_raw, re.IGNORECASE)
        if m and m.group(1).lower() in present_section_keys: kept.append(p)
    parsed["provisions"] = kept
    return parsed

# ------------------------------------------------------------
# Context & Querying Tools
# ------------------------------------------------------------
def build_context(results: RetrievalResults, char_limit: int = CHUNK_CHAR_LIMIT) -> str:
    parts = []
    for i, res in enumerate(results):
        meta = res["metadata"]
        section = meta.get("article") or meta.get("section") or meta.get("para") or ""
        snippet = res["text"][:char_limit].replace("\n", " ")
        parts.append(f"[{i+1}] Source: {meta.get('source', 'unknown')}, Section/Article: {section}\n{snippet}")
    return "\n\n".join(parts)

def extract_legal_issues(fact_pattern: str, model: str = DEFAULT_MODEL) -> List[str]:
    system_msg = (
        "You are an expert legal analyst for Indian and Chhattisgarh law. "
        "Extract the specific legal issues from the fact pattern. "
        "Return a JSON object with key 'issues' containing 3-6 concise phrases. "
        "Each phrase MUST be phrased as '<legal concept> under <act>' (e.g., 'wrongful termination under Industrial Disputes Act'). "
        "Output ONLY the raw JSON object. Do not include any introductory text, markdown fences, or explanations."
    )
    user_msg = f"Fact pattern:\n{fact_pattern}\n\nOutput the JSON object now."

    response = call_groq_with_retry(model=model, messages=[
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ], temperature=0.1, max_tokens=600)

    if response:
        content = response.choices[0].message.content
        parsed = extract_json(content)
        if parsed and isinstance(parsed.get("issues"), list):
            issues = [str(x).strip() for x in parsed["issues"] if str(x).strip()]
            print(f"[DEBUG] Issues extracted: {issues}")
            return issues
            
    print("[WARNING] Failed to extract issues. Falling back to full fact pattern.")
    return []
# ------------------------------------------------------------
# Mode 1: Retrieval + Answer (Q&A)
# ------------------------------------------------------------
def generate_answer(question: str, top_k: int = 20, model: str = DEFAULT_MODEL, verbose: bool = True) -> Tuple[Optional[ParsedData], Optional[RetrievalResults]]:
    print(f"\n[DEBUG] Searching for: {question} (EXHAUSTIVE MODE)")
    # WE SET exhaustive=True HERE
    results = hybrid_search(question, top_k=top_k, exhaustive=True)
    


    print("\n--- CHUNKS FED TO LLM ---")
    for i, r in enumerate(results):
        print(f"[{i+1}] {r['metadata'].get('source')} | Section: {r['metadata'].get('section')} | Match: {r.get('match_type')}")
    print("-------------------------\n")
    print(f"[DEBUG] Found {len(results)} chunks after exhaustive reranking.")
    
    context = build_context(results)
    print(f"[DEBUG] Context length: {len(context)} chars.")

    system_msg = (
        "You are a precise legal research assistant. Use ONLY the provided context to answer. "
        "If not found, respond with: 'Not found in provided context.' "
        "Output JSON with keys: answer (string), citations (list of integers), confidence (float)."
    )
    
    print(f"[DEBUG] Calling Groq API with model: {model}...")
    response = call_groq_with_retry(model=model, messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}], temperature=0.0, max_tokens=1500)
    
    content = response.choices[0].message.content if response else ""
    if verbose: 
        print(f"[DEBUG] Raw Output from Groq:\n{content}\n")

    parsed = extract_json(content)
    if not isinstance(parsed, dict):
        print("[DEBUG] extract_json failed to find a JSON object.")
        parsed = {"answer": content.strip() if content else "Error.", "citations": [], "confidence": 0.1}
    else:
        parsed = validate_citations(parsed, results)

    return parsed, results

# ------------------------------------------------------------
# Mode 2: Reasoning with Query Expansion + Regeneration
# ------------------------------------------------------------
def generate_reasoning(fact_pattern: str, top_k: int = 15, model: str = DEFAULT_MODEL, verbose: bool = True) -> Tuple[Optional[ParsedData], Optional[RetrievalResults]]:
    print(f"\n[DEBUG] Extracting legal issues for fact pattern...")
    legal_issues = extract_legal_issues(fact_pattern, model=model)
    
    # --- MULTI-QUERY RETRIEVAL ---
    print(f"[DEBUG] Running Multi-Query Retrieval for {len(legal_issues) + 1} distinct queries...")
    
    all_results = []
    seen_ids = set()
    
    # Combine the full fact pattern and the extracted issues into a list of queries
    queries = [fact_pattern]
    if legal_issues:
        queries.extend(legal_issues)
        
    for q in queries:
        # Search the database for each specific query
        res = hybrid_search(q, top_k=4, exhaustive=True)
        for r in res:
            if r["id"] not in seen_ids:
                all_results.append(r)
                seen_ids.add(r["id"])
                
    # Sort the combined results by their retrieval score
    all_results = sorted(all_results, key=lambda x: x.get("fused_score", 0), reverse=True)
    
    # Take the top_k best chunks from the combined pool
    results = all_results[:top_k]
    print(f"[DEBUG] Combined {len(all_results)} total chunks from all queries. Kept top {len(results)}.")
    
    if not results: return None, None

    context = build_context(results)
    print(f"[DEBUG] Context length: {len(context)} chars.")

    system_msg_pass1 = (
        "You are simulating how a Chhattisgarh court might reason through a legal question. "
        "Use ONLY the provided context. Do not invent provisions. "
        "Note: The context may contain laws from multiple acts (e.g., IPC, DPDP Act, Labour Codes). "
        "Prioritize specific acts over general codes when both apply. "
        "Return a JSON object with EXACT keys: "
        "issues (list), provisions (list of objects: source, section, relevance, quote), "
        "petitioner_arguments (list), respondent_arguments (list), likely_reasoning (string), "
        "possible_outcome (string), confidence (float), citations (list of context integers). "
        "CRITICAL: Output ONLY the raw JSON object. Do NOT wrap it in markdown code fences (```json). Do NOT include any conversational text."
    )
    
    print(f"[DEBUG] Calling Groq API for reasoning with model: {model}...")
    response = call_groq_with_retry(model=model, messages=[
        {"role": "system", "content": system_msg_pass1}, 
        {"role": "user", "content": f"Fact pattern:\n{fact_pattern}\n\nContext:\n{context}"}
    ], temperature=0.2, max_tokens=3000)

    if not response: 
        print("[DEBUG] Groq API returned None.")
        return None, results
        
    content = response.choices[0].message.content
    if verbose: 
        print(f"[DEBUG] Raw Output from Groq:\n{content}\n")
    
    initial_parsed = extract_json(content)

    # 1. FIRST: Verify it's a valid dictionary
    if not initial_parsed or not isinstance(initial_parsed, dict):
        print("[DEBUG] extract_json failed to find a valid JSON object.")
        return None, results

    # 2. SECOND: Add the fallback for possible_outcomes
    if "possible_outcomes" not in initial_parsed:
        # Fallback if the LLM forgot to generate the ranked outcomes
        initial_parsed["possible_outcomes"] = [{
            "likelihood": "Most Likely", 
            "scenario": initial_parsed.get("possible_outcome", "See reasoning above."), 
            "rationale": "Based on the statutory provisions and factual matrix."
        }]

    # 3. THIRD: Run your validations
    parsed = validate_citations(initial_parsed, results)
    parsed = validate_reasoning_provisions(parsed, results)
    
    parsed, dropped_count = verify_provision_quotes(parsed, results, strict=True)
    if dropped_count > 0 and parsed:
        print(f"[ACTION] Fabrication detected ({dropped_count} dropped). Regenerating...")
        parsed = regenerate_narrative(parsed, fact_pattern, results, model=model)

    return parsed, results

if __name__ == "__main__":
    print("=" * 60)
    print("           CG-LEGAL RAG GENERATOR SERVICE")
    print("=" * 60)
    mode_choice = input("Select Mode (query for Q&A, reason for Reasoning): ").strip().lower()

    if mode_choice == "query":
        question = input("\nEnter the legal question you want to ask: ")
        answer_data, retrieved = generate_answer(question, top_k=8)
        
        print("\n" + "=" * 20 + " FINAL ANSWER REPORT " + "=" * 20)
        if answer_data:
            print(f"\n✅ Answer:\n{answer_data.get('answer', 'N/A')}")
    elif mode_choice == "reason":
        pattern = input("\nEnter the detailed fact pattern for reasoning simulation: ")
        result, retrieved = generate_reasoning(pattern, top_k=15)
        
        if result:
            print("\n[1] Legal Issues Identified:")
            for issue in result.get("issues", []): print(f"  - {issue}")
            print("\n[2] Validated & Relevant Provisions Used:")
            for i, prov in enumerate(result.get("provisions", [])):
                verified_status = "✓" if prov.get("quote_verified") else "?"
                print(f"  [{verified_status}] {prov.get('source')} {prov.get('section')}: {prov.get('relevance')}")
            print("\n[3] Likely Reasoning:\n", result.get('likely_reasoning', 'N/A'))
            print("\n[4] Possible Outcome:\n", result.get('possible_outcome', 'N/A'))
        else:
            print("\n[SYSTEM] Reasoning failed to produce a structured output.")
    else:
        print("Invalid mode selected. Exiting.")