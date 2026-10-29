"""
Generate (query, act, section) training triplets from the chunk corpus
using distant supervision via the Groq LLM.
"""
import os
import re
import sys
import json
import time
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Set

try:
    from dotenv import load_dotenv
except ImportError:
    print("Error: python-dotenv is required.")
    sys.exit(1)
from groq import Groq

def setup_environment():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable is not set.")
        sys.exit(1)
    return Groq(api_key=api_key), Path(__file__).resolve().parents[2]

CLIENT, REPO_ROOT = setup_environment()

CHUNKS_PATH = REPO_ROOT / "data/processed/chunks_v2.jsonl"
OUTPUT_PATH = REPO_ROOT / "data/classifier/training_data.jsonl"
PROGRESS_PATH = REPO_ROOT / "data/classifier/progress.json"

MODEL = "openai/gpt-oss-20b"
BATCH_SIZE = 3
LLM_DELAY_SECONDS = 8
MIN_CHUNK_LEN = 250
MAX_CHUNK_LEN = 1200

ACT_LABEL_MAP = [
    # Central Acts
    ("Bharatiya_Nagarik", "BNSS"), ("IPC_eng", "IPC"), ("BNS", "BNS"), ("BSA", "BSA"),
    ("criminal_procedure", "CrPC"), ("civil_procedure", "CPC"), ("iea_1872", "Indian Evidence Act"),
    ("Hindu_Marriage", "Hindu Marriage Act"), ("Special_Marriage", "Special Marriage Act"),
    ("Dowry_Prohibition", "Dowry Prohibition Act"), ("Domestic_Violence", "Domestic Violence Act"),
    ("Child_Marriage", "Prohibition of Child Marriage Act"), ("Negotiable", "Negotiable Instruments Act"),
    ("Motor", "Motor Vehicles Act"), ("Narcotic", "NDPS Act"), ("pocso", "POCSO Act"), ("ArmsAct", "Arms Act"),
    ("Corruption", "Prevention of Corruption Act"), ("Scheduled", "SC/ST (PoA) Act"), ("Juvenile", "Juvenile Justice Act"),
    ("Maintenance_and_Welfare", "Maintenance of Parents Act"), ("pesa", "PESA"), ("FRA", "Forest Rights Act Rules"),
    ("constitution", "Constitution"),
    
    # New Commercial & Civil
    ("Contract", "Indian Contract Act"), ("Transfer_Property", "Transfer of Property Act"),
    ("Specific_Relief", "Specific Relief Act"), ("Arbitration", "Arbitration and Conciliation Act"),
    ("RERA", "Real Estate (Regulation and Development) Act"), ("Consumer", "Consumer Protection Act"),
    ("IBC", "Insolvency and Bankruptcy Code"),
    
    # New Labour
    ("Labour_Wages", "Code on Wages"), ("Labour_Relations", "Industrial Relations Code"),
    ("Labour_Security", "Code on Social Security"), ("Labour_OSH", "Occupational Safety Code"),
    ("POSH", "POSH Act"),
    
    # New Tech & Cyber
    ("DPDP", "Digital Personal Data Protection Act"), ("IT_Act", "Information Technology Act"),
    ("PMLA", "Prevention of Money Laundering Act"),
    
    # New Env & Admin
    ("Environment", "Environment Protection Act"), ("RTI", "Right to Information Act"),

    # Chhattisgarh Acts
    ("Land_Revenue", "CG Land Revenue Code"), ("Excise", "CG Excise Act"), ("Rent_Control", "CG Rent Control Act"),
    ("Municipal_Corp", "CG Municipal Corporation Act"), ("Municipalities", "CG Municipalities Act"),
    ("Panchayat", "CG Panchayat Raj Act"), ("tendu", "CG Tendu Patta Act"), ("vanopaj", "CG Vanopaj Act"),
    ("Lok-Sewa", "CG Lok Sewa Guarantee Act"), ("Lokayukt", "CG Lokayukta Act"), 
    ("Special Courts", "CG Special Courts Act"), ("Cattle", "CG Cattle Preservation Rules"),
    ("CG_Police", "CG Police Act"), ("Cooperative", "CG Co-operative Societies Act"),
    ("Mandi", "CG Krishi Upaj Mandi Act"), ("Vishwavidyalaya", "CG University Act"),
    ("Town_Planning", "CG Town and Country Planning Act"), ("Shops", "CG Shops and Establishments Act"),
    ("CG_Industrial", "CG Industrial Employment Act"),
]

def canonical_act(source_file: str) -> str | None:
    s = source_file.lower()
    for fragment, label in ACT_LABEL_MAP:
        if fragment.lower() in s: return label
    return None

SYSTEM_PROMPT = """You are generating training data for a legal query classifier.

Given several legal text excerpts from different sections of Indian and Chhattisgarh acts, generate exactly 3 natural-language questions for each excerpt.

Requirements:
- Each question must be answerable by the provided excerpt ONLY.
- Vary phrasing across the 3 questions:
  - Question 1: formal, using precise legal terminology.
  - Question 2: casual, phrased like a layperson asking for advice.
  - Question 3: specific and factual, focusing on details.
- DO NOT mention the act name in the question.
- Only include the section number if it is critical to the natural phrasing.
- Return ONLY a JSON object with this structure:
{
  "items": [
    {"chunk_index": 0, "questions": ["...", "...", "..."]}
  ]
}
Do not include any explanatory text, code fences, or anything outside the JSON structure."""

def build_user_prompt(batch: List[Dict[str, Any]]) -> str:
    lines = []
    for i, chunk in enumerate(batch):
        sec = chunk.get("section") or "N/A"
        text = chunk["text"][:MAX_CHUNK_LEN].replace("\n", " ").strip()
        lines.append(f"--- Chunk {i} (Section {sec}, Act: {chunk['act']}) ---\n{text}\n")
    return "\n".join(lines) + "\nGenerate exactly 3 questions per chunk based on the above text. Return ONLY the structured JSON object."

def extract_json(text: str) -> dict | None:
    text = re.sub(r'<think>.*?(?:</think>|$)', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'```(?:json)?\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    try: return json.loads(text)
    except json.JSONDecodeError: pass
    start = text.find('{')
    if start == -1: return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{': depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                try: return json.loads(text[start:i + 1])
                except json.JSONDecodeError: continue
    truncated = text[start:]
    open_braces = truncated.count('{') - truncated.count('}')
    open_brackets = truncated.count('[') - truncated.count(']')
    if open_braces > 0 or open_brackets > 0:
        repaired = truncated
        if repaired.endswith('"'): repaired = repaired[:-1]
        final_repair = repaired + (']' * open_brackets) + ('}' * open_braces)
        try: return json.loads(final_repair)
        except json.JSONDecodeError: pass
    return None

def call_llm(messages: List[Dict[str, str]], max_retries: int = 3) -> str | None:
    for attempt in range(max_retries):
        try:
            response = CLIENT.chat.completions.create(model=MODEL, messages=messages, temperature=0.7, max_tokens=2000)
            return response.choices[0].message.content
        except Exception as e:
            err = str(e)
            is_rate = any(x in err for x in ("rate_limit", "429"))
            if is_rate and attempt < max_retries - 1:
                wait = 15 * (attempt + 1)
                print(f"    [RATE LIMIT] Waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"    [LLM ERROR] {err[:200]}...")
            return None
    return None

def load_act_chunks() -> List[Dict[str, Any]]:
    chunks = []
    try:
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    c = json.loads(line)
                    meta = c.get("metadata")
                    if not meta: continue
                    doc_type = meta.get("doc_type")
                    if doc_type in ["judgment", "constitution"]: continue
                    act = canonical_act(meta.get("source", ""))
                    if not act: continue
                    sec = meta.get("section")
                    if not sec: continue
                    text = c["text"]
                    if len(text) < MIN_CHUNK_LEN: continue
                    chunks.append({"chunk_id": c.get("id") or f"{meta.get('source')}_{sec}", "source_file": meta.get("source"), "act": act, "section": sec, "text": text})
                except json.JSONDecodeError: print(f"Warning: Skipping malformed JSON line.")
    except FileNotFoundError:
        print(f"Critical Error: Chunk file not found at {CHUNKS_PATH}.")
        return []
    return chunks

def stratified_sample(chunks: List[Dict[str, Any]], per_act: int) -> List[Dict[str, Any]]:
    random.seed(42)
    by_act = defaultdict(list)
    for c in chunks: by_act[c["act"]].append(c)
    sampled = []
    for act, items in sorted(by_act.items()):
        random.shuffle(items)
        sampled.extend(items[:per_act])
    random.shuffle(sampled)
    return sampled

def load_progress() -> Set[str]:
    if PROGRESS_PATH.exists():
        try:
            with open(PROGRESS_PATH, "r", encoding="utf-8") as f: return set(json.load(f))
        except json.JSONDecodeError: pass
    return set()

def save_progress(processed_ids: Set[str]):
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f: json.dump(list(processed_ids), f, indent=2)

def run_generation(args) -> None:
    all_chunks = load_act_chunks()
    if not all_chunks: return
    sampled = stratified_sample(all_chunks, args.chunks_per_act)
    processed = load_progress()
    if processed:
        sampled = [c for c in sampled if c["chunk_id"] not in processed]

    total_written = 0
    total_failed = 0

    try:
        with open(OUTPUT_PATH, "a", encoding="utf-8") as output_file:
            batches = [sampled[i:i + args.batch_size] for i in range(0, len(sampled), args.batch_size)]
            for bi, batch in enumerate(batches):
                print(f"\n>>> Batch {bi+1}/{len(batches)} | {len(batch)} chunks <<<")
                user_prompt = build_user_prompt(batch)
                content = call_llm([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}])
                if not content:
                    total_failed += 1
                    time.sleep(LLM_DELAY_SECONDS)
                    continue
                parsed = extract_json(content)
                if not parsed or "items" not in parsed:
                    total_failed += 1
                    time.sleep(LLM_DELAY_SECONDS)
                    continue

                records_written = 0
                for item in parsed["items"]:
                    idx = item.get("chunk_index")
                    questions = item.get("questions", [])
                    if not isinstance(idx, int) or not (0 <= idx < len(batch)): continue
                    chunk = batch[idx]
                    for qi, q in enumerate(questions[:3]):
                        if not isinstance(q, str) or len(q.strip()) < 8: continue
                        record = {"query": q.strip(), "act": chunk["act"], "section": chunk["section"], "source_file": chunk["source_file"], "chunk_id": chunk["chunk_id"], "question_type": ["formal", "casual", "specific"][qi]}
                        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        records_written += 1

                for c in batch: processed.add(c["chunk_id"])
                total_written += records_written
                output_file.flush()
                save_progress(processed)
                print(f"    [SUCCESS] Wrote {records_written} examples.")
                time.sleep(LLM_DELAY_SECONDS)
    finally:
        print(f"\n✅ Done. Total written: {total_written}, Failed batches: {total_failed}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-per-act", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if args.fresh:
        OUTPUT_PATH.unlink(missing_ok=True)
        PROGRESS_PATH.unlink(missing_ok=True)
    run_generation(args)

if __name__ == "__main__":
    main()