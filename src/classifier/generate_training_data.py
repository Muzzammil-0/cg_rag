"""
Generate (query, act, section) training triplets from the chunk corpus
using distant supervision via the Groq LLM.

The idea:
  - Sample chunks across all acts.
  - Ask the LLM to write 3 natural questions per chunk.
  - Tag each question with the chunk's act and section as ground truth.

Output:
  data/classifier/training_data.jsonl

Usage:
  python src/classifier/generate_training_data.py --chunks-per-act 15
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

from dotenv import load_dotenv
from groq import Groq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = REPO_ROOT / "data/processed/chunks_v2.jsonl"
OUTPUT_PATH = REPO_ROOT / "data/classifier/training_data.jsonl"
PROGRESS_PATH = REPO_ROOT / "data/classifier/progress.json"

MODEL = "openai/gpt-oss-20b"
BATCH_SIZE = 3
LLM_DELAY_SECONDS = 8
MIN_CHUNK_LEN = 250
MAX_CHUNK_LEN = 1200

# ------------------------------------------------------------------
# Canonical act labels (the target classes the classifier will learn)
# ------------------------------------------------------------------
ACT_LABEL_MAP = [
    ("Bharatiya_Nagarik", "BNSS"),
    ("IPC_eng", "IPC"),
    ("BNS", "BNS"),
    ("BSA", "BSA"),
    ("criminal_procedure", "CrPC"),
    ("civil_procedure", "CPC"),
    ("iea_1872", "Indian Evidence Act"),
    ("Hindu_Marriage", "Hindu Marriage Act"),
    ("Special_Marriage", "Special Marriage Act"),
    ("Dowry_Prohibition", "Dowry Prohibition Act"),
    ("Domestic_Violence", "Domestic Violence Act"),
    ("Child_Marriage", "Prohibition of Child Marriage Act"),
    ("Negotiable", "Negotiable Instruments Act"),
    ("Motor", "Motor Vehicles Act"),
    ("Narcotic", "NDPS Act"),
    ("pocso", "POCSO Act"),
    ("ArmsAct", "Arms Act"),
    ("Corruption", "Prevention of Corruption Act"),
    ("Scheduled", "SC/ST (PoA) Act"),
    ("Juvenile", "Juvenile Justice Act"),
    ("Maintenance_and_Welfare", "Maintenance of Parents Act"),
    ("pesa", "PESA"),
    ("FRA", "Forest Rights Act Rules"),
    ("Land_Revenue", "CG Land Revenue Code"),
    ("Excise", "CG Excise Act"),
    ("Rent_Control", "CG Rent Control Act"),
    ("Municipalities", "CG Municipalities Act"),
    ("Panchayat", "CG Panchayat Raj Act"),
    ("tendu", "CG Tendu Patta Act"),
    ("vanopaj", "CG Vanopaj Act"),
    ("Lok-Sewa", "CG Lok Sewa Guarantee Act"),
    ("Special Courts", "CG Special Courts Act"),
    ("Cattle", "CG Cattle Preservation Rules"),
    ("constitution", "Constitution"),
]


def canonical_act(source_file: str):
    s = source_file.lower()
    for fragment, label in ACT_LABEL_MAP:
        if fragment.lower() in s:
            return label
    return None


# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are generating training data for a legal query classifier.

Given several legal text excerpts from different sections of Indian and Chhattisgarh acts, generate 3 natural-language questions for each excerpt.

Requirements:
- Each question must be answerable by the excerpt provided.
- Vary phrasing across the 3 questions:
  - Question 1: formal, legal terminology (e.g., "grounds for eviction", "punishment for theft")
  - Question 2: casual, layperson phrasing (e.g., "can my landlord kick me out?")
  - Question 3: specific and factual (e.g., "notice period for tenant eviction")
- Do NOT mention the act name in the question. The classifier must learn to infer the act from context.
- Do NOT include the section number unless it's the natural way to ask.

Return ONLY a JSON object with this structure:
{
  "items": [
    {"chunk_index": 0, "questions": ["...", "...", "..."]},
    {"chunk_index": 1, "questions": ["...", "...", "..."]}
  ]
}

No text outside the JSON."""


def build_user_prompt(batch):
    lines = []
    for i, chunk in enumerate(batch):
        sec = chunk.get("section") or "N/A"
        text = chunk["text"][:MAX_CHUNK_LEN].replace("\n", " ").strip()
        lines.append(f"Chunk {i} (Section {sec}):\n{text}\n")
    return "\n".join(lines) + "\nGenerate 3 questions per chunk. Return the JSON."

# ------------------------------------------------------------------
# JSON parsing (robust to code fences, thinking blocks, etc.)
# ------------------------------------------------------------------
def extract_json(text):
    text = re.sub(r'<think>.*?(?:</think>|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```', '', text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None

    # JSON was truncated — attempt to repair by closing open brackets/braces
    truncated = text[start:]
    open_braces = truncated.count('{') - truncated.count('}')
    open_brackets = truncated.count('[') - truncated.count(']')
    if open_braces > 0 or open_brackets > 0:
        repaired = truncated
        # Strip any dangling partial string
        if repaired.count('"') % 2 == 1:
            repaired = repaired.rsplit('"', 1)[0]
        repaired += (']' * open_brackets) + ('}' * open_braces)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    return None

# ------------------------------------------------------------------
# LLM call with retry
# ------------------------------------------------------------------
def call_llm(messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            err = str(e)
            is_rate = any(x in err for x in ("rate_limit", "413", "429", "too large"))
            if is_rate and attempt < max_retries - 1:
                wait = 15 * (attempt + 1)
                print(f"    [rate limit] waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"    [LLM error] {err[:200]}")
            return None
    return None


# ------------------------------------------------------------------
# Load + sample chunks
# ------------------------------------------------------------------
def load_act_chunks():
    """Return list of act chunks with a section number and reasonable length."""
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            meta = c.get("metadata")
            if meta and meta.get("doc_type") == "judgment":
                continue
            if meta and meta.get("doc_type") == "constitution":
                continue  # skip; separate handling if needed later

            act = canonical_act(meta.get("source", "") if meta else "")
            if not act:
                continue

            sec = meta.get("section") if meta else None
            if not sec:
                continue

            text = c["text"]
            if len(text) < MIN_CHUNK_LEN:
                continue

            chunks.append({
                "chunk_id": c.get("id") or f"{meta.get('source')}_{sec}",
                "source_file": meta.get("source"),
                "act": act,
                "section": sec,
                "text": text,
            })
    return chunks


def stratified_sample(chunks, per_act, seed=42):
    """Sample up to `per_act` chunks per act, balanced across acts."""
    random.seed(seed)
    by_act = defaultdict(list)
    for c in chunks:
        by_act[c["act"]].append(c)

    sampled = []
    for act, items in sorted(by_act.items()):
        random.shuffle(items)
        take = items[:per_act]
        sampled.extend(take)
        print(f"  {act:<40} sampled {len(take):>3} / {len(items):>4} chunks")

    random.shuffle(sampled)
    return sampled


# ------------------------------------------------------------------
# Main generation loop
# ------------------------------------------------------------------
def load_progress():
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_progress(processed_ids):
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(list(processed_ids), f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-per-act", type=int, default=15,
                        help="Max chunks sampled per act (default: 15)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore existing progress and output, start clean")
    args = parser.parse_args()

    if args.fresh:
        OUTPUT_PATH.unlink(missing_ok=True)
        PROGRESS_PATH.unlink(missing_ok=True)

    print("Loading act chunks...")
    all_chunks = load_act_chunks()
    print(f"Total act chunks in corpus: {len(all_chunks)}\n")

    print(f"Stratified sampling ({args.chunks_per_act} per act):")
    sampled = stratified_sample(all_chunks, args.chunks_per_act)
    print(f"\nSampled for generation: {len(sampled)} chunks\n")

    processed = load_progress()
    if processed:
        before = len(sampled)
        sampled = [c for c in sampled if c["chunk_id"] not in processed]
        print(f"Resuming: {before - len(sampled)} chunks already processed, "
              f"{len(sampled)} remaining\n")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_file = open(OUTPUT_PATH, "a", encoding="utf-8")

    batches = [sampled[i:i + args.batch_size]
               for i in range(0, len(sampled), args.batch_size)]

    total_written = 0
    total_failed = 0

    for bi, batch in enumerate(batches):
        print(f"[batch {bi+1}/{len(batches)}] "
              f"{len(batch)} chunks — acts: "
              f"{', '.join(sorted(set(c['act'] for c in batch)))}")

        user_prompt = build_user_prompt(batch)
        content = call_llm([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])

        if not content:
            total_failed += 1
            time.sleep(LLM_DELAY_SECONDS)
            continue

        parsed = extract_json(content)
        if not parsed or "items" not in parsed:
            print("    [parse failed]")
            debug_path = OUTPUT_PATH.parent / "debug_failures.jsonl"
            with open(debug_path, "a", encoding="utf-8") as df:
                df.write(json.dumps({"batch_index": bi, "raw": content}) + "\n")
            total_failed += 1
            time.sleep(LLM_DELAY_SECONDS)
            continue

        # Map chunk_index → original chunk
        for item in parsed["items"]:
            idx = item.get("chunk_index")
            questions = item.get("questions", [])
            if not isinstance(idx, int) or not (0 <= idx < len(batch)):
                continue
            chunk = batch[idx]
            for qi, q in enumerate(questions[:3]):
                if not isinstance(q, str) or len(q) < 8:
                    continue
                record = {
                    "query": q.strip(),
                    "act": chunk["act"],
                    "section": chunk["section"],
                    "source_file": chunk["source_file"],
                    "chunk_id": chunk["chunk_id"],
                    "question_type": ["formal", "casual", "specific"][qi],
                }
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1

        for c in batch:
            processed.add(c["chunk_id"])

        output_file.flush()
        save_progress(processed)

        print(f"    written this batch: "
              f"{sum(len(i.get('questions', [])) for i in parsed['items'])} "
              f"(total: {total_written})")

        time.sleep(LLM_DELAY_SECONDS)

    output_file.close()

    print(f"\n{'=' * 60}")
    print(f"Total training examples: {total_written}")
    print(f"Failed batches: {total_failed}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()