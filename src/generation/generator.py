import os
import json
import re
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.retrieval.retriever import hybrid_search

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
CHUNK_CHAR_LIMIT = 800
GROQ_MAX_RETRIES = 3
GROQ_BACKOFF_BASE = 5


# ------------------------------------------------------------
# Retry wrapper
# ------------------------------------------------------------
def call_groq_with_retry(max_retries=GROQ_MAX_RETRIES, **kwargs):
    last_error = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            last_error = e
            err_str = str(e)
            is_rate_limit = (
                "rate_limit_exceeded" in err_str
                or "Request too large" in err_str
                or "413" in err_str
                or "429" in err_str
            )
            if is_rate_limit and attempt < max_retries - 1:
                wait = GROQ_BACKOFF_BASE * (2 ** attempt)
                print(f"[rate limit] attempt {attempt + 1}/{max_retries} — waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
    raise last_error


# ------------------------------------------------------------
# JSON extraction
# ------------------------------------------------------------
def extract_json(text):
    text = re.sub(r'<think>.*?(?:</think>|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```', '', text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find('{')
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    json_str = text[start:i + 1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        continue

    matches = list(re.finditer(r'\{.*\}', text, re.DOTALL))
    if matches:
        for match in reversed(matches):
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue

    return None


def validate_citations(parsed, results):
    valid_citations = []
    if parsed and "citations" in parsed and isinstance(parsed["citations"], list):
        for c in parsed["citations"]:
            if isinstance(c, int) and 1 <= c <= len(results):
                valid_citations.append(c)
            elif isinstance(c, str) and c.isdigit():
                c_int = int(c)
                if 1 <= c_int <= len(results):
                    valid_citations.append(c_int)
    parsed["citations"] = valid_citations
    return parsed


def validate_reasoning_provisions(parsed, results):
    """
    Drop provisions whose section numbers do not appear in the metadata
    of any retrieved chunk.
    """
    if not parsed or not isinstance(parsed.get("provisions"), list):
        return parsed

    present = set()
    for r in results:
        meta = r["metadata"]
        for key in ("section", "article", "para"):
            val = meta.get(key)
            if not val:
                continue
            s = str(val).lower().strip().rstrip('.')
            present.add(s)
            m = re.match(r'(\d+[A-Za-z]*)', s)
            if m:
                present.add(m.group(1).lower())

    kept = []
    dropped = []
    for p in parsed["provisions"]:
        sec_raw = str(p.get("section", "")).strip()
        if not sec_raw:
            kept.append(p)
            continue

        m = re.match(r'(?:section\s+)?(\d+[A-Za-z]*)', sec_raw, re.IGNORECASE)
        if not m:
            kept.append(p)
            continue
        sec_key = m.group(1).lower()

        if sec_key in present:
            kept.append(p)
        else:
            dropped.append(p)

    parsed["provisions"] = kept
    if dropped:
        print(f"[validate_reasoning] dropped {len(dropped)} hallucinated provision(s):")
        for d in dropped:
            print(f"  - {d.get('source', '?')} {d.get('section', '?')}")

    return parsed


# ------------------------------------------------------------
# Context builder
# ------------------------------------------------------------
def build_context(results, char_limit=CHUNK_CHAR_LIMIT):
    parts = []
    for i, res in enumerate(results):
        meta = res["metadata"]
        src = meta.get("source", "unknown")
        section = (
            meta.get("article")
            or meta.get("section")
            or meta.get("para")
            or ""
        )
        snippet = res["text"][:char_limit].replace("\n", " ")
        parts.append(f"[{i+1}] Source: {src}, Section/Article: {section}\n{snippet}")
    return "\n\n".join(parts)


# ------------------------------------------------------------
# Legal issue extraction (query expansion for reasoning)
# ------------------------------------------------------------
def extract_legal_issues(fact_pattern, model="openai/gpt-oss-120b"):
    """
    Given a plain-English fact pattern, ask the LLM to surface the
    legal issues using proper legal terminology. Used to improve retrieval
    before running reasoning mode.
    """
    system_msg = (
        "You are a legal analyst for Indian and Chhattisgarh law. "
        "Given a fact pattern in plain English, identify the specific legal issues it raises "
        "and phrase them using proper legal terminology that would appear in statutes. "
        "Use terms like 'rape', 'consent', 'murder', 'theft', 'bail', 'eviction', "
        "'land acquisition', 'mutation', 'partition', 'cheque bounce', 'anticipatory bail', etc. "
        "Return ONLY a JSON object with key 'issues' containing a list of 3-5 short phrases. "
        "No thinking, no explanation, output only the JSON."
    )
    user_msg = f"Fact pattern:\n{fact_pattern}\n\nReturn the JSON."

    try:
        response = call_groq_with_retry(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        content = response.choices[0].message.content
        parsed = extract_json(content)
        if parsed and isinstance(parsed.get("issues"), list):
            return [str(x) for x in parsed["issues"]]
    except Exception as e:
        print(f"[extract_legal_issues] failed: {e}")

    return []


# ------------------------------------------------------------
# MODE 1: Retrieval + answer
# ------------------------------------------------------------
def generate_answer(question, top_k=5, model="openai/gpt-oss-120b", verbose=True):
    results = hybrid_search(question, top_k=top_k)
    context = build_context(results)

    system_msg = (
        "You are a legal research assistant for Indian law and Chhattisgarh law. "
        "Use ONLY the provided context to answer the user's question. "
        "If the answer is not in the context, respond with exactly: Not found in provided context. "
        "When answering, cite the context numbers like [1], [2], etc. "
        "Every factual claim in your answer must be directly supported by the text of the cited chunk. "
        "If a claim is not present in any retrieved chunk, do not include it. "
        "Prefer shorter answers over answers that mix retrieved and remembered information. "
        "Return your response as a JSON object with keys: answer (string), citations (list of numbers), confidence (float between 0 and 1). "
        "Do not include any text outside the JSON object. "
        "Do not hallucinate or make up information. "
        "Do not include any thinking or reasoning. Output only the JSON object."
    )

    user_msg = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Provide the JSON response."
    )

    response = call_groq_with_retry(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=1024,
    )

    content = response.choices[0].message.content
    if verbose:
        print("Raw model output:\n", content)

    parsed = extract_json(content)
    if parsed is None:
        parsed = {
            "answer": content.strip(),
            "citations": [],
            "confidence": 0.0,
        }
    else:
        parsed = validate_citations(parsed, results)

    return parsed, results


# ------------------------------------------------------------
# MODE 2: Reasoning with query expansion
# ------------------------------------------------------------
def generate_reasoning(fact_pattern, top_k=10, model="openai/gpt-oss-120b", verbose=True):
    """
    Simulate how a Chhattisgarh court might reason through a fact pattern.
    Educational tool for law students. Not legal advice, not a prediction.
    """
    # Step 1: Extract legal issues to improve retrieval vocabulary
    legal_issues = extract_legal_issues(fact_pattern, model=model)
    if verbose:
        print(f"[reasoning] extracted legal issues: {legal_issues}")

    # Step 2: Build an augmented search query
    if legal_issues:
        augmented_query = fact_pattern + " | " + " ".join(legal_issues)
    else:
        augmented_query = fact_pattern

    # Step 3: Retrieve with the augmented query
    results = hybrid_search(augmented_query, top_k=top_k)
    context = build_context(results)

    system_msg = (
        "You are simulating how a Chhattisgarh court might reason through a legal question. "
        "This is an EDUCATIONAL exercise for law students, not legal advice. "
        "Use ONLY the provided context from statutes and judgments. "
        "Do not invent provisions, case names, or citations. "
        "Every section number you cite MUST appear in the retrieved context with its act name. "
        "Use the act name exactly as it appears in the source (e.g., 'IPC Section 376', 'BNS Section 64'). "
        "If the context is insufficient, say so in the relevant field. "
        "Return a JSON object with these exact keys: "
        "issues (list of strings), "
        "provisions (list of objects with keys: source, section, relevance), "
        "petitioner_arguments (list of strings), "
        "respondent_arguments (list of strings), "
        "likely_reasoning (string), "
        "possible_outcome (string, hedged with caveats), "
        "confidence (float between 0 and 1), "
        "citations (list of context numbers used as integers). "
        "Be concise: each list should have 3-5 items max, and each string should be under 200 characters. "
        "Do not include any text outside the JSON object. "
        "Do not include any thinking or reasoning. Output only the JSON object."
    )

    user_msg = (
        f"Fact pattern:\n{fact_pattern}\n\n"
        f"Retrieved legal context:\n{context}\n\n"
        "Produce the reasoning JSON."
    )

    response = call_groq_with_retry(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=3000,
    )

    content = response.choices[0].message.content
    if verbose:
        print("Raw model output:\n", content)

    parsed = extract_json(content)
    if parsed is None:
        print("\n[WARNING] Parse failed. Output may be truncated.")
        parsed = {
            "issues": [], "provisions": [],
            "petitioner_arguments": [], "respondent_arguments": [],
            "likely_reasoning": "", "possible_outcome": "",
            "confidence": 0.0, "citations": [],
            "error": "parse_failed", "raw": content,
        }
    else:
        parsed = validate_citations(parsed, results)
        parsed = validate_reasoning_provisions(parsed, results)

    return parsed, results


# ------------------------------------------------------------
# CLI test
# ------------------------------------------------------------
if __name__ == "__main__":
    mode = input("Mode (query/reason): ").strip().lower()

    if mode == "reason":
        pattern = input("Enter fact pattern: ")
        result, retrieved = generate_reasoning(pattern)

        print("\n=== REASONING OUTPUT ===")
        print("\nIssues:")
        for issue in result.get("issues", []):
            print(f"  - {issue}")

        print("\nProvisions (validated):")
        for prov in result.get("provisions", []):
            print(f"  - {prov.get('source')} {prov.get('section')}: {prov.get('relevance')}")

        print("\nPetitioner Arguments:")
        for arg in result.get("petitioner_arguments", []):
            print(f"  - {arg}")

        print("\nRespondent Arguments:")
        for arg in result.get("respondent_arguments", []):
            print(f"  - {arg}")

        print("\nLikely Reasoning:")
        print(f"  {result.get('likely_reasoning', '')}")

        print("\nPossible Outcome:")
        print(f"  {result.get('possible_outcome', '')}")

        print(f"\nConfidence: {result.get('confidence', 0.0)}")
        print(f"Citations: {result.get('citations', [])}")

        print("\nRetrieved sources:")
        for i, r in enumerate(retrieved):
            m = r["metadata"]
            sec = m.get("section", m.get("article", m.get("para", "")))
            print(f"  [{i+1}] {m.get('source')} — {sec}")
    else:
        question = input("Enter your legal question: ")
        answer_data, retrieved = generate_answer(question)

        print("\nAnswer:", answer_data.get("answer"))
        print("Citations:", answer_data.get("citations"))
        print("Confidence:", answer_data.get("confidence"))

        print("\nRetrieved context snippets:")
        for i, res in enumerate(retrieved):
            print(f"[{i+1}] {res['text'][:200]}...")