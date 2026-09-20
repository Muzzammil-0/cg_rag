import os
import json
import re
import sys
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.retrieval.retriever import hybrid_search

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def extract_json(text):
    """Try to extract the first balanced JSON object from text, stripping thinking blocks."""
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


# ------------------------------------------------------------------
# MODE 1: Retrieval + answer (existing behavior)
# ------------------------------------------------------------------
def generate_answer(question, top_k=5, model="openai/gpt-oss-120b", verbose=True):
    results = hybrid_search(question, top_k=top_k)

    context_parts = []
    for i, res in enumerate(results):
        meta = res["metadata"]
        src = meta.get("source", "unknown")
        section = meta.get("article") or meta.get("section") or meta.get("para") or ""
        context_parts.append(
            f"[{i+1}] Source: {src}, Section/Article: {section}\n{res['text']}"
        )
    context = "\n\n".join(context_parts)

    system_msg = (
        "You are a legal research assistant for Indian law and Chhattisgarh law. "
        "Use ONLY the provided context to answer the user's question. "
        "If the answer is not in the context, respond with exactly: Not found in provided context. "
        "When answering, cite the context numbers like [1], [2], etc. "
        "Every factual claim in your answer must be directly supported by the text of the cited chunk. "
        "If a claim is not present in any retrieved chunk, do not include it. "
        "Prefer shorter answers over answers that mix retrieved and remembered information. "
        "Return your response as a JSON object with keys: answer (string), citations (list of numbers), confidence (float between 0 and 1). "
        "Be concise: each list should have 3-5 items max, and each string should be under 200 characters."
        "Do not include any thinking or reaosoning. Output only the JSON object."
        "Do not hallucinate or make up information. "
        "Do not include any thinking or reasoning. Output only the JSON object."
    )

    user_msg = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Provide the JSON response."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=3000,
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


# ------------------------------------------------------------------
# MODE 2: Simulated judicial reasoning (educational)
# ------------------------------------------------------------------
def generate_reasoning(fact_pattern, top_k=8, model="openai/gpt-oss-120b", verbose=True):
    """
    Simulate how a Chhattisgarh court might reason through a fact pattern.
    Educational tool for law students. Not legal advice, not a prediction.
    """
    results = hybrid_search(fact_pattern, top_k=top_k)

    context_parts = []
    for i, res in enumerate(results):
        meta = res["metadata"]
        src = meta.get("source", "unknown")
        sec = meta.get("section") or meta.get("article") or meta.get("para") or ""
        context_parts.append(f"[{i+1}] Source: {src}, Section/Article: {sec}\n{res['text']}")
    context = "\n\n".join(context_parts)

    system_msg = (
        "You are simulating how a Chhattisgarh court might reason through a legal question. "
        "This is an EDUCATIONAL exercise for law students, not legal advice. "
        "Use ONLY the provided context from statutes and judgments. "
        "Do not invent provisions, case names, or citations. "
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
        "Do not include any text outside the JSON object. "
        "Do not include any thinking or reasoning. Output only the JSON object."
    )

    user_msg = (
        f"Fact pattern:\n{fact_pattern}\n\n"
        f"Retrieved legal context:\n{context}\n\n"
        "Produce the reasoning JSON."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=3000,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if verbose:
        print("Raw model output:\n", content)

    parsed = extract_json(content)
    if parsed is None:
        print("\n[WARNING] Failed to parse JSON — model output may have been truncated.")
        print("Consider increasing max_tokens or reducing prompt verbosity.")
        parsed = {
            "issues": [],
            "provisions": [],
            "petitioner_arguments": [],
            "respondent_arguments": [],
            "likely_reasoning": "",
            "possible_outcome": "",
            "confidence": 0.0,
            "citations": [],
            "error": "parse_failed",
            "raw": content,
        }
    else:
        parsed = validate_citations(parsed, results)

    return parsed, results

# ------------------------------------------------------------------
# CLI test harness
# ------------------------------------------------------------------
if __name__ == "__main__":
    mode = input("Mode (query/reason): ").strip().lower()

    if mode == "reason":
        pattern = input("Enter fact pattern: ")
        result, retrieved = generate_reasoning(pattern)

        print("\n=== REASONING OUTPUT ===")
        print("\nIssues:")
        for issue in result.get("issues", []):
            print(f"  - {issue}")

        print("\nProvisions:")
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