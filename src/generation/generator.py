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
    # Remove thinking blocks (both closed and unclosed)
    text = re.sub(r'<think>.*?(?:</think>|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```', '', text)
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find first balanced JSON object
    start = text.find('{')
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    json_str = text[start:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        continue

    # Fallback: try to find JSON near the end (after any thinking)
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

def generate_answer(question, top_k=5, model="openai/gpt-oss-120b", verbose=True):
    # 1. Retrieve relevant chunks
    results = hybrid_search(question, top_k=top_k)

    # 2. Build context string
    context_parts = []
    for i, res in enumerate(results):
        meta = res["metadata"]
        src = meta.get("source", "unknown")
        section = meta.get("article") or meta.get("section") or ""
        context_parts.append(
            f"[{i+1}] Source: {src}, Section/Article: {section}\n{res['text']}"
        )
    context = "\n\n".join(context_parts)

    # 3. Create prompt
    system_msg = (
        "You are a legal research assistant for Indian law and Chhattisgarh law. "
        "Every factual claim in your answer must be directly supported by the text of the cited chunk. If a claim isn't in any retrieved chunk, do not include it. Prefer shorter answers over answers that mix retrievedand remembered information.cs"
        "Use ONLY the provided context to answer the user's question. "
        "If the answer is not in the context, respond with exactly: Not found in provided context. "
        "When answering, cite the context numbers like [1], [2], etc. "
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

    # 4. Call Groq
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.0,
        max_tokens=1024,
    )

    content = response.choices[0].message.content
    if verbose:
        print("Raw model output:\n", content)

    # 5. Parse JSON and validate
    parsed = extract_json(content)
    if parsed is None:
        parsed = {
            "answer": content.strip(),
            "citations": [],
            "confidence": 0.0
        }
    else:
        parsed = validate_citations(parsed, results)

    return parsed, results

if __name__ == "__main__":
    question = input("Enter your legal question: ")
    answer_data, retrieved = generate_answer(question)

    print("\nAnswer:", answer_data.get("answer"))
    print("Citations:", answer_data.get("citations"))
    print("Confidence:", answer_data.get("confidence"))

    print("\nRetrieved context snippets:")
    for i, res in enumerate(retrieved):
        print(f"[{i+1}] {res['text'][:200]}...")