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
    """Try to extract the first balanced JSON object from text."""
    # Removing markdown code fences if present
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```', '', text)
    text = text.strip()

    # First trying direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Finding first { and matching closing }
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
                json_str = text[start:i+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
    return None

def generate_answer(question, top_k=5, model="qwen/qwen3.6-27b"):
    # 1. Retrieve relevant chunks
    results = hybrid_search(question, top_k=top_k)

    # 2. Build context string with numbered chunks
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
        "Use ONLY the provided context to answer the user's question. "
        "If the answer is not in the context, respond with exactly: Not found in provided context. "
        "When answering, cite the context numbers like [1], [2], etc. "
        "Return your response as a JSON object with keys: answer (string), citations (list of numbers), confidence (float between 0 and 1). "
        "Do not include any text outside the JSON object."
        "Do not hallucinate or make up information. If the context is insufficient, answer with 'Not found in provided context.'"
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

    # Debug: print raw content to console (you can remove later)
    print("Raw model output:\n", content)

    # 5. Parse JSON
    parsed = extract_json(content)
    parsed = validate_citations(parsed, results)

    if parsed is None:
        # Fallback: return raw text as answer, no citations
        parsed = {
            "answer": content.strip(),
            "citations": [],
            "confidence": 0.0
        }

    return parsed, results

def validate_citations(parsed, results):
    valid_citations = []
    if "citations" in parsed and isinstance(parsed["citations"],list):
        for c in parsed["citations"]:
            if isinstance(c, int) and 1 <= c <= len(results):
                valid_citations.append(c)
            elif isinstance(c, str) and c.isdigit():
                c_int = int(c)
                if 1 <= c_int <= len(results):
                    valid_citations.append(c_int)
    parsed["citations"] = valid_citations
    return parsed

if __name__ == "__main__":
    question = input("Enter your legal question: ")
    answer_data, retrieved = generate_answer(question)

    print("\nAnswer:", answer_data.get("answer"))
    print("Citations:", answer_data.get("citations"))
    print("Confidence:", answer_data.get("confidence"))

    print("\nRetrieved context snippets:")
    for i, res in enumerate(retrieved):
        print(f"[{i+1}] {res['text'][:200]}...")