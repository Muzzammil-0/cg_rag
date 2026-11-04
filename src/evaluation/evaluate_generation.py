"""
Generation-level evaluation for CG Legal RAG.

Complements evaluate.py (which scores retrieval) by scoring the FINAL answer:
  * answer_rate        — in-domain questions that produced a real answer
  * citation_grounding — at least one cited chunk matches the expected source/section
  * keyword_coverage   — fraction of expected keywords present in the answer
  * not_found_accuracy — out-of-domain questions correctly refused

This runs the real Groq LLM, so it needs GROQ_API_KEY set and consumes API quota.
Use --limit to keep it cheap.

Usage:
  python src/evaluation/evaluate_generation.py --limit 15
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.evaluation.evaluate import is_relevant, load_eval_set
from src.generation.generator import generate_answer

EVAL_SET_PATH = PROJECT_ROOT / "data/evaluation/eval_set.json"
RESULTS_PATH = PROJECT_ROOT / "data/evaluation/generation_results.json"


def keyword_coverage(answer: str, keywords: List[str]) -> float:
    """Fraction of expected keywords found in the answer (case-insensitive)."""
    if not keywords:
        return 0.0
    text = (answer or "").lower()
    hits = sum(1 for k in keywords if str(k).lower() in text)
    return hits / len(keywords)


def is_real_answer(answer: str) -> bool:
    """True when the answer is neither empty, an error, nor a not-found refusal."""
    a = (answer or "").strip().lower()
    return bool(a) and a != "error." and "not found in provided context" not in a


def cited_chunks(results: List[Dict], citations: List[int]) -> List[Dict]:
    """Map 1-based citation integers back to retrieved chunks, dropping bad indices."""
    out = []
    for c in citations or []:
        idx = c - 1
        if 0 <= idx < len(results):
            out.append(results[idx])
    return out


def aggregate_generation_metrics(rows: List[Dict]) -> Dict[str, Any]:
    """Average the per-question generation scores over evaluable rows."""
    scored = [r for r in rows if r.get("expected_source") is not None and "error" not in r]
    n = len(scored)
    if n == 0:
        return {"total_evaluated": 0}
    return {
        "total_evaluated": n,
        "answer_rate": round(sum(1 for r in scored if r["answered"]) / n, 3),
        "citation_grounding": round(sum(1 for r in scored if r["citation_hit"]) / n, 3),
        "keyword_coverage": round(sum(r["keyword_coverage"] for r in scored) / n, 3),
        "mean_confidence": round(sum(r.get("confidence", 0.0) for r in scored) / n, 3),
    }


def run_generation_eval(eval_set: List[Dict], top_k: int = 10, limit: int = None) -> List[Dict]:
    rows: List[Dict] = []
    evaluable = [item for item in eval_set if item.get("expected_source") is not None]
    if limit:
        evaluable = evaluable[:limit]

    for item in evaluable:
        qid = item["id"]
        question = item["question"]
        expected_source = item.get("expected_source")
        expected_section = item.get("expected_section")
        keywords = item.get("expected_keywords", [])

        try:
            answer_data, results = generate_answer(question, top_k=top_k, verbose=False)
        except Exception as e:
            print(f"[ERROR] {qid}: generation failed: {e}")
            rows.append({"id": qid, "question": question, "expected_source": expected_source, "error": str(e)})
            continue

        answer_data = answer_data or {}
        results = results or []
        answer = answer_data.get("answer", "")
        citations = answer_data.get("citations", [])
        citation_hit = any(is_relevant(r, expected_source, expected_section) for r in cited_chunks(results, citations))
        coverage = round(keyword_coverage(answer, keywords), 3)

        rows.append({
            "id": qid,
            "question": question,
            "expected_source": expected_source,
            "expected_section": expected_section,
            "answered": is_real_answer(answer),
            "citation_hit": citation_hit,
            "num_citations": len(citations),
            "keyword_coverage": coverage,
            "confidence": float(answer_data.get("confidence", 0.0)),
            "answer": answer[:300],
        })
        print(f"[{'OK' if citation_hit else 'MISS'}] {qid}: answered={rows[-1]['answered']} citations={len(citations)} kw={coverage}")

    return rows


def run_notfound_eval(eval_set: List[Dict], limit: int = None) -> List[Dict]:
    rows: List[Dict] = []
    notfound = [item for item in eval_set if item.get("expected_source") is None]
    if limit:
        notfound = notfound[:limit]

    for item in notfound:
        qid = item["id"]
        question = item["question"]
        try:
            answer_data, _ = generate_answer(question, top_k=3, verbose=False)
        except Exception as e:
            print(f"[ERROR] {qid}: {e}")
            rows.append({"id": qid, "question": question, "error": str(e), "passed": False})
            continue
        answer = (answer_data or {}).get("answer", "")
        passed = "not found" in answer.lower()
        rows.append({"id": qid, "question": question, "answer": answer[:200], "passed": passed})
        print(f"[{'PASS' if passed else 'FAIL'}] {qid}: {answer[:100]}")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Generation-level evaluation for CG Legal RAG.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="Cap number of questions (saves API quota).")
    parser.add_argument("--skip-notfound", action="store_true")
    args = parser.parse_args()

    eval_set = load_eval_set(EVAL_SET_PATH)
    print(f"--- Generation evaluation on {len(eval_set)} questions (limit={args.limit}) ---")

    gen_rows = run_generation_eval(eval_set, top_k=args.top_k, limit=args.limit)
    summary = aggregate_generation_metrics(gen_rows)

    notfound_rows = [] if args.skip_notfound else run_notfound_eval(eval_set, limit=args.limit)
    if notfound_rows:
        summary["not_found_passed"] = sum(1 for r in notfound_rows if r.get("passed"))
        summary["not_found_total"] = len(notfound_rows)

    print("\n===== GENERATION METRICS =====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "generation": gen_rows, "notfound": notfound_rows}, f, indent=2, ensure_ascii=False)
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()