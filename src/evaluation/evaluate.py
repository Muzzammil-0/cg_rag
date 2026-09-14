"""
Evaluation script for the CG Legal RAG.
Computes retrieval metrics (recall@k, MRR) and generation metrics (keyword hit).
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.retrieval.retriever import hybrid_search, detect_source_filter

EVAL_SET_PATH = Path("data/evaluation/eval_set.json")
RESULTS_PATH = Path("data/evaluation/results.json")


def load_eval_set(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize(s):
    """Normalize source strings for comparison (underscore/hyphen/space insensitive)."""
    if not s:
        return ""
    return s.lower().replace("_", " ").replace("-", " ").strip()


def section_matches(retrieved_section, expected_section):
    if not expected_section:
        return True
    if not retrieved_section:
        return False
    r = str(retrieved_section).strip().upper().rstrip('.')
    e = str(expected_section).strip().upper().rstrip('.')
    return r == e


def source_matches(retrieved_source, expected_source):
    if not expected_source:
        return True
    if not retrieved_source:
        return False
    return _normalize(expected_source) in _normalize(retrieved_source)


def is_relevant(result, expected_source, expected_section):
    meta = result.get("metadata", {})
    src_ok = source_matches(meta.get("source"), expected_source)
    sec_val = meta.get("section", meta.get("article", ""))
    sec_ok = section_matches(sec_val, expected_section)
    return src_ok and sec_ok


def recall_at_k(results, expected_source, expected_section, k):
    for r in results[:k]:
        if is_relevant(r, expected_source, expected_section):
            return 1
    return 0


def reciprocal_rank(results, expected_source, expected_section):
    for i, r in enumerate(results):
        if is_relevant(r, expected_source, expected_section):
            return 1.0 / (i + 1)
    return 0.0


def run_retrieval_eval(eval_set, top_k=10):
    rows = []
    metrics = {"recall@1": [], "recall@3": [], "recall@5": [], "recall@10": [], "mrr": []}

    for item in eval_set:
        qid = item["id"]
        question = item["question"]
        expected_source = item.get("expected_source")
        expected_section = item.get("expected_section")

        if expected_source is None:
            print(f"[SKIP] {qid} — not-found test")
            continue

        try:
            results = hybrid_search(question, top_k=top_k)
        except Exception as e:
            print(f"[ERROR] {qid}: {e}")
            rows.append({"id": qid, "question": question, "error": str(e)})
            continue

        detected_filter = detect_source_filter(question)

        r1 = recall_at_k(results, expected_source, expected_section, 1)
        r3 = recall_at_k(results, expected_source, expected_section, 3)
        r5 = recall_at_k(results, expected_source, expected_section, 5)
        r10 = recall_at_k(results, expected_source, expected_section, 10)
        mrr = reciprocal_rank(results, expected_source, expected_section)

        metrics["recall@1"].append(r1)
        metrics["recall@3"].append(r3)
        metrics["recall@5"].append(r5)
        metrics["recall@10"].append(r10)
        metrics["mrr"].append(mrr)

        top_sections = []
        for r in results[:5]:
            meta = r.get("metadata", {})
            top_sections.append({
                "source": meta.get("source", ""),
                "section": meta.get("section", meta.get("article", "")),
                "snippet": r["text"][:100].replace("\n", " "),
                "match_type": r.get("match_type", ""),
            })

        rows.append({
            "id": qid,
            "question": question,
            "expected_source": expected_source,
            "expected_section": expected_section,
            "detected_filter": detected_filter,
            "recall@1": r1,
            "recall@3": r3,
            "recall@5": r5,
            "recall@10": r10,
            "mrr": round(mrr, 3),
            "top5": top_sections,
        })

    print("\n===== RETRIEVAL METRICS =====")
    n = len(metrics["recall@1"])
    if n == 0:
        print("No evaluable questions.")
        return rows, {}

    summary = {}
    for k in ["recall@1", "recall@3", "recall@5", "recall@10"]:
        avg = sum(metrics[k]) / n
        summary[k] = round(avg, 3)
        print(f"{k}: {avg:.3f}  ({sum(metrics[k])}/{n})")
    summary["mrr"] = round(sum(metrics["mrr"]) / n, 3)
    print(f"MRR: {summary['mrr']:.3f}")
    summary["total_evaluated"] = n

    return rows, summary


def run_notfound_eval(eval_set):
    print("\n===== NOT-FOUND TESTS =====")
    try:
        from src.generation.generator import generate_answer
    except Exception as e:
        print(f"Could not import generator: {e}")
        return []

    rows = []
    for item in eval_set:
        if item.get("expected_source") is not None:
            continue
        qid = item["id"]
        question = item["question"]
        try:
            answer_data, _ = generate_answer(question)
            answer_text = answer_data.get("answer", "").lower()
            passed = "not found" in answer_text
            print(f"[{'PASS' if passed else 'FAIL'}] {qid}: '{answer_data.get('answer', '')[:120]}'")
            rows.append({
                "id": qid,
                "question": question,
                "answer": answer_data.get("answer", ""),
                "passed": passed,
            })
        except Exception as e:
            print(f"[ERROR] {qid}: {e}")
            rows.append({"id": qid, "question": question, "error": str(e)})
    return rows


def save_results(retrieval_rows, summary, notfound_rows):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "retrieval": retrieval_rows,
            "notfound": notfound_rows,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results written to {RESULTS_PATH}")


def print_failures(retrieval_rows):
    print("\n===== FAILED RETRIEVALS (recall@5 = 0) =====")
    for row in retrieval_rows:
        if row.get("recall@5", 0) == 0:
            print(f"\n[{row['id']}] {row['question']}")
            print(f"  Expected: {row['expected_source']} sec={row['expected_section']}")
            print(f"  Detected filter: {row['detected_filter']}")
            print("  Top-3 retrieved:")
            for r in row.get("top5", [])[:3]:
                print(f"    - {r['source']} sec={r['section']} ({r.get('match_type','')})")


def main():
    if not EVAL_SET_PATH.exists():
        print(f"Eval set not found at {EVAL_SET_PATH}")
        print("Create data/evaluation/eval_set.json first.")
        return

    eval_set = load_eval_set(EVAL_SET_PATH)
    print(f"Loaded {len(eval_set)} evaluation questions.")

    retrieval_rows, summary = run_retrieval_eval(eval_set, top_k=10)

    run_notfound = "--skip-notfound" not in sys.argv
    notfound_rows = run_notfound_eval(eval_set) if run_notfound else []

    save_results(retrieval_rows, summary, notfound_rows)
    print_failures(retrieval_rows)


if __name__ == "__main__":
    main()