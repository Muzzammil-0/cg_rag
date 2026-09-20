"""
Evaluate case-law retrieval quality.

For each question:
1. Did the query trigger the judgment filter?
2. Are the top-K retrieved sources all judgments?
3. Do the retrieved chunks contain at least one expected keyword?
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.retrieval.retriever import hybrid_search, query_wants_judgments


EVAL_SET_PATH = Path("data/evaluation/eval_set_case_law.json")
RESULTS_PATH = Path("data/evaluation/case_law_results.json")
TOP_K = 5


def keyword_in_chunks(results, keywords, k=5):
    """Return True if at least ONE keyword appears in top-k chunks (OR logic)."""
    combined = " ".join(r["text"].lower() for r in results[:k])
    return any(kw.lower() in combined for kw in keywords)


def judgment_sources_in_top_k(results, k=5):
    """Return count of judgment chunks in top-k."""
    count = 0
    for r in results[:k]:
        if r.get("metadata", {}).get("doc_type") == "judgment":
            count += 1
    return count


def main():
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    print(f"Loaded {len(eval_set)} case-law questions.\n")

    rows = []
    trigger_count = 0
    all_judgment_count = 0
    keyword_hit_count = 0
    pass_count = 0

    for item in eval_set:
        qid = item["id"]
        question = item["question"]
        keywords = item.get("expected_keywords", [])

        should_trigger = query_wants_judgments(question)

        try:
            results = hybrid_search(question, top_k=TOP_K)
        except Exception as e:
            print(f"[{qid}] ERROR: {e}")
            rows.append({"id": qid, "error": str(e)})
            continue

        judgment_in_top = judgment_sources_in_top_k(results, TOP_K)
        keyword_hit = keyword_in_chunks(results, keywords, TOP_K)

        if should_trigger:
            trigger_count += 1
        if judgment_in_top == TOP_K:
            all_judgment_count += 1
        if keyword_hit:
            keyword_hit_count += 1

        passed = (should_trigger and judgment_in_top >= 3 and keyword_hit)
        if passed:
            pass_count += 1
        status = "PASS" if passed else "FAIL"

        print(f"[{status}] {qid}")
        print(f"        Q: {question}")
        print(f"        trigger={should_trigger}  judgments_in_top5={judgment_in_top}  keyword_hit={keyword_hit}")
        if not passed:
            print(f"        top sources:")
            for r in results[:3]:
                m = r["metadata"]
                src = m.get("source", "")
                sec = m.get("section", m.get("article", m.get("para", "")))
                print(f"          - {src} {sec}")
        print()

        rows.append({
            "id": qid,
            "question": question,
            "triggered": should_trigger,
            "judgments_in_top5": judgment_in_top,
            "keyword_hit": keyword_hit,
            "status": status,
            "keywords": keywords,
            "top_sources": [
                {
                    "source": r["metadata"].get("source", ""),
                    "section": r["metadata"].get("section", r["metadata"].get("article", r["metadata"].get("para", ""))),
                    "doc_type": r["metadata"].get("doc_type", ""),
                }
                for r in results[:5]
            ],
        })

    total = len(rows)
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Questions:                 {total}")
    print(f"Judgment filter triggered: {trigger_count}/{total}  ({trigger_count/total*100:.1f}%)")
    print(f"All top-5 are judgments:   {all_judgment_count}/{total}  ({all_judgment_count/total*100:.1f}%)")
    print(f"Keyword hits:              {keyword_hit_count}/{total}  ({keyword_hit_count/total*100:.1f}%)")
    print(f"Pass rate:                 {pass_count}/{total}  ({pass_count/total*100:.1f}%)")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "total": total,
            "trigger_rate": trigger_count / total,
            "all_judgment_rate": all_judgment_count / total,
            "keyword_hit_rate": keyword_hit_count / total,
            "pass_rate": pass_count / total,
            "rows": rows,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()