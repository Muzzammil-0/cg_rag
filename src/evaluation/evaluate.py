"""
Evaluation script for the CG Legal RAG.
Computes retrieval metrics (recall@k, MRR) and generation metrics (keyword hit).
"""
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Ensure project root is on path for module imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# --- DOCKER PATH FIX ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from src.retrieval.retriever import hybrid_search, detect_source_filter
except ImportError as e:
    print(f"Error importing required modules (Check 'src/retrieval/retriever.py' and dependencies): {e}")
    # Set dummy functions to allow the script structure to be tested even if imports fail
    def hybrid_search(*args, **kwargs) -> List[Dict]: return []
    def detect_source_filter(*args, **kwargs) -> str: return "none"

# UPDATE PATHS
EVAL_SET_PATH = PROJECT_ROOT / "data/evaluation/eval_set.json"
RESULTS_PATH = PROJECT_ROOT / "data/evaluation/results.json"


def load_eval_set(path: Path) -> List[Dict[str, Any]]:
    """Loads and handles JSON loading errors for the evaluation set."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Evaluation set not found at {path}")
        sys.exit(1) # Exit if the fundamental dataset is missing
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to decode JSON from {path}: {e}")
        sys.exit(1)


def _normalize(s: str | None) -> str:
    """Normalize source strings for comparison (underscore/hyphen/space insensitive)."""
    if not s:
        return ""
    # Ensure s is treated as string before calling replace methods
    return str(s).lower().replace("_", " ").replace("-", " ").strip()


def section_matches(retrieved_section: str | None, expected_section: str | None) -> bool:
    """Checks if the retrieved section matches the expected section."""
    if not expected_section:
        return True # No expectation means it always passes
    if not retrieved_section:
        return False # Expectation exists but nothing was retrieved
    
    # Normalize and clean both strings for comparison
    r = str(retrieved_section).strip().upper().rstrip('.')
    e = str(expected_section).strip().upper().rstrip('.')
    
    # Simple equality check after cleaning/normalizing is usually best
    return r == e


def source_matches(retrieved_source: str | None, expected_source: str | None) -> bool:
    """Checks if the retrieved source matches the expected source using fuzzy matching."""
    if not expected_source:
        return True
    if not retrieved_source:
        # If we expect a source but get none, it fails unless we are lenient
        return False 

    norm_expected = _normalize(expected_source)
    norm_retrieved = _normalize(retrieved_source)
    
    # Checking if expected content is contained within the retrieved source name
    return norm_expected in norm_retrieved


def is_relevant(result: Dict[str, Any], expected_source: str | None, expected_section: str | None) -> bool:
    """Determines if a given retrieval result meets both source and section criteria."""
    meta = result.get("metadata", {})
    src_ok = source_matches(meta.get("source"), expected_source)
    sec_val = meta.get("section", meta.get("article", ""))
    sec_ok = section_matches(sec_val, expected_section)
    return src_ok and sec_ok


def recall_at_k(results: List[Dict], expected_source: str | None, expected_section: str | None, k: int) -> int:
    """Calculates if at least one relevant result is present within the top K."""
    for r in results[:k]:
        if is_relevant(r, expected_source, expected_section):
            return 1 # Found it
    return 0


def reciprocal_rank(results: List[Dict], expected_source: str | None, expected_section: str | None) -> float:
    """Calculates the rank (1/i) of the first relevant result."""
    for i, r in enumerate(results):
        if is_relevant(r, expected_source, expected_section):
            # Return 1.0 / position (index starts at 0, so position is i+1)
            return 1.0 / (i + 1)
    return 0.0


def run_retrieval_eval(eval_set: List[Dict], top_k: int = 10) -> Tuple[List[Dict], Dict]:
    """Runs the main retrieval evaluation loop and calculates summary metrics."""
    rows: List[Dict] = []
    # Store raw lists for accurate summation later
    metrics: Dict[str, List[float]] = {
        "recall@1": [], "recall@3": [], "recall@5": [], "recall@10": [], "mrr": []
    }

    for item in eval_set:
        qid: str = item["id"]
        question: str = item["question"]
        expected_source: str | None = item.get("expected_source")
        expected_section: str | None = item.get("expected_section")

        if expected_source is None:
            print(f"[SKIP] {qid} — not-found test (Skipped retrieval metrics)")
            # Still record the question to maintain row alignment, but with flags
            rows.append({
                "id": qid, "question": question, 
                "expected_source": None, "expected_section": None, 
                "detected_filter": "N/A",
                "recall@1": 0, "recall@3": 0, "recall@5": 0, "recall@10": 0, "mrr": 0.0, "top5": []
            })
            continue

        try:
            # Step 1: Retrieval
            results = hybrid_search(question, top_k=top_k, exhaustive=True)
        except Exception as e:
            print(f"[ERROR] {qid}: Failed during retrieval: {e}")
            rows.append({"id": qid, "question": question, "error": str(e), 
                          "expected_source": expected_source, "expected_section": expected_section})
            continue

        # Step 2: Filtering and Metrics Calculation
        detected_filter = detect_source_filter(question)

        r1 = recall_at_k(results, expected_source, expected_section, 1)
        r3 = recall_at_k(results, expected_source, expected_section, 3)
        r5 = recall_at_k(results, expected_source, expected_section, 5)
        r10 = recall_at_k(results, expected_source, expected_section, 10)
        mrr = reciprocal_rank(results, expected_source, expected_section)

        metrics["recall@1"].append(float(r1))
        metrics["recall@3"].append(float(r3))
        metrics["recall@5"].append(float(r5))
        metrics["recall@10"].append(float(r10))
        metrics["mrr"].append(mrr)

        # Step 3: Formatting Top K results for display/debugging
        top_sections = []
        for r in results[:5]:
            meta = r.get("metadata", {})
            snippet = r.get("text", "")
            top_sections.append({
                "source": meta.get("source", ""),
                "section": meta.get("section", meta.get("article", "")),
                "snippet": snippet[:100].replace("\n", " "), # Truncated to 100 chars
                "match_type": r.get("match_type", ""),
            })

        rows.append({
            "id": qid,
            "question": question,
            "expected_source": expected_source,
            "expected_section": expected_section,
            "detected_filter": detected_filter,
            "recall@1": r1, "recall@3": r3, "recall@5": r5, "recall@10": r10, 
            "mrr": round(mrr, 3),
            "top5": top_sections,
        })

    print("\n===== RETRIEVAL METRICS =====")
    n = len(metrics["recall@1"]) # Total number of questions that were evaluable
    if n == 0:
        print("No evaluable questions found.")
        return rows, {}

    summary: Dict[str, Any] = {"total_evaluated": float(n)}
    for k in ["recall@1", "recall@3", "recall@5", "recall@10"]:
        metrics_list = metrics[k]
        sum_val = sum(metrics_list)
        avg = sum_val / n
        summary[f"{k}_sum"] = sum_val # Store raw sum
        summary[k] = round(avg, 3)     # Store average
        print(f"{k}: {avg:.3f}  ({int(sum_val)}/{n})")
    
    mrr_list = metrics["mrr"]
    summary["mrr_sum"] = sum(mrr_list)
    summary["mrr"] = round(sum(mrr_list) / n, 3)
    print(f"MRR: {summary['mrr']:.3f}")

    return rows, summary


def run_notfound_eval(eval_set: List[Dict]) -> List[Dict]:
    """Runs evaluation for questions expected to result in 'not found' answers."""
    print("\n===== NOT-FOUND TESTS =====")
    
    # Attempt import only here, as it's conditional logic
    try:
        from src.generation.generator import generate_answer
    except ImportError as e:
        print(f"[CRITICAL] Could not import generator module for Not Found tests: {e}")
        return []

    rows: List[Dict] = []
    for item in eval_set:
        if item.get("expected_source") is not None:
            continue # Only run if no expected source means it's a 'not found' test
            
        qid: str = item["id"]
        question: str = item["question"]
        
        try:
            # Pass dummy top_k and use verbose=False as per app.py structure
            answer_data, _ = generate_answer(question, top_k=1, verbose=False) 
            
            answer_text = answer_data.get("answer", "")
            passed = "not found" in answer_text.lower()
            
            status = 'PASS' if passed else 'FAIL'
            print(f"[{status}] {qid}: '{answer_text[:120]}...'")
            
            rows.append({
                "id": qid,
                "question": question,
                "expected_source": None, # Explicitly set for clarity in results
                "expected_section": None,
                "detected_filter": "N/A",
                "answer": answer_text,
                "passed": passed,
            })
        except Exception as e:
            print(f"[ERROR] {qid}: Failed during generation test due to internal error: {e}")
            rows.append({"id": qid, "question": question, "error": str(e), 
                          "expected_source": None, "expected_section": None})
    return rows


def save_results(retrieval_rows: List[Dict], summary: Dict[str, Any], notfound_rows: List[Dict]):
    """Saves all evaluation results into a structured JSON file."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "retrieval": retrieval_rows,
            "notfound": notfound_rows,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Detailed results written to {RESULTS_PATH}")


def print_failures(retrieval_rows: List[Dict]):
    """Prints a summary of questions that failed key retrieval metrics."""
    print("\n===== FAILED RETRIEVALS (recall@5 = 0) =====")
    failed_count = 0
    for row in retrieval_rows:
        if row.get("recall@5", 0) == 0 and row.get("expected_source") is not None:
            print(f"\n[FAIL] {row['id']} | Question: {row['question'][:80]}...")
            print(f"  Expected: {row['expected_source']} sec={row['expected_section']}")
            # Print only the top 3 for brevity in console output
            top5_display = row.get("top5", [])[:3]
            for r in top5_display:
                print(f"  -> Found Source: {r['source']} sec={r['section']}. Snippet: '{r['snippet'][:60]}...'")
            failed_count += 1
    
    if failed_count == 0 and retrieval_rows:
        print("No retrievals failed the recall@5 test.")


def main():
    """Main execution function for running all evaluation steps."""
    eval_set = load_eval_set(EVAL_SET_PATH)
    print(f"--- Starting Evaluation on {len(eval_set)} Questions ---")

    # 1. Run Retrieval Evaluation (Source/Section match, Recall@K, MRR)
    retrieval_rows, summary = run_retrieval_eval(eval_set, top_k=10)

    # 2. Conditional Not Found Test Execution
    run_notfound: bool = "--skip-notfound" not in sys.argv
    if run_notfound:
        notfound_rows = run_notfound_eval(eval_set)
    else:
        print("\n[SKIPPED] Running 'not found' generation tests.")
        notfound_rows = []

    # 3. Save and Report
    save_results(retrieval_rows, summary, notfound_rows)
    print_failures(retrieval_rows)


if __name__ == "__main__":
    main()