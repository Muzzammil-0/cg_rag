"""
Remap training data labels by merging confusable act classes.

Run once on data/classifier/training_data.jsonl.
Creates a backup and rewrites the file in place.
"""

import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data/classifier/training_data.jsonl"
BACKUP_PATH = REPO_ROOT / "data/classifier/training_data_original.jsonl"

MERGE_MAP = {
    "IPC": "Criminal Code",
    "BNS": "Criminal Code",
    "CrPC": "Criminal Procedure",
    "BNSS": "Criminal Procedure",
    "Indian Evidence Act": "Evidence Law",
    "BSA": "Evidence Law",
    "Hindu Marriage Act": "Marriage Law",
    "Special Marriage Act": "Marriage Law",
    # Everything else stays as-is
}


def main():
    if not DATA_PATH.exists():
        print(f"Not found: {DATA_PATH}")
        return

    if not BACKUP_PATH.exists():
        shutil.copy(DATA_PATH, BACKUP_PATH)
        print(f"Backup saved to {BACKUP_PATH}")

    rows = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["act"] = MERGE_MAP.get(row["act"], row["act"])
            rows.append(row)

    with open(DATA_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Report distribution
    from collections import Counter
    counts = Counter(r["act"] for r in rows)
    print(f"\nMerged to {len(counts)} classes:\n")
    for act, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {act:<40} {count}")


if __name__ == "__main__":
    main()