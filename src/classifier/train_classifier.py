"""
Train an act-routing classifier on top of BGE-small.

Output:
  data/classifier/model/       — trained model + tokenizer
  data/classifier/metrics.json — validation/test metrics, per-act breakdown

Usage:
  python src/classifier/train_classifier.py
"""

import json
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data/classifier/training_data.jsonl"
OUTPUT_DIR = REPO_ROOT / "data/classifier/model"
METRICS_PATH = REPO_ROOT / "data/classifier/metrics.json"

ENCODER = "BAAI/bge-small-en-v1.5"
MAX_LEN = 128
SEED = 42

# ------------------------------------------------------------------
# Load + split
# ------------------------------------------------------------------
def load_examples(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def split_data(rows, seed=SEED):
    by_act = defaultdict(list)
    for r in rows:
        by_act[r["act"]].append(r)

    train, val, test = [], [], []
    small_classes = []

    for act, items in by_act.items():
        random.seed(seed)
        random.shuffle(items)

        if len(items) < 5:
            train.extend(items)
            small_classes.append((act, len(items)))
            continue

        n = len(items)
        n_test = max(1, int(n * 0.1))
        n_val = max(1, int(n * 0.1))
        n_train = n - n_test - n_val

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    random.seed(seed)
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test, small_classes


# ------------------------------------------------------------------
# Torch Dataset
# ------------------------------------------------------------------
class ActDataset(Dataset):
    def __init__(self, rows, tokenizer, act_to_idx):
        self.rows = rows
        self.tokenizer = tokenizer
        self.act_to_idx = act_to_idx

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = self.tokenizer(
            row["query"],
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.act_to_idx[row["act"]], dtype=torch.long),
        }


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print("Loading training data...")
    rows = load_examples(DATA_PATH)
    print(f"Total examples: {len(rows)}\n")

    acts = sorted(set(r["act"] for r in rows))
    act_to_idx = {a: i for i, a in enumerate(acts)}
    idx_to_act = {i: a for a, i in act_to_idx.items()}
    num_classes = len(acts)
    print(f"Classes: {num_classes}")
    for i, a in enumerate(acts):
        print(f"  {i:>2}  {a}")
    print()

    train_rows, val_rows, test_rows, small_classes = split_data(rows)
    print(f"Split sizes: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")
    if small_classes:
        print(f"Small classes (<5 examples, all in train): {small_classes}")
    print()

    print("Loading tokenizer + model...")
    tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    model = AutoModelForSequenceClassification.from_pretrained(
        ENCODER, num_labels=num_classes
    )

    train_ds = ActDataset(train_rows, tokenizer, act_to_idx)
    val_ds = ActDataset(val_rows, tokenizer, act_to_idx)
    test_ds = ActDataset(test_rows, tokenizer, act_to_idx)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Loosened hyperparameters for higher confidence
    # - more epochs so the model actually converges
    # - higher LR to escape the underfit regime
    # - much lower weight decay so the model can be confident
    # - shorter warmup so training gets to the point faster
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=5e-5,
        warmup_ratio=0.05,
        weight_decay=0.001,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=SEED,
        report_to="none",
        fp16=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )

    print("\nTraining...")
    trainer.train()

    print("\nEvaluating on validation set...")
    val_metrics = trainer.evaluate(val_ds)
    print(val_metrics)

    print("\nEvaluating on test set...")
    test_output = trainer.predict(test_ds)
    test_preds = np.argmax(test_output.predictions, axis=1)
    test_labels = test_output.label_ids

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)

    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test f1_macro: {test_f1:.4f}")

    print("\nPer-class report (test set):")
    target_names = [idx_to_act[i] for i in range(num_classes)]
    report = classification_report(
        test_labels, test_preds,
        labels=list(range(num_classes)),
        target_names=target_names,
        zero_division=0,
        digits=3,
    )
    print(report)

    print(f"\nSaving model to {OUTPUT_DIR}...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    with open(OUTPUT_DIR / "act_labels.json", "w", encoding="utf-8") as f:
        json.dump({"act_to_idx": act_to_idx, "idx_to_act": idx_to_act}, f, indent=2)

    metrics = {
        "val": val_metrics,
        "test": {"accuracy": test_acc, "f1_macro": test_f1},
        "num_classes": num_classes,
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "num_test": len(test_rows),
        "small_classes": small_classes,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics written to {METRICS_PATH}")


if __name__ == "__main__":
    main()