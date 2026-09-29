"""Evaluate trained models on the speaker-disjoint EmoDB holdout split.

The split deliberately matches train.py: GroupShuffleSplit with seed 42 and a
20% test set. The generated JSON files are consumed by the web interface.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from app.audio import to_mel
from app.model import EmotionCNN, TransferEmotionCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = Path(args.artifacts)
    with Path(args.manifest).open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows or set(rows[0]) != {"path", "label", "speaker"}:
        raise SystemExit("The CSV must contain exactly path,label,speaker.")

    labels = json.loads((artifacts / "labels.json").read_text(encoding="utf-8"))
    targets = np.array([labels.index(row["label"]) for row in rows])
    speakers = np.array([row["speaker"] for row in rows])
    split = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=args.seed)
    train_idx, test_idx = next(split.split(np.zeros(len(rows)), targets, speakers))
    features = np.stack([to_mel(Path(row["path"])) for row in rows])
    test = TensorDataset(torch.tensor(features[test_idx]), torch.tensor(targets[test_idx]))

    for architecture in ("scratch", "transfer"):
        weights_path = artifacts / f"model-{architecture}.pt"
        if not weights_path.exists():
            print(f"Skipping {architecture}: {weights_path} does not exist.")
            continue
        model = TransferEmotionCNN(len(labels), pretrained=False) if architecture == "transfer" else EmotionCNN(len(labels))
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        model.eval()
        actual, predicted = [], []
        with torch.inference_mode():
            for features_batch, targets_batch in DataLoader(test, batch_size=args.batch_size):
                actual.extend(targets_batch.tolist())
                predicted.extend(model(features_batch).argmax(dim=1).tolist())

        matrix = confusion_matrix(actual, predicted, labels=list(range(len(labels))))
        recalls = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
        report = {
            "architecture": architecture,
            "seed": args.seed,
            "labels": labels,
            "speakers_train": sorted(set(speakers[train_idx])),
            "speakers_test": sorted(set(speakers[test_idx])),
            "test_samples": len(actual),
            "accuracy": accuracy_score(actual, predicted),
            "macro_f1": f1_score(actual, predicted, average="macro"),
            "confusion_matrix": matrix.tolist(),
            "recall_by_label": {label: float(recall) for label, recall in zip(labels, recalls)},
        }
        output = artifacts / f"evaluation-{architecture}.json"
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved {output}: accuracy={report['accuracy']:.3f}, macro_f1={report['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
