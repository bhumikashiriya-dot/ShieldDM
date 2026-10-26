
"""
ShieldDM - Final Model Training
Input:  train.csv
Output: model.joblib, model.pkl, metrics.json
"""

import os
import sys
import json
import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "train.csv")
MODEL_PATH = os.path.join(BASE, "model.joblib")
MODEL_PKL_PATH = os.path.join(BASE, "model.pkl")
METRICS_PATH = os.path.join(BASE, "metrics.json")

TOXIC_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

TEXT_CANDIDATES = [
    "comment_text", "text", "message", "msg",
    "content", "comment", "chat", "sentence"
]

SAFE_LABELS = {
    "safe", "normal", "neutral", "clean", "ok",
    "none", "non-toxic", "nontoxic", "not harmful",
    "non harmful", "benign", "0", "no", "false", "harmless"
}


def build_pipeline():
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_features=30000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    classifier = SGDClassifier(
        loss="log_loss",
        max_iter=10,
        tol=None,
        alpha=1e-5,
        class_weight="balanced",
        random_state=42,
        average=True,
    )

    return Pipeline([
        ("features", vectorizer),
        ("clf", classifier),
    ])


def main():
    if not os.path.exists(DATA_PATH):
        sys.exit("ERROR: train.csv not found next to train.py.")

    print("Loading dataset...")

    try:
        df = pd.read_csv(DATA_PATH, low_memory=False)
    except Exception as exc:
        sys.exit(f"ERROR reading train.csv: {exc}")

    df.columns = df.columns.str.strip().str.lower()

    text_col = next(
        (col for col in TEXT_CANDIDATES if col in df.columns),
        None
    )

    if text_col is None:
        sys.exit(
            "ERROR: No message column found. "
            "Expected comment_text, text, message, or comment."
        )

    toxicity_cols = [
        col for col in TOXIC_COLUMNS if col in df.columns
    ]

    if toxicity_cols:
        print("Using toxicity columns:", toxicity_cols)

        flags = df[toxicity_cols].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0)

        df["label"] = np.where(
            flags.gt(0).any(axis=1),
            "harassment",
            "safe"
        )
    else:
        label_col = next(
            (
                col for col in [
                    "label", "category", "class",
                    "target", "type", "tag", "y"
                ]
                if col in df.columns and col != text_col
            ),
            None
        )

        if label_col is None:
            sys.exit(
                "ERROR: No toxicity flags or label column found."
            )

        values = df[label_col].astype(str).str.strip().str.lower()

        df["label"] = values.apply(
            lambda value: (
                "safe" if value in SAFE_LABELS else "harassment"
            )
        )

    df = df[[text_col, "label"]].dropna().copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""]

    # Remove exact duplicate message-label pairs.
    df = df.drop_duplicates(
        subset=[text_col, "label"]
    ).reset_index(drop=True)

    if len(df) < 10 or df["label"].nunique() < 2:
        sys.exit(
            "ERROR: Need at least 10 messages and both safe "
            "and harassment examples."
        )

    X = df[text_col].to_numpy()
    y = df["label"].to_numpy()

    counts = pd.Series(y).value_counts().to_dict()

    print(f"Message column: {text_col}")
    print(f"Total messages: {len(df):,}")
    print("Label counts:", counts)

    print("\nSplitting dataset...")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

    # Evaluate using the held-out test set.
    print("\nTraining evaluation model...")

    evaluation_model = build_pipeline()
    evaluation_model.fit(X_train, y_train)

    predictions = evaluation_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    labels = ["safe", "harassment"]

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    ).tolist()

    actual_binary = (y_test == "harassment").astype(int)
    predicted_binary = (predictions == "harassment").astype(int)

    bp, br, bf, _ = precision_recall_fscore_support(
        actual_binary,
        predicted_binary,
        average="binary",
        zero_division=0,
    )

    binary_accuracy = accuracy_score(
        actual_binary, predicted_binary
    )

    # Train the final deployable model on all available messages.
    print("\nTraining final model on all messages...")
    print("Please wait; training may take a few minutes.")

    final_model = build_pipeline()
    final_model.fit(X, y)

    print("\nSaving model files...")

    # Save both filenames for compatibility with the existing app.
    joblib.dump(final_model, MODEL_PATH, compress=3)
    joblib.dump(final_model, MODEL_PKL_PATH, compress=3)

    metrics = {
        "trained_at": datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M"
        ),
        "text_col": text_col,
        "label_col": "label",
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_counts": {
            str(key): int(value)
            for key, value in counts.items()
        },
        "labels": labels,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "binary": {
            "accuracy": float(binary_accuracy),
            "precision": float(bp),
            "recall": float(br),
            "f1": float(bf),
        },
        "cv_f1_mean": None,
        "cv_f1_std": None,
        "report": report,
        "confusion_matrix": matrix,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 score:  {f1:.3f}")

    print("\n=== Harmful vs Safe ===")
    print(f"Accuracy:  {binary_accuracy:.3f}")
    print(f"Precision: {bp:.3f}")
    print(f"Recall:    {br:.3f}")
    print(f"F1 score:  {bf:.3f}")

    print(f"\nSaved model:   {MODEL_PATH}")
    print(f"Saved model:   {MODEL_PKL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print("\nTRAINING COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
