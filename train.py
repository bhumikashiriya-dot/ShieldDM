"""Train ShieldDM's text classifier.
Input: dataset.csv (or jigsaw-toxic-comment-train.csv) in this folder.
Output: model.pkl and metrics.json.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

BASE = Path(__file__).parent
CANDIDATES = ["dataset.csv", "jigsaw-toxic-comment-train.csv", "train.csv"]
path = next((BASE / name for name in CANDIDATES if (BASE / name).exists()), None)
if path is None:
    raise FileNotFoundError("Put your labelled CSV beside train.py (dataset.csv or jigsaw-toxic-comment-train.csv).")

df = pd.read_csv(path).dropna(how="all")
text_col = next((c for c in ["comment_text", "text", "message", "Message"] if c in df.columns), None)
if not text_col:
    raise ValueError("CSV needs a text column: comment_text, text, message or Message.")

# Supports either a ready-made label column or the standard Jigsaw toxicity columns.
label_col = next((c for c in ["label", "category", "class", "target"] if c in df.columns), None)
if label_col:
    labels = df[label_col].astype(str).str.strip().str.lower()
    mapping = {"safe":"Safe", "non-toxic":"Safe", "non_toxic":"Safe",
               "threat":"Threat", "threatening":"Threat", "abusive":"Abusive",
               "toxic":"Abusive", "harassment":"Abusive", "normal":"Safe", "clean":"Safe"}
    y = labels.map(mapping)
    # Also accept numeric binary labels (0 = Safe, 1 = Abusive).
    if y.notna().sum() == 0:
        numeric = pd.to_numeric(df[label_col], errors="coerce")
        if numeric.notna().any():
            unique = sorted(numeric.dropna().unique())
            numeric_map = {0:"Safe", 1:"Abusive", 2:"Threat"}
            y = numeric.map(numeric_map)
else:
    toxic_cols = [c for c in ["toxic","severe_toxic","obscene","insult","identity_hate"] if c in df.columns]
    if not toxic_cols:
        raise ValueError("CSV needs a label column or Jigsaw toxicity columns.")
    threat = df["threat"].fillna(0).astype(float) if "threat" in df else 0
    toxic = df[toxic_cols].fillna(0).astype(float).max(axis=1)
    y = pd.Series("Safe", index=df.index)
    y.loc[toxic > 0] = "Abusive"
    y.loc[threat > 0] = "Threat"

data = pd.DataFrame({"text":df[text_col].astype(str), "label":y}).dropna()
data = data[data.text.str.strip().ne("") & data.label.isin(["Safe","Abusive","Threat"])]
if data.label.nunique() < 2 or data.label.value_counts().min() < 2:
    raise ValueError("Each training class needs at least two examples (Safe, Abusive, Threat).")

X_train, X_test, y_train, y_test = train_test_split(
    data.text, data.label, test_size=0.2, random_state=42, stratify=data.label
)
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=50000,
                              stop_words="english", min_df=1, sublinear_tf=True)),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
labels = sorted(data.label.unique())

# app.py loads (vectorizer, classifier), so save that exact format.
joblib.dump((model.named_steps["tfidf"], model.named_steps["classifier"]), BASE / "model.pkl")
metrics = {
    "accuracy":float(accuracy_score(y_test,pred)),
    "macro_f1":float(f1_score(y_test,pred,average="macro")),
    "train_rows":int(len(X_train)), "test_rows":int(len(X_test)),
    "labels":labels,
    "report":classification_report(y_test,pred,labels=labels,output_dict=True,zero_division=0),
    "confusion_matrix":confusion_matrix(y_test,pred,labels=labels).tolist()
}
(BASE / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(f"Training complete. Accuracy: {metrics['accuracy']:.3f} | Macro F1: {metrics['macro_f1']:.3f}")
print("Created model.pkl and metrics.json")
