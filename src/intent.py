"""
Intent classifier — TF-IDF+LogReg (primary cheap) + DistilBERT loader + NVIDIA few-shot fallback.

Taxonomy 8 classes from docs/intent_schema.md
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
INTENTS = ["order_shipping","refund_billing","account_access","product_technical","cancellation_return","complaint_escalation","information_request","other"]
INTENT_TO_ID = {k:i for i,k in enumerate(INTENTS)}
ID_TO_INTENT = {i:k for k,i in INTENT_TO_ID.items()}

# keyword heuristics for weak labeling / fallback
KEYWORDS = {
    "order_shipping": ["order","shipping","delivery","tracking","arrived","package","shipment","delayed","where is my order"],
    "refund_billing": ["refund","charged","billing","charge","invoice","payment","double charged","billed"],
    "account_access": ["login","password","account","hacked","2fa","locked","sign in","reset"],
    "product_technical": ["crash","bug","not working","broken","defect","update","restart","won't turn on"],
    "cancellation_return": ["cancel","return","exchange","stop shipment","refund my order"],
    "complaint_escalation": ["legal","lawyer","manager","terrible","worst","sue","complaint","escalate"],
    "information_request": ["ship to","delivery time","policy","how to","do you","hours","when will"],
}

def heuristic_label(text: str) -> str:
    t = text.lower()
    scores = {k: sum(1 for kw in v if kw in t) for k,v in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    if scores[best]==0:
        return "other"
    # complaint override
    if any(kw in t for kw in KEYWORDS["complaint_escalation"]):
        return "complaint_escalation"
    return best

class TfidfIntentClassifier:
    def __init__(self, max_features=10000):
        self.pipe = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=max_features)),
            ("clf", LogisticRegression(max_iter=1000, n_jobs=None, class_weight="balanced")),
        ])
        self.fitted = False

    def fit(self, texts: List[str], labels: List[str]):
        y = [INTENT_TO_ID[l] for l in labels]
        self.pipe.fit(texts, y)
        self.fitted = True
        print(f"[intent] fit on {len(texts)} examples, classes {set(labels)}")

    def predict(self, texts: List[str]) -> List[Tuple[str, float]]:
        if not self.fitted:
            # heuristic fallback
            return [(heuristic_label(t), 0.4) for t in texts]
        probs = self.pipe.predict_proba(texts)
        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)
        return [(ID_TO_INTENT[p], float(c)) for p,c in zip(preds, confs)]

    def predict_one(self, text: str) -> Tuple[str, float]:
        return self.predict([text])[0]

    def save(self, path: Path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.pipe, f)
        print(f"[intent] saved to {path}")

    def load(self, path: Path):
        with open(path, "rb") as f:
            self.pipe = pickle.load(f)
        self.fitted = True
        print(f"[intent] loaded from {path}")

# DistilBERT wrapper — optional, loads if transformers+torch available
class DistilBertIntent:
    def __init__(self, model_path: str | Path | None = None):
        self.model = None
        self.tokenizer = None
        self.model_path = str(model_path) if model_path else "distilbert-base-uncased"
        self.fitted = False

    def try_load(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, num_labels=len(INTENTS))
            self.fitted = True  # pretrained, not fine-tuned
            print(f"[intent] loaded DistilBERT {self.model_path}")
            return True
        except Exception as e:
            print(f"[intent] DistilBERT not available: {e}")
            return False

    def predict(self, texts: List[str]) -> List[Tuple[str,float]]:
        if not self.fitted or self.model is None:
            return [(heuristic_label(t), 0.4) for t in texts]
        import torch
        self.model.eval()
        out = []
        for t in texts:
            enc = self.tokenizer(t, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1).squeeze().numpy()
                pred = int(np.argmax(probs))
                conf = float(np.max(probs))
                # map pred id to intent — if not fine-tuned, mapping is arbitrary, fallback to heuristic for now
                # we treat DistilBERT as not yet fine-tuned unless model_path contains "hiver"
                # so we don't mislabel
                if "hiver" not in self.model_path.lower() and "customer-support" not in self.model_path.lower():
                    # not fine-tuned → heuristic proxy
                    out.append((heuristic_label(t), 0.45))
                else:
                    out.append((ID_TO_INTENT[pred], conf))
        return out

def weak_labels_for_train(thread_df: pd.DataFrame, n: int = 5000) -> pd.DataFrame:
    """Generate weak labels via heuristics for training when no human labels."""
    df = thread_df.copy()
    if len(df) > n:
        df = df.sample(n=n, random_state=42)
    df["weak_intent"] = df["inbound_text"].apply(heuristic_label)
    return df

def train_and_save(thread_df: pd.DataFrame, out_path: Path = ROOT/"models/tfidf_intent.pkl", weak_n: int = 5000):
    weak = weak_labels_for_train(thread_df, n=weak_n)
    texts = weak["inbound_text"].tolist()
    labels = weak["weak_intent"].tolist()
    clf = TfidfIntentClassifier()
    clf.fit(texts, labels)
    clf.save(out_path)
    # quick eval on weak holdout
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(texts, labels, test_size=0.2, random_state=42)
    clf2 = TfidfIntentClassifier()
    clf2.fit(Xtr, ytr)
    preds = [p for p,_ in clf2.predict(Xte)]
    print(classification_report(yte, preds, zero_division=0))
    print(f"acc {accuracy_score(yte, preds):.3f} macro-F1 {f1_score(yte, preds, average='macro', zero_division=0):.3f}")
    return clf
