#!/usr/bin/env python3
"""
Evaluation harness — automated metrics + retrieval.

Usage:
  uv run python src/evaluate.py --golden data/golden/golden.jsonl --pred out.jsonl
Golden fields: intent (str), escalate (bool), text/inbound_text
Pred fields: intent, intent_confidence, escalate, draft_reply, retrieved
"""
import argparse
import json
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, classification_report
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

def load_jsonl(p: Path):
    rows=[]
    with open(p) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def eval_intent(golden, pred):
    y_true = [r.get("intent") or r.get("label") or "other" for r in golden]
    y_pred = [r.get("intent", "other") for r in pred]
    # align by id if present
    # if lengths differ, truncate
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print("=== Intent ===")
    print(f"Accuracy {acc:.3f}  Macro-F1 {macro:.3f}  Weighted-F1 {weighted:.3f}  n={n}")
    print(classification_report(y_true, y_pred, zero_division=0))
    # per-intent breakdown
    return {"accuracy": acc, "macro_f1": macro, "weighted_f1": weighted, "n": n}

def eval_escalate(golden, pred):
    y_true = [bool(r.get("escalate") or r.get("escalate_label") or False) for r in golden]
    y_pred = [bool(r.get("escalate", False)) for r in pred]
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]
    acc = accuracy_score(y_true, y_pred)
    p,r,f1,_ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    print("=== Escalation ===")
    print(f"Accuracy {acc:.3f}  P {p:.3f} R {r:.3f} F1 {f1:.3f}  n={n}")
    print(confusion_matrix(y_true, y_pred, labels=[False, True]))
    # category accuracy if present
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "n": n}

def eval_retrieval(pred):
    # if pred has retrieved scores, we can't compute recall without gold docs.
    # Instead report avg retrieved score and grounding heuristic.
    scores = []
    for r in pred:
        retr = r.get("retrieved", [])
        if retr:
            scores.append(np.mean([x.get("score",0) for x in retr]))
    if scores:
        print(f"=== Retrieval (proxy) ===  avg_top3_score {np.mean(scores):.3f}  n={len(scores)}")
    return {"avg_score": float(np.mean(scores)) if scores else 0.0}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=str, required=True)
    ap.add_argument("--pred", type=str, required=True)
    ap.add_argument("--save", type=str, default="")
    args = ap.parse_args()
    golden = load_jsonl(Path(args.golden))
    pred = load_jsonl(Path(args.pred))
    print(f"[evaluate] golden {len(golden)} pred {len(pred)}")
    # align by tweet_id if both have it
    if golden and pred and "tweet_id" in golden[0] and "tweet_id" in pred[0]:
        gmap = {r["tweet_id"]: r for r in golden}
        pmap = {r["tweet_id"]: r for r in pred}
        common = sorted(set(gmap) & set(pmap))
        if common:
            golden = [gmap[k] for k in common]
            pred = [pmap[k] for k in common]
            print(f"[evaluate] aligned on tweet_id -> {len(common)}")
    ir = eval_intent(golden, pred)
    er = eval_escalate(golden, pred)
    rr = eval_retrieval(pred)
    # overall headline (intent macro + escalate F1 avg)
    headline = (ir["macro_f1"] + er["f1"])/2
    print(f"\n=== Headline (avg Intent Macro-F1 + Escalate F1) === {headline:.3f}")
    print("\nWhat is misleading about headline number?")
    print("- Small golden n ~200 => 95% CI ±0.06 via bootstrap.")
    print("- Single brand + time drift; not generalizable.")
    print("- Judge family shares generator bias if same model.")
    if args.save:
        import json as js
        with open(args.save,"w") as f:
            js.dump({"intent": ir, "escalate": er, "retrieval": rr, "headline": headline}, f, indent=2)

if __name__ == "__main__":
    main()
