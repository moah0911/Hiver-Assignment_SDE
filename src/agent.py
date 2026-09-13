#!/usr/bin/env python3
"""
Orchestrator: classify -> retrieve -> draft -> route

Usage:
  uv run python src/agent.py --input data/golden/golden.jsonl --output out.jsonl
  uv run python src/agent.py --text "my order hasn't arrived"
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from intent import TfidfIntentClassifier, heuristic_label
from retriever import Retriever
from generator import generate_reply
from router import decide

ROOT = Path(__file__).resolve().parents[1]

class Agent:
    def __init__(self, retriever_path: Path = ROOT/"data/index/retriever.pkl", intent_path: Path = ROOT/"models/tfidf_intent.pkl"):
        self.retriever = Retriever()
        if Path(retriever_path).exists():
            try:
                self.retriever.load(retriever_path)
                print(f"[agent] loaded retriever {retriever_path}")
            except Exception as e:
                print(f"[agent] retriever load failed {e}, building tiny fallback")
                self._build_fallback_retriever()
        else:
            print(f"[agent] no retriever at {retriever_path}, using fallback")
            self._build_fallback_retriever()

        self.intent_clf = TfidfIntentClassifier()
        if Path(intent_path).exists():
            try:
                self.intent_clf.load(intent_path)
            except Exception as e:
                print(f"[agent] intent load failed {e}, using heuristic")
        else:
            print(f"[agent] no intent model at {intent_path}, using heuristic")

    def _build_fallback_retriever(self):
        docs = [
            {"tweet_id": "1", "brand_reply": "Sorry for the delay, please DM your order ID and we'll check tracking."},
            {"tweet_id": "2", "brand_reply": "We can help with billing, please DM your order ID and last 4 digits."},
            {"tweet_id": "3", "brand_reply": "For login issues, please try password reset and DM your email if still stuck."},
            {"tweet_id": "4", "brand_reply": "We're sorry for the experience, please DM details and a senior agent will review."},
        ]
        self.retriever.build(docs)

    def handle(self, inbound_text: str, brand: str = "AmazonHelp", prev_turn: str = "", thread_len: int = 1, use_api: bool = True) -> dict:
        # 1. classify
        intent, conf = self.intent_clf.predict_one(inbound_text) if self.intent_clf.fitted else (heuristic_label(inbound_text), 0.45)
        # 2. retrieve — query = inbound + prev_turn
        query = inbound_text + (" " + prev_turn if prev_turn else "")
        retrieved = self.retriever.query(query, top_k=3)
        # 3. draft
        gen = generate_reply(brand, inbound_text, intent, retrieved, prev_turn, use_api=use_api)
        reply = gen["reply"]
        # 4. route
        escalate, reason, cat = decide(inbound_text, intent, conf, thread_len, prev_turn)
        return {
            "intent": intent,
            "intent_confidence": conf,
            "retrieved": retrieved,
            "draft_reply": reply,
            "draft_model": gen.get("model", ""),
            "escalate": escalate,
            "escalate_reason": reason,
            "escalate_category": cat,
        }

def run_batch(in_path: Path, out_path: Path, use_api: bool = True):
    agent = Agent()
    # support jsonl with fields: text / inbound_text / tweet_text
    rows = []
    with open(in_path) as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(obj)
    out = []
    for r in rows:
        text = r.get("inbound_text") or r.get("text") or r.get("tweet_text") or ""
        prev = r.get("prev_turn") or r.get("prev") or ""
        brand = r.get("brand") or "AmazonHelp"
        thread_len = int(r.get("thread_len") or 1)
        res = agent.handle(text, brand=brand, prev_turn=prev, thread_len=thread_len, use_api=use_api)
        out.append({**r, **res})
    with open(out_path, "w") as outf:
        for o in out:
            outf.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"[agent] wrote {out_path} ({len(out)} rows)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="")
    ap.add_argument("--output", type=str, default="out.jsonl")
    ap.add_argument("--text", type=str, default="")
    ap.add_argument("--no-api", action="store_true", help="force heuristic generator")
    ap.add_argument("--brand", type=str, default="AmazonHelp")
    args = ap.parse_args()

    if args.text:
        ag = Agent()
        res = ag.handle(args.text, brand=args.brand, use_api=not args.no_api)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return

    if not args.input:
        ap.error("--input or --text required")
    run_batch(Path(args.input), Path(args.output), use_api=not args.no_api)

if __name__ == "__main__":
    main()
