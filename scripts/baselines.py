#!/usr/bin/env python3
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from intent import heuristic_label
from router import decide_keyword_only, decide
from generator import heuristic_reply

import argparse

def trivial(golden_path, out_path):
    rows=[json.loads(l) for l in open(golden_path)]
    out=[]
    for r in rows:
        out.append({
            **r,
            "intent": "other",
            "intent_confidence": 1.0,
            "escalate": False,
            "escalate_reason": "trivial always auto",
            "escalate_category": "auto",
            "draft_reply": "Thanks for reaching out! Please DM your order ID so we can help.",
            "draft_model": "trivial",
            "retrieved": [],
        })
    with open(out_path,"w") as f:
        for o in out:
            f.write(json.dumps(o,ensure_ascii=False)+"\n")
    print(f"wrote trivial {out_path}")

def simple(golden_path, out_path):
    rows=[json.loads(l) for l in open(golden_path)]
    out=[]
    for r in rows:
        txt=r.get("inbound_text","")
        intent=heuristic_label(txt)
        # keyword escalation
        esc, reason, cat = decide_keyword_only(txt, intent, 0.6)
        # simple template reply
        reply=heuristic_reply(r.get("brand","AmazonHelp"), intent, txt, [])
        out.append({
            **r,
            "intent": intent,
            "intent_confidence": 0.6,
            "escalate": esc,
            "escalate_reason": reason,
            "escalate_category": cat,
            "draft_reply": reply,
            "draft_model": "heuristic",
            "retrieved": [],
        })
    with open(out_path,"w") as f:
        for o in out:
            f.write(json.dumps(o,ensure_ascii=False)+"\n")
    print(f"wrote simple {out_path}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--golden", default="data/golden/golden.jsonl")
    ap.add_argument("--outdir", default="eval")
    args=ap.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    trivial(args.golden, f"{args.outdir}/pred_trivial.jsonl")
    simple(args.golden, f"{args.outdir}/pred_simple.jsonl")
