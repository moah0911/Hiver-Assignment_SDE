#!/usr/bin/env python3
"""
Sample and scaffold golden set 200. Human still must label intent/escalate.

Creates data/golden/golden.jsonl with weak labels for you to edit.

Usage:
  uv run python scripts/make_golden.py --input data/raw/sample.csv --n 200 --brand AmazonHelp
"""
import argparse
import json
import random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import load_raw, build_threads, dedup_by_text
from intent import heuristic_label
import re

ROOT = Path(__file__).resolve().parents[1]

# keyword buckets to stratify rare intents
BUCKETS = {
    "order_shipping": ["order","shipping","tracking","delivery","package"],
    "refund_billing": ["refund","charged","billing","invoice","payment"],
    "account_access": ["login","password","hacked","account"],
    "product_technical": ["crash","broken","not working","bug"],
    "cancellation_return": ["cancel","return","exchange"],
    "complaint_escalation": ["legal","manager","terrible","sue"],
    "information_request": ["ship to","policy","how to"],
}

def stratify_sample(tdf, n=200):
    # 60% random + 40% bucket-boost
    random.seed(42)
    n_rand = int(n*0.6)
    n_strat = n - n_rand
    rand_part = tdf.sample(n=min(n_rand, len(tdf)), random_state=42) if len(tdf)>=n_rand else tdf
    remaining = tdf.drop(rand_part.index) if len(rand_part)<len(tdf) else tdf
    # bucket sampling
    strat_rows = []
    per_bucket = max(1, n_strat // len(BUCKETS))
    for intent, kws in BUCKETS.items():
        pat = "|".join(kws)
        candidates = remaining[remaining["inbound_text"].str.contains(pat, case=False, na=False)]
        take = min(per_bucket, len(candidates))
        if take>0:
            strat_rows.append(candidates.sample(n=take, random_state=42))
    if strat_rows:
        strat_part = pd.concat(strat_rows).drop_duplicates()
        # fill up to n_strat with random from remaining
        if len(strat_part) < n_strat:
            need = n_strat - len(strat_part)
            pool = remaining.drop(strat_part.index, errors="ignore")
            if len(pool)>0:
                add = pool.sample(n=min(need, len(pool)), random_state=1)
                strat_part = pd.concat([strat_part, add])
    else:
        strat_part = remaining.sample(n=min(n_strat, len(remaining)), random_state=1) if len(remaining)>0 else rand_part
    combined = pd.concat([rand_part, strat_part]).drop_duplicates().head(n)
    # if still short, fill random
    if len(combined) < n:
        need = n - len(combined)
        pool = tdf.drop(combined.index, errors="ignore")
        if len(pool)>0:
            combined = pd.concat([combined, pool.sample(n=min(need, len(pool)), random_state=2)])
    return combined

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=str(ROOT/"data/raw/sample.csv"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--brand", type=str, default="AmazonHelp")
    ap.add_argument("--out", type=str, default=str(ROOT/"data/golden/golden.jsonl"))
    args = ap.parse_args()
    import pandas as pd
    df = load_raw(args.input)
    if args.brand and "brand" in df.columns:
        # keep brand but don't filter too aggressively
        filtered = df[df["brand"]==args.brand]
        if len(filtered) >= 500:
            df = filtered
            print(f"[golden] brand {args.brand} -> {len(df)}")
        else:
            print(f"[golden] brand {args.brand} has only {len(filtered)}, keeping all")
    tdf = build_threads(df)
    print(f"[golden] threads {len(tdf)}")
    if len(tdf) < args.n:
        print(f"[golden] warning: only {len(tdf)} threads, will duplicate to reach {args.n} with dedup disabled")
        sample = tdf
    else:
        # dedup first lightly
        tdf = dedup_by_text(tdf, thresh=0.95)
        print(f"[golden] after dedup {len(tdf)}")
        sample = stratify_sample(tdf, n=args.n)
    # create weak labels
    rows=[]
    for r in sample.itertuples():
        text = getattr(r, "inbound_text", "") or ""
        weak = heuristic_label(text)
        # escalation heuristic for scaffold
        escalate = weak=="complaint_escalation" or ("hacked" in text.lower()) or (getattr(r, "thread_len",1) >4)
        rows.append({
            "tweet_id": str(getattr(r, "tweet_id", "")),
            "brand": getattr(r, "brand", args.brand),
            "inbound_text": text,
            "prev_turn": getattr(r, "prev_turn", "") or "",
            "thread_len": int(getattr(r, "thread_len",1)),
            "brand_reply": getattr(r, "brand_reply",""),
            "created_at": str(getattr(r, "created_at","")),
            # human must verify / edit these:
            "intent": weak,
            "intent_rationale": f"heuristic: keyword match {weak}",
            "escalate": bool(escalate),
            "escalate_reason": "heuristic: complaint/sensitive/long thread" if escalate else "heuristic: confident auto",
            "escalate_category": "high_risk" if weak=="complaint_escalation" else ("sensitive" if "hacked" in text.lower() else "auto"),
            "needs_human_review": True,
        })
    # shuffle
    import random; random.seed(42); random.shuffle(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")
    print(f"[golden] wrote {args.out} ({len(rows)}) — PLEASE EDIT intent/escalate fields, then set needs_human_review=false")
    # also write sampling notes
    notes = ROOT/"data/golden/sampling_notes.md"
    with open(notes, "w") as nf:
        nf.write(f"""# Sampling Notes
- Source: {args.input} brand={args.brand}
- Raw rows {len(df)} -> threads {len(tdf)} -> golden {len(rows)}
- Method: 60% random uniform + 40% stratified keyword boost per intent bucket (see scripts/make_golden.py BUCKETS) to cover rare intents
- Dedup: TF-IDF cosine >0.95 removed before sampling
- Strategy: time-order preserved after shuffle (random_state 42)
- Weak labels are heuristic (scripts/make_golden.py heuristic_label) and MUST be hand-corrected before evaluation — this is your labeling protocol.
- Intent schema: docs/intent_schema.md
""")
    print(f"[golden] wrote {notes}")

if __name__ == "__main__":
    import pandas as pd
    main()
