#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import load_raw, build_threads
from retriever import build_from_threads

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=str(ROOT/"data/raw/sample.csv"))
    ap.add_argument("--out", type=str, default=str(ROOT/"data/index/retriever.pkl"))
    ap.add_argument("--brand", type=str, default="")
    args = ap.parse_args()
    df = load_raw(args.input)
    if args.brand:
        df = df[df["brand"]==args.brand]
        print(f"[build_index] filtered brand={args.brand} -> {len(df)} rows")
    tdf = build_threads(df)
    print(f"[build_index] threads {len(tdf)} (brand replies available {(tdf['brand_reply']!='').sum()})")
    if len(tdf)==0:
        # fallback: use raw as threads
        tdf = pd.DataFrame([{"tweet_id": str(r.tweet_id), "brand_reply": r.text_clean, "inbound_text": r.text_clean} for r in df.itertuples()])
    ret = build_from_threads(tdf, Path(args.out))
    # quick test query
    res = ret.query("my order hasn't arrived tracking no update", top_k=3)
    print("[build_index] test query results:")
    for r in res:
        print(f"  score={r['score']:.3f} text={r['text'][:120]}")

if __name__ == "__main__":
    main()
