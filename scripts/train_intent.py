#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import load_raw, build_threads
from intent import train_and_save

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=str(ROOT/"data/raw/sample.csv"))
    ap.add_argument("--out", type=str, default=str(ROOT/"models/tfidf_intent.pkl"))
    ap.add_argument("--brand", type=str, default="")
    ap.add_argument("--weak_n", type=int, default=5000)
    args = ap.parse_args()
    df = load_raw(args.input)
    if args.brand:
        df = df[df["brand"]==args.brand]
    tdf = build_threads(df)
    print(f"[train] threads {len(tdf)}")
    if len(tdf)==0:
        # fallback
        tdf = df.rename(columns={"text_clean":"inbound_text"})[["text_clean"]].rename(columns={"text_clean":"inbound_text"})
        tdf["inbound_text"] = df["text_clean"]
    train_and_save(tdf, Path(args.out), weak_n=args.weak_n)

if __name__ == "__main__":
    main()
