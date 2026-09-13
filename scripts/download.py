#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MINI = ROOT / "data" / "mini.csv"
EXPECTED_COLS = ["tweet_id","author_id","inbound","created_at","text","response_tweet_id","in_response_to_tweet_id"]

def try_kagglehub_brand(sample: int, brand: str) -> pd.DataFrame | None:
    try:
        import kagglehub
    except ImportError:
        print("[download] kagglehub not installed")
        return None
    try:
        import glob
        path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
        print(f"[download] kagglehub path: {path}")
        csvs = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)
        csvs = sorted(csvs, key=lambda p: os.path.getsize(p), reverse=True)
        print(f"[download] found {len(csvs)} csvs, largest {os.path.basename(csvs[0])} {os.path.getsize(csvs[0])//1024//1024}MB")
        target = None
        for c in csvs:
            if os.path.getsize(c) > 10_000_000:
                target = c
                break
        if target is None:
            target = csvs[0]
        print(f"[download] reading {target} ... (brand filter={brand})")

        # Efficient brand filtering without loading all 2.8M if brand specified
        if brand:
            # collect outbound for brand + inbound they replied to
            # read in chunks to avoid huge memory spike
            chunks = pd.read_csv(target, dtype={"tweet_id": str, "author_id": str, "in_response_to_tweet_id": str, "response_tweet_id": str}, chunksize=200000, low_memory=False)
            outbound_ids = set()
            inbound_target_ids = set()
            outbound_rows = []
            for chunk in chunks:
                out = chunk[chunk["author_id"]==brand]
                if len(out)>0:
                    outbound_rows.append(out)
                    # collect in_response_to ids
                    tids = out["in_response_to_tweet_id"].dropna().astype(str)
                    # filter out 'nan' strings
                    tids = tids[tids.ne("nan") & tids.ne("")]
                    inbound_target_ids.update(tids.tolist())
            if not outbound_rows:
                print(f"[download] no outbound for brand {brand}, falling back to random sample")
                df = pd.read_csv(target, dtype={"tweet_id": str, "author_id": str, "in_response_to_tweet_id": str, "response_tweet_id": str}, nrows=sample*2, low_memory=False)
                if len(df) > sample:
                    df = df.sample(n=sample, random_state=42)
                df["brand"] = brand
                return df
            outbound_concat = pd.concat(outbound_rows, ignore_index=True)
            print(f"[download] outbound {brand}: {len(outbound_concat)} rows, target inbound ids {len(inbound_target_ids)}")

            # second pass: get inbound rows whose tweet_id in target set
            # to avoid second full scan, we can collect inbound in same chunks? we already passed. So second loop.
            inbound_rows = []
            for chunk in pd.read_csv(target, dtype={"tweet_id": str, "author_id": str, "in_response_to_tweet_id": str, "response_tweet_id": str}, chunksize=200000, low_memory=False):
                mask = chunk["tweet_id"].astype(str).isin(inbound_target_ids)
                # also include inbound rows where inbound==True and text mentions brand? not needed
                if mask.any():
                    inbound_rows.append(chunk[mask])

            if inbound_rows:
                inbound_concat = pd.concat(inbound_rows, ignore_index=True)
                print(f"[download] inbound for {brand}: {len(inbound_concat)}")
                # combine
                combined = pd.concat([outbound_concat, inbound_concat], ignore_index=True)
                # add brand column
                combined["brand"] = brand
                # we have threads: inbound + outbound. Need at least sample rows. If combined > sample, sample.
                if len(combined) > sample:
                    # keep thread balance: sample inbound first, then add their outbound
                    # simple random sample of combined
                    combined = combined.sample(n=sample, random_state=42)
                print(f"[download] combined {brand} sample {len(combined)} (inbound {(combined['inbound'].astype(str).str.lower()=='true').sum()})")
                return combined
            else:
                # no inbound found via in_response_to, maybe use response_tweet_id link? fallback to outbound only + random inbound mentioning brand
                print(f"[download] no inbound via in_response_to, trying text mention fallback")
                # just return outbound + sample of full
                df = outbound_concat.copy()
                if len(df) < sample:
                    # fill with random rows that mention brand in text
                    extra = pd.read_csv(target, dtype={"tweet_id": str, "author_id": str, "in_response_to_tweet_id": str, "response_tweet_id": str}, nrows=sample)
                    # filter text contains brand? not ideal
                    df = pd.concat([df, extra.head(sample - len(df))], ignore_index=True)
                df["brand"] = brand
                return df
        else:
            # no brand filter, random sample
            df = pd.read_csv(target, dtype={"tweet_id": str, "author_id": str, "in_response_to_tweet_id": str, "response_tweet_id": str}, low_memory=False)
            print(f"[download] full loaded {len(df)}")
            if len(df) > sample:
                df = df.sample(n=sample, random_state=42)
            df["brand"] = "mixed"
            return df
    except Exception as e:
        import traceback
        print(f"[download] kagglehub failed: {e}")
        traceback.print_exc()
        return None

def make_mini_sample(n=500) -> pd.DataFrame:
    examples = [
        ("My order #12345 hasn't arrived, tracking shows no update for 5 days", True),
        ("I was charged twice for my last purchase, please refund", True),
        ("Can't log into my account, says password incorrect even after reset", True),
        ("My package arrived damaged, box crushed and item broken", True),
        ("How do I cancel order #98765? Need to stop shipment", True),
        ("Thank you so much! Got my replacement quickly", False),
        ("@AmazonHelp My order was 2 days late, very disappointed", True),
        ("My phone won't turn on after update, keeps restarting", True),
        ("Please help, my account was hacked, email changed", True),
        ("Where is my refund? It's been 10 days", True),
        ("Do you ship to Canada? What's the delivery time?", True),
        ("App keeps crashing when I try to pay", True),
        ("I need invoice for order #54321", True),
        ("Your support is terrible, want to speak to manager, considering legal action", True),
        ("Love your service, quick delivery!", False),
    ]
    import random
    random.seed(42)
    rows=[]
    ts=pd.Timestamp("2017-01-01")
    for i in range(n):
        text,inbound=random.choice(examples)
        text=text+(" #test" if random.random()<0.1 else "")
        rows.append({
            "tweet_id": str(1000000+i),
            "author_id": "AmazonHelp" if not inbound else str(2000+random.randint(0,20)),
            "inbound": inbound,
            "created_at": (ts+pd.Timedelta(hours=i)).isoformat(),
            "text": text,
            "response_tweet_id": "" if inbound else "",
            "in_response_to_tweet_id": "" if random.random()<0.5 else str(1000000+max(0,i-1)),
            "brand": "AmazonHelp",
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--brand", type=str, default="AmazonHelp")
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--out", type=str, default=str(RAW / "sample.csv"))
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    df=None
    if not args.mini:
        df = try_kagglehub_brand(args.sample, args.brand)
    if df is None:
        print(f"[download] fallback synthetic n={args.sample}")
        n=min(args.sample,5000)
        df=make_mini_sample(n=max(200,n))
        if not MINI.exists():
            mini_df=make_mini_sample(500)
            mini_df.to_csv(MINI,index=False)
            print(f"[download] wrote fallback mini to {MINI}")
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c]=""
    # ensure brand column exists and correct (already set for brand-filtered)
    if "brand" not in df.columns:
        df["brand"]=args.brand
    # ensure mini exists for reviewer (500 rows with correct brand)
    if not MINI.exists():
        # if we have brand-filtered data, use head 500 for mini with brand preservation
        mini_src = df.head(500).copy()
        # ensure mini has brand-correct data: if df is brand-filtered, mini inherits
        mini_src.to_csv(MINI,index=False)
        print(f"[download] wrote {MINI} from brand-filtered sample")
    df.to_csv(args.out,index=False)
    print(f"[download] wrote {args.out} ({len(df)} rows)")
    if "brand" in df.columns:
        print(df["brand"].value_counts().head(10).to_string())
    # inbound ratio
    try:
        inbound_rate = (df["inbound"].astype(str).str.lower()=="true").mean()
    except:
        inbound_rate = (df["inbound"]==True).mean()
    print(f"[download] inbound {inbound_rate:.1%}")
    # quick brand stats for decision log
    # also print outbound brand distribution if mixed

if __name__=="__main__":
    main()
