"""
Data pipeline: clean, thread reconstruction, brand inference, splits.

Handles the 7-col Twitter dataset:
 tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id
"""
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[1]

# cleaning
URL_RE = re.compile(r"https?://\S+")
HANDLE_RE = re.compile(r"@\w+")
MASK_RE = re.compile(r"__\w+__")  # keep as token but normalize
WS_RE = re.compile(r"\s+")

def clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = URL_RE.sub(" [URL] ", t)
    # keep handles as token for brand but normalize
    t = HANDLE_RE.sub(" @user ", t)
    # keep masks like __email__ as token
    t = t.strip()
    t = WS_RE.sub(" ", t)
    return t

def normalize_inbound(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).lower().strip()
    return s in ("true","1","t","yes","y")

def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"tweet_id": str, "author_id": str, "in_response_to_tweet_id": str, "response_tweet_id": str})
    # normalize
    if "inbound" in df.columns:
        df["inbound"] = df["inbound"].apply(normalize_inbound)
    else:
        df["inbound"] = True
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    else:
        df["created_at"] = pd.Timestamp("2017-01-01")
    if "text" not in df.columns:
        df["text"] = ""
    df["text_clean"] = df["text"].astype(str).apply(clean_text)
    # ensure tweet_id string
    df["tweet_id"] = df["tweet_id"].astype(str)
    if "in_response_to_tweet_id" not in df.columns:
        df["in_response_to_tweet_id"] = ""
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].fillna("").astype(str)
    if "brand" not in df.columns:
        df["brand"] = "AmazonHelp"
    return df

def build_threads(df: pd.DataFrame) -> pd.DataFrame:
    """Create thread context: for each inbound tweet, attach brand reply and prev turn."""
    # index for quick lookup
    id_to_row = {str(r.tweet_id): r for r in df.itertuples()}
    # also build response map
    rows = []
    for r in df.itertuples():
        if not r.inbound:
            continue
        # find direct brand reply via in_response_to of outbound
        # outbound where in_response_to_tweet_id == r.tweet_id
        brand_replies = df[df["in_response_to_tweet_id"]==str(r.tweet_id)]
        # also check response_tweet_id field (comma separated)
        resp_text = ""
        if hasattr(r, "response_tweet_id") and str(r.response_tweet_id).strip() not in ("", "nan"):
            # take first response id
            rid = str(r.response_tweet_id).split(",")[0].strip()
            if rid in id_to_row:
                resp_text = id_to_row[rid].text_clean
        if not resp_text and len(brand_replies)>0:
            # take first outbound
            brand_replies = brand_replies[~brand_replies["inbound"]]
            if len(brand_replies)>0:
                resp_text = brand_replies.iloc[0]["text_clean"]
        # prev turn: if this inbound is itself a response, get parent
        parent_text = ""
        if str(r.in_response_to_tweet_id).strip() not in ("", "nan"):
            pid = str(r.in_response_to_tweet_id)
            if pid in id_to_row:
                parent_text = id_to_row[pid].text_clean
        rows.append({
            "tweet_id": str(r.tweet_id),
            "author_id": str(r.author_id),
            "created_at": r.created_at,
            "brand": getattr(r, "brand", "AmazonHelp"),
            "inbound_text": r.text_clean,
            "inbound_raw": r.text,
            "brand_reply": resp_text,
            "prev_turn": parent_text,
            "thread_len": 1 + (1 if resp_text else 0) + (1 if parent_text else 0),
            "inbound": True,
        })
    tdf = pd.DataFrame(rows)
    if len(tdf)>0:
        tdf = tdf.sort_values("created_at")
    return tdf

def brand_stats(df: pd.DataFrame, raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Print per-brand stats for brand selection."""
    if raw is not None:
        df = raw
    # use thread df if has brand
    if "brand" not in df.columns:
        return pd.DataFrame()
    grouped = df.groupby("brand").agg(
        total=("brand","size"),
        inbound=("inbound", lambda x: (x==True).sum() if x.dtype==bool else (x.astype(str)=="True").sum()),
    ) if "inbound" in df.columns else df.groupby("brand").size().reset_index(name="total")
    print(grouped.to_string())
    return grouped

def time_split(tdf: pd.DataFrame, cutoff: str = "2017-06-01", test_ratio: float = 0.15):
    """Time-based split. If no created_at, random split."""
    if "created_at" not in tdf.columns or tdf["created_at"].isna().all():
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(tdf, test_size=test_ratio, random_state=42)
        return train, test
    # normalize cutoff to tdf's tz
    cutoff_dt = pd.to_datetime(cutoff)
    if tdf["created_at"].dt.tz is not None:
        cutoff_dt = cutoff_dt.tz_localize("UTC")
    train = tdf[tdf["created_at"] < cutoff_dt]
    test = tdf[tdf["created_at"] >= cutoff_dt]
    if len(test) < 200:
        tdf_sorted = tdf.sort_values("created_at")
        split_idx = int(len(tdf_sorted)*(1-test_ratio))
        train = tdf_sorted.iloc[:split_idx]
        test = tdf_sorted.iloc[split_idx:]
    return train, test

def dedup_by_text(df: pd.DataFrame, thresh: float = 0.9) -> pd.DataFrame:
    """Lightweight near-dedup via TF-IDF cosine > thresh. Keep first."""
    if len(df) < 2:
        return df
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(stop_words="english", max_features=5000).fit_transform(df["inbound_text"].fillna(""))
        # for small df compute full matrix, for large do iterative
        if len(df) > 5000:
            return df  # skip for large
        sim = cosine_similarity(vec)
        # greedy keep
        keep = []
        kept_idx = set()
        for i in range(len(df)):
            if i in kept_idx:
                continue
            keep.append(i)
            # mark near-duplicates after i
            dup = np.where(sim[i] > thresh)[0]
            for j in dup:
                if j != i:
                    kept_idx.add(j)
        return df.iloc[keep].reset_index(drop=True)
    except Exception:
        return df
