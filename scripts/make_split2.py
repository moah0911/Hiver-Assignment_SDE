#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from data import load_raw, build_threads
ROOT=Path(__file__).resolve().parents[1]
df=load_raw(ROOT/"data/raw/sample.csv")
tdf=build_threads(df)
tdf_sorted=tdf.sort_values("created_at")
split_idx=int(len(tdf_sorted)*0.8)
train=tdf_sorted.iloc[:split_idx]
test=tdf_sorted.iloc[split_idx:]
print(f"train {len(train)} test {len(test)}")
print(f"train {train['created_at'].min()} to {train['created_at'].max()}")
print(f"test {test['created_at'].min()} to {test['created_at'].max()}")
train.to_csv(ROOT/"data/train_threads.csv",index=False)
test.to_csv(ROOT/"data/test_threads.csv",index=False)
print("wrote")
