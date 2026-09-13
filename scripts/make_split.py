#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from data import load_raw, build_threads, time_split

ROOT = Path(__file__).resolve().parents[1]
df = load_raw(ROOT/"data/raw/sample.csv")
tdf = build_threads(df)
print(f"threads {len(tdf)}")
train, test = time_split(tdf, cutoff="2017-06-01", test_ratio=0.2)
print(f"train {len(train)} test {len(test)}")
print(f"train dates {train['created_at'].min()} to {train['created_at'].max()}")
print(f"test dates {test['created_at'].min()} to {test['created_at'].max()}")
train.to_csv(ROOT/"data/train_threads.csv", index=False)
test.to_csv(ROOT/"data/test_threads.csv", index=False)
print("wrote data/train_threads.csv and data/test_threads.csv")
