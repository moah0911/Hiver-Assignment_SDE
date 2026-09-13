def test_pipeline_smoke():
    from pathlib import Path
    import json
    import pandas as pd
    # check data exists
    assert Path("data/raw/sample.csv").exists()
    assert Path("data/train_threads.csv").exists()
    assert Path("data/test_threads.csv").exists()
    assert Path("data/golden/golden.jsonl").exists()
    assert Path("models/tfidf_intent.pkl").exists()
    assert Path("data/index/retriever.pkl").exists()
    # check golden size 150-250
    rows=[json.loads(l) for l in open("data/golden/golden.jsonl")]
    assert 150 <= len(rows) <= 250
    # check out.jsonl if exists else run agent
    import sys
    sys.path.insert(0, "src")
    from agent import Agent
    ag=Agent()
    res=ag.handle("My order hasn't arrived, tracking shows no update")
    assert "intent" in res and "draft_reply" in res and "escalate" in res
    # check evaluate works
    from pathlib import Path as P
    import subprocess
    # simple check that files have required fields
    for r in rows:
        assert "intent" in r and "escalate" in r and "inbound_text" in r
    print("smoke passed", len(rows))

def test_intent_schema():
    from pathlib import Path
    assert Path("docs/intent_schema.md").exists()
    text=Path("docs/intent_schema.md").read_text()
    for intent in ["order_shipping","refund_billing","account_access","product_technical","cancellation_return","complaint_escalation","information_request","other"]:
        assert intent in text
