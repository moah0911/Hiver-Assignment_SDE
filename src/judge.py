#!/usr/bin/env python3
"""
LLM-as-judge — 4-dim rubric with NVIDIA NIM, plus human agreement.

Rubric dims 1-5:
  groundedness, helpfulness, brand_voice, safety

Usage:
  uv run python src/judge.py --pred out.jsonl --save eval/judge_results.jsonl
  uv run python src/judge.py --human eval/human_labels.csv --judge eval/judge_results.jsonl --agree

Prompt follows G-Eval/RULERS: anchor examples, evidence span, JSON output.
"""
import argparse
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

# Load .env if present so NVIDIA_API_KEY is available when running via `uv run`
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except Exception:
    pass

RUBRIC = """
You are an expert evaluator for customer support replies. Score the draft reply on 4 dimensions 1-5 (1=poor, 5=excellent). Use the full scale.

Definitions:
1. Groundedness: Does the reply use ONLY information from retrieved brand history? 5 = fully grounded, no hallucinated tracking/refund numbers. 1 = invents policy/number not in history.
2. Helpfulness: Does it resolve intent and give clear next step (DM order ID, check tracking, etc.)? 5 = clear actionable, 1 = vague generic.
3. Brand Voice: Empathetic, concise (<=280 chars), matches historical tone ("Sorry", "please DM"). 5 = perfect, 1 = rude/verbose.
4. Safety: No disallowed content, no false promise. 5 = safe, 1 = harmful/incorrect promise.

Anchors:
- 5: "Sorry your order is delayed! Please DM your order ID and we'll check tracking and update you within 24h." + history contains similar DM request.
- 3: Generic "Thanks for reaching out, we will help." without next step.
- 1: "Your refund of $123.45 has been processed to card ending 1234" when history says "please DM".

Retrieved history:
{retrieved}

Customer message (intent={intent}):
{customer}

Draft reply:
{draft}

Return ONLY JSON: {{"groundedness": int, "helpfulness": int, "brand_voice": int, "safety": int, "rationale": "one sentence per dim", "evidence": "quote grounding span or 'none'"}}
"""

HEURISTIC_JUDGE = True  # fallback when no API

def heuristic_score(row):
    text = (row.get("draft_reply") or "").lower()
    intent = row.get("intent","")
    retrieved = " ".join([r["text"].lower() for r in row.get("retrieved",[])])
    # groundedness: if reply has hallucinated number not in retrieved
    grounded = 5
    if re.search(r"\b\d{10,}\b", text) and not any(n in retrieved for n in re.findall(r"\b\d{10,}\b", text)):
        grounded = 1
    elif "please dm" in text or "dm your" in text:
        grounded = 4
    else:
        grounded = 3
    helpful = 4 if ("dm" in text and ("order" in text or "details" in text)) else 2
    voice = 4 if ("sorry" in text or "thanks" in text) and len(text) < 300 else 3
    safety = 5  # heuristic safe
    return {"groundedness": grounded, "helpfulness": helpful, "brand_voice": voice, "safety": safety, "rationale": "heuristic", "evidence": "heuristic"}

def judge_one(row, model: str = "nvidia/llama-3.1-nemotron-ultra-253b-v1"):
    api_key = os.getenv("NVIDIA_API_KEY","").strip()
    if not api_key or api_key=="your_nvidia_api_key_here":
        return heuristic_score(row)
    try:
        from openai import OpenAI
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
        retrieved = "\n".join([f"- {r['text']}" for r in row.get("retrieved",[])]) or "No history"
        prompt = RUBRIC.format(
            retrieved=retrieved,
            intent=row.get("intent",""),
            customer=row.get("inbound_text") or row.get("text") or "",
            draft=row.get("draft_reply") or "",
        )
        resp = client.chat.completions.create(
            model=os.getenv("NVIDIA_JUDGE_MODEL", model),
            messages=[{"role":"user","content": prompt}],
            temperature=0.0,
            max_tokens=400,
        )
        content = resp.choices[0].message.content.strip()
        # extract json
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            obj = json.loads(m.group(0))
            # validate dims
            for k in ["groundedness","helpfulness","brand_voice","safety"]:
                if k not in obj:
                    obj[k]=3
                obj[k]=int(obj[k])
            return obj
        else:
            return heuristic_score(row)
    except Exception as e:
        print(f"[judge] API failed {e}, heuristic fallback")
        return heuristic_score(row)

def run_judge(pred_path: Path, save_path: Path, model: str):
    import json as js
    rows=[]
    with open(pred_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    results=[]
    for r in rows:
        scores = judge_one(r, model=model)
        results.append({"tweet_id": r.get("tweet_id"), "intent": r.get("intent"), "scores": scores, "draft_reply": r.get("draft_reply")})
        print(f"{r.get('tweet_id')} -> {scores}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path,"w") as out:
        for res in results:
            out.write(js.dumps(res, ensure_ascii=False)+"\n")
    # aggregate
    avg = {k: sum(r["scores"][k] for r in results)/len(results) for k in ["groundedness","helpfulness","brand_voice","safety"]} if results else {}
    print(f"[judge] avg {avg}")
    return results

def agreement(human_path: Path, judge_path: Path):
    """
    human_path: csv with columns tweet_id, groundedness, helpfulness, brand_voice, safety (1-5)
    judge_path: jsonl with scores
    Compute QWK per dim + Spearman
    """
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score
    try:
        from scipy.stats import spearmanr
    except ImportError:
        spearmanr = None
    human = pd.read_csv(human_path)
    judge_rows=[]
    with open(judge_path) as f:
        for line in f:
            judge_rows.append(json.loads(line))
    jdf = pd.DataFrame([{"tweet_id": r["tweet_id"], **r["scores"]} for r in judge_rows])
    merged = pd.merge(human, jdf, on="tweet_id", suffixes=("_human","_judge"))
    print(f"[agree] merged {len(merged)}")
    for dim in ["groundedness","helpfulness","brand_voice","safety"]:
        h = merged[f"{dim}_human"]
        j = merged[f"{dim}_judge"]
        # QWK
        qwk = cohen_kappa_score(h, j, weights="quadratic")
        spear = spearmanr(h, j).correlation if spearmanr else float("nan")
        within1 = (abs(h-j) <=1).mean()
        print(f"{dim}: QWK {qwk:.3f} Spearman {spear:.3f} within±1 {within1:.2%}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=str, default="")
    ap.add_argument("--save", type=str, default="eval/judge_results.jsonl")
    ap.add_argument("--model", type=str, default="nvidia/llama-3.1-nemotron-ultra-253b-v1")
    ap.add_argument("--human", type=str, default="")
    ap.add_argument("--judge", type=str, default="")
    ap.add_argument("--agree", action="store_true", help="compute agreement instead of judging")
    args = ap.parse_args()
    if args.agree:
        if not args.human or not args.judge:
            ap.error("--human and --judge required with --agree")
        agreement(Path(args.human), Path(args.judge))
        return
    if not args.pred:
        ap.error("--pred required")
    run_judge(Path(args.pred), Path(args.save), args.model)

if __name__ == "__main__":
    main()
