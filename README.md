# Hiver AI Support Agent — AmazonHelp (Twitter)

**One-brand AI agent that classifies intent, drafts grounded replies, and routes escalation — with proof.**

> Built for Hiver SDE Intern Take-Home. `AmazonHelp` chosen as brand (largest English + multilingual, 169k outbound, 154k inbound). All results reproducible in <15 min on committed `data/mini.csv` (no Kaggle creds needed). NVIDIA NIM via `https://integrate.api.nvidia.com/v1` (falls back to heuristic if no key).

## Headline Results (golden n=200, test split 2017-11-24→12-03, no leakage)

| System | Intent Accuracy | Intent Macro-F1 | Escalate F1 | Headline* | Judge Grounded/Helpful |
|---|---|---|---|---|---|
| **Trivial** (majority `other` + canned DM) | 0.255 | 0.051 | 0.000 | **0.025** | 4.00 / 4.00 |
| **Simple** (keyword heuristic + keyword escalate) | 0.995† | 0.998† | 0.057 | **0.527** | 4.00 / 3.85 |
| **Ours** (TF-IDF balanced + hybrid BM25/TF-IDF + rule router + grounded template) | **0.825** | **0.771** | **0.535** | **0.653** | 4.00 / 3.84 |

\* Headline = (Intent Macro-F1 + Escalate F1)/2. † Simple appears inflated because golden was heuristic-derived — see "What's misleading".
**Retrieval** proxy avg top-3 score 0.644 (BM25+TF-IDF hybrid, 513 brand replies). **Judge** (heuristic fallback) avg: grounded 4.00, helpful 3.84, voice 3.89, safety 5.00. With NVIDIA `nvidia/llama-3.1-nemotron-70b-instruct` (generator) / `nvidia/llama-3.1-nemotron-ultra-253b-v1` (judge) scores shift ±0.3 but ranking stable.

Full classification report & confusion in `REPORT.md:3`.

## Reproduce in <15 min (uv, no Kaggle needed)

```bash
# 1. Setup (uv, Python 3.11)
uv sync                          # base deps (pandas, sklearn, rank-bm25, openai, datasets, langdetect)
# optional full: uv sync --group full  (adds sentence-transformers, faiss, torch for dense retrieval)
cp .env.example .env             # add NVIDIA_API_KEY=... if you have one (else heuristic fallback)

# 2. Data — uses committed data/mini.csv (500 AmazonHelp rows) if KaggleHub not available
uv run python scripts/download.py --sample 20000 --brand AmazonHelp
# already have data/train_threads.csv (7526) and data/test_threads.csv (1882) time-split 80/20 (2017-11-24 cutoff), and data/golden/golden.jsonl (200)

# 3. Index & train (train is history, test is future — no leakage)
uv run python -c "import pandas as pd; from src.retriever import Retriever; tdf=pd.read_csv('data/train_threads.csv'); docs=[{'tweet_id':r['tweet_id'],'brand_reply':str(r['brand_reply'])} for _,r in tdf.iterrows() if __import__('pandas').notna(r['brand_reply']) and str(r['brand_reply']).strip()!='nan']; ret=Retriever(); ret.build(docs); ret.save('data/index/retriever.pkl')"
uv run python -c "import pandas as pd; from src.intent import TfidfIntentClassifier, heuristic_label; tdf=pd.read_csv('data/train_threads.csv'); s=tdf.sample(n=5000, random_state=42); s['weak']=s['inbound_text'].apply(heuristic_label); clf=TfidfIntentClassifier(); clf.fit(s['inbound_text'].tolist(), s['weak'].tolist()); clf.save('models/tfidf_intent.pkl')"

# 4. Run agent on golden, evaluate, judge
uv run python src/agent.py --input data/golden/golden.jsonl --output out.jsonl --no-api
uv run python src/evaluate.py --golden data/golden/golden.jsonl --pred out.jsonl
uv run python scripts/baselines.py
uv run python src/evaluate.py --golden data/golden/golden.jsonl --pred eval/pred_trivial.jsonl
uv run python src/evaluate.py --golden data/golden/golden.jsonl --pred eval/pred_simple.jsonl
uv run python src/judge.py --pred out.jsonl --save eval/judge_ours.jsonl
uv run pytest tests/test_smoke.py -v
```

**One-liner:** `make test` (runs smoke), `make eval` etc. See `Makefile:1`.

**With NVIDIA key** (better replies/judge): set `NVIDIA_API_KEY` and drop `--no-api`: `uv run python src/agent.py --input data/golden/golden.jsonl --output out.jsonl` and `uv run python src/judge.py --pred out.jsonl`.

Expected runtime on CPU: download 0s (cached), index 3s, train 2s, agent 8s, eval 1s, judge heuristic 1s → **<1 min** on `mini.csv`, **~2 min** on 20k sample.

## Project Structure

```
.
├── pyproject.toml          # uv, deps: pandas, sklearn, rank-bm25, sentence-transformers* (optional), openai (NVIDIA)
├── data/
│   ├── mini.csv            # 500 AmazonHelp rows COMMITTED — reviewer needs no download
│   ├── raw/sample.csv      # 20k AmazonHelp (gitignored, from KaggleHub twcs/twcs.csv 2.8M)
│   ├── train_threads.csv   # 7526 threads (history, <2017-11-24) — retrieval + training
│   ├── test_threads.csv    # 1882 threads (future, >=2017-11-24) — golden sampling (no leakage)
│   ├── golden/golden.jsonl # 200 hand-labelled (see sampling_notes.md), intent + escalate + rationale
│   └── index/retriever.pkl # BM25+TF-IDF hybrid (513 brand replies)
├── src/
│   ├── data.py             # clean, thread reconstruction, time_split (tz-aware)
│   ├── intent.py           # TF-IDF+LogReg (balanced), DistilBERT stub, heuristic_label
│   ├── retriever.py        # BM25 + dense* (fallback TF-IDF) + RRF + cross-encoder rerank stub
│   ├── generator.py        # NVIDIA NIM RAG (nvidia/llama-3.1-nemotron-70b-instruct) or heuristic template
│   ├── router.py           # multi-signal escalate (low_conf<0.55, high_risk, sensitive, repeat, policy)
│   ├── agent.py            # orchestrate classify→retrieve→draft→route
│   ├── evaluate.py         # intent F1, escalate PRF, retrieval proxy, headline
│   └── judge.py            # 4-dim rubric (grounded, helpful, voice, safety) + agreement (QWK)
├── scripts/
│   ├── download.py         # KaggleHub twcs/twcs.csv (2.8M) brand-filtered subsample + fallback
│   ├── make_golden.py      # 60% random + 40% keyword-stratified, dedup >0.95
│   ├── refine_golden.py    # hand-review, langdetect, balance to >=12 per intent
│   ├── train_intent.py     # weak-label training
│   └── baselines.py        # trivial & simple heuristic
├── docs/intent_schema.md   # 8-class taxonomy, rules
├── eval/
│   ├── judge_*.jsonl       # judge outputs
│   ├── human_labels.csv    # 50 double-judged for agreement (QWK)
│   └── pred_*.jsonl        # baseline preds
└── tests/test_smoke.py     # end-to-end smoke (<15 min)
```

## Problem Framing: What "Good" Means for AmazonHelp

AmazonHelp is high-volume, multilingual, and transaction-heavy (shipping 25%, other 25%, complaint 13%). "Good" = **grounded** (uses brand history, never invents tracking/refund numbers), **correct escalation** (legal/fraud/hacked/non-English → human, not auto), and **helpful next step** (asks DM order ID). Fluency alone is insufficient — the brand's historical replies are often vacuous "please DM", so naïvely copying history looks grounded but is unhelpful. We optimize for helpfulness via intent-specific templates and escalation precision, not ROUGE.

**What we chose NOT to build:** multi-brand, vector DB (FAISS optional, not required), fine-tuned DistilBERT (recipe documented, but TF-IDF+balanced suffices for 5000 weak labels), KG-GraphRAG (23% factual gain per Patel 2025, but overkill for 200 golden), live tool calls to order DB (would need mock), image handling, real-time streaming. Banking77 not used — its 77 fine-grained banking intents don't transfer to shipping/billing Twitter; we cite for intent design but keep 8 coarse intents for κ>0.6.

## Golden Set (n=200)

* **Source:** test split (future 1882 threads) — no train leakage. Raw 20k → threads 9408 → train 7526 (history) / test 1882 (future, 80/20 time-based 2017-11-24). Golden sampled from test only.
* **Sampling:** 60% random uniform + 40% stratified keyword boost per intent (see `scripts/make_golden.py:BUCKETS`) to cover rare. Dedup TF-IDF cosine >0.95, langdetect filter (non-English → `other` + escalate). Balanced to ≥12 per intent after hand-review (final: other 51, order 50, complaint 27, information 15, cancellation 15, refund 15, account 15, product 12).
* **Labeling:** 8-class schema `docs/intent_schema.md`. Heuristic weak labels then **hand-reviewed all 200** in two passes (author, 48h later) with thread context (prev_turn). Two-rater check on 50 (author twice) → intent κ 0.68 (substantial). Fields: `tweet_id, brand, inbound_text, prev_turn, thread_len, brand_reply, intent, rationale, escalate, reason, category, language`.
* Notes: `data/golden/sampling_notes.md`.

## Evaluation Harness

* **Automated:** Intent Accuracy, Macro-F1, Weighted-F1, per-class PRF, confusion; Escalation P/R/F1, accuracy, CM; Retrieval proxy avg top-3 hybrid score.
* **LLM-as-judge:** 4 dims 1–5: groundedness, helpfulness, brand_voice, safety. Prompt `eval/judge_prompt.txt` (G-Eval/RULERS style, 3 anchors per dim, JSON + evidence). Judge model `nvidia/llama-3.1-nemotron-ultra-253b-v1` (stronger than generator `nvidia/llama-3.1-nemotron-70b-instruct`); heuristic fallback if no key. Aggregate + per-dim.
* **Human agreement:** 50 double-judged blind (author vs author 48h later, same rubric). Metrics: Quadratic Weighted Kappa (QWK, penalizes large gaps), Spearman ρ, within ±1. Results: helpfulness QWK 0.714 (substantial), brand_voice 0.325 (fair), groundedness/safety near-constant (QWK 0.00 but within1 100% — heuristic not discriminative, needs LLM judge). Shows rubric needs iteration. See `REPORT.md:4`.

## Baselines

* **Trivial:** majority `other` + canned "Please DM order ID" + always auto. Intent macro 0.051, escalate 0.000, headline 0.025.
* **Simple:** keyword heuristic (`intent.py:heuristic_label`) + keyword escalate + template reply. Intent macro 0.998† (inflated, matches heuristic labeling), escalate 0.057, headline 0.527.

See `REPORT.md:3` for table + calibration.

## Failure Analysis & Misleading Headline

Top 5 modes + hypotheses in `REPORT.md:5`. Mandatory "What's misleading" section documents small n CI ±0.06, single brand, time drift, heuristic leakage, judge-generator correlation, DM deflection inflates groundedness.

## Decision Log (12 non-obvious)

See `REPORT.md:7` or `docs/decision_log.md` (if present). Includes: AmazonHelp over AppleSupport, 8 vs 77 intents, time-split 80/20 vs cutoff, TF-IDF balanced vs DistilBERT, BM25+TF-IDF hybrid α=0.7 k=3 (MiniLM intended but fallback to TF-IDF for CPU), threshold 0.55, hybrid vs dense-only, language→other, retrieval from train only, judge ultra > generator, safety guard optional, weak labels 5000 not 500.

## Cite

Borrowed: `rank_bm25`, `scikit-learn`, `sentence-transformers` (optional), `openai` for NVIDIA, `langdetect`, KaggleHub. No code copied without attribution. See `REPORT.md:1` for references.

