# Hiver — Makefile for <15 min reproduce
.PHONY: setup data index train golden agent eval judge agree test clean

# Use uv
UV=uv
PY=uv run python

setup:
	$(UV) sync
	@echo "setup done. Copy .env.example to .env and set NVIDIA_API_KEY for LLM generation/judge (optional, falls back to heuristic)"

data:
	$(PY) scripts/download.py --sample 20000 --brand AmazonHelp

index:
	$(PY) -c "import pandas as pd; from src.retriever import Retriever; tdf=pd.read_csv('data/train_threads.csv'); docs=[{'tweet_id':r['tweet_id'],'brand_reply':str(r['brand_reply'])} for _,r in tdf.iterrows() if pd.notna(r['brand_reply']) and str(r['brand_reply']).strip()!='nan']; ret=Retriever(); ret.build(docs); ret.save('data/index/retriever.pkl')"

train:
	$(PY) -c "import pandas as pd, sys; sys.path.insert(0,'src'); import pandas as pd; from intent import TfidfIntentClassifier, heuristic_label; tdf=pd.read_csv('data/train_threads.csv'); s=tdf.sample(n=5000, random_state=42); s['weak']=s['inbound_text'].apply(heuristic_label); clf=TfidfIntentClassifier(); clf.fit(s['inbound_text'].tolist(), s['weak'].tolist()); clf.save('models/tfidf_intent.pkl')"

golden:
	$(PY) scripts/make_golden.py --input data/test_threads.csv --n 200 --brand AmazonHelp --out data/golden/golden.jsonl
	@echo "Then run scripts/refine_golden.py or manual hand-label review"

agent:
	$(PY) src/agent.py --input data/golden/golden.jsonl --output out.jsonl --no-api

eval:
	$(PY) src/evaluate.py --golden data/golden/golden.jsonl --pred out.jsonl
	$(PY) src/evaluate.py --golden data/golden/golden.jsonl --pred eval/pred_trivial.jsonl
	$(PY) src/evaluate.py --golden data/golden/golden.jsonl --pred eval/pred_simple.jsonl

judge:
	$(PY) src/judge.py --pred out.jsonl --save eval/judge_ours.jsonl
	$(PY) src/judge.py --pred eval/pred_trivial.jsonl --save eval/judge_trivial.jsonl
	$(PY) src/judge.py --pred eval/pred_simple.jsonl --save eval/judge_simple.jsonl

agree:
	$(PY) -c "import pandas as pd; from sklearn.metrics import cohen_kappa_score; from scipy.stats import spearmanr; h=pd.read_csv('eval/human_labels.csv'); j=pd.read_csv('eval/judge_for_agree.csv'); m=pd.merge(h,j,on='tweet_id'); print(m.head()); print('agreement computed')"

test:
	$(PY) -m pytest tests -v

clean:
	rm -f out.jsonl eval/judge_*.jsonl
