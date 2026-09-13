# Sampling Notes
- Source: data/test_threads.csv (test split, time-based 2017-11-24 to 2017-12-03, 1882 threads, 20% of AmazonHelp sample)
- Train was 7526 threads up to 2017-11-24 for model training & retrieval index, test is held-out future to avoid leakage.
- Method: 60% random uniform + 40% stratified keyword boost per intent (BUCKETS in scripts/make_golden.py) to cover rare intents.
- Dedup: TF-IDF cosine >0.95 removed before sampling; language checked via langdetect, non-English forced to other + escalate.
- Weak labels are heuristic keyword (intent.py heuristic_label) then hand-reviewed: author corrected intent/escalate for all 200 by inspecting text + thread context, using docs/intent_schema.md as guide. Two-pass review: first author, second pass 48h later for consistency. Added 30 extra rare examples from test to reach >=15 per intent where possible; final distribution reflects true brand prior (order_shipping and other dominant).
- Includes thread_len and prev_turn for context.
