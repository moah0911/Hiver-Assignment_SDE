"""
Retriever — Local Hybrid BM25 + Dense + Reranker (no API)

Index = brand replies (grounding source). Query = inbound_text + optional prev_turn.
Hybrid fusion: α * norm(BM25) + (1-α) * dense_cosine -> RRF -> cross-encoder rerank top-10 -> top-3

Local embeddings: all-MiniLM-L6-v2 (384d) via sentence-transformers.
Falls back to TF-IDF cosine if sentence-transformers not installed or no torch.
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]

def tokenize(s: str) -> List[str]:
    return s.lower().split()

class Retriever:
    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha
        self.bm25 = None
        self.corpus: List[str] = []
        self.doc_ids: List[str] = []
        self.doc_texts_raw: List[str] = []
        self.embeddings: np.ndarray | None = None
        self.embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embed_model = None
        self.reranker = None
        self.tfidf_vec = None
        self.tfidf_mat = None
        self.use_dense = False

    def _try_load_dense(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer(self.embed_model_name)
            self.use_dense = True
            print(f"[retriever] loaded dense {self.embed_model_name}")
        except Exception as e:
            print(f"[retriever] dense fallback to TF-IDF: {e}")
            self.use_dense = False
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf_vec = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1,2))

    def _try_load_reranker(self):
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("[retriever] loaded reranker ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"[retriever] no reranker: {e}")
            self.reranker = None

    def build(self, docs: List[Dict], text_field: str = "brand_reply"):
        """
        docs: list of {id, brand_reply, inbound_text, ...} where brand_reply is retrieval corpus.
        Fallback to inbound_text if brand_reply empty.
        """
        # filter docs with non-empty brand_reply, else use inbound
        corpus_raw = []
        ids = []
        for d in docs:
            t = d.get(text_field, "") or d.get("brand_reply", "") or d.get("inbound_text", "")
            t = (t or "").strip()
            if not t:
                continue
            corpus_raw.append(t)
            ids.append(str(d.get("tweet_id", d.get("id", len(ids)))))
        self.doc_texts_raw = corpus_raw
        self.doc_ids = ids
        self.corpus = [tokenize(t) for t in corpus_raw]
        self.bm25 = BM25Okapi(self.corpus)
        print(f"[retriever] built BM25 over {len(self.corpus)} docs")

        # dense
        self._try_load_dense()
        if self.use_dense:
            # encode corpus
            self.embeddings = self.embed_model.encode(corpus_raw, show_progress_bar=False, normalize_embeddings=True)
            print(f"[retriever] dense embeddings shape {self.embeddings.shape}")
        else:
            # TF-IDF fallback
            self.tfidf_mat = self.tfidf_vec.fit_transform(corpus_raw)
            print(f"[retriever] TF-IDF fallback shape {self.tfidf_mat.shape}")
        self._try_load_reranker()

    def _bm25_scores(self, query: str) -> np.ndarray:
        q_tokens = tokenize(query)
        scores = np.array(self.bm25.get_scores(q_tokens), dtype=float)
        # normalize to 0-1 via min-max
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores = np.zeros_like(scores)
        return scores

    def _dense_scores(self, query: str) -> np.ndarray:
        if self.use_dense:
            q_emb = self.embed_model.encode([query], normalize_embeddings=True)[0]
            # cosine (already normalized)
            scores = self.embeddings @ q_emb
            # normalize 0-1
            scores = (scores + 1)/2
            return scores
        else:
            # TF-IDF cosine
            from sklearn.metrics.pairwise import cosine_similarity
            q_vec = self.tfidf_vec.transform([query])
            scores = cosine_similarity(q_vec, self.tfidf_mat).flatten()
            return scores

    def query(self, text: str, top_k: int = 3, rerank_k: int = 10) -> List[Dict]:
        if not self.bm25:
            raise ValueError("call build() first")
        q = (text or "").strip()
        if not q:
            return []
        bm25_s = self._bm25_scores(q)
        dense_s = self._dense_scores(q)
        hybrid = self.alpha * bm25_s + (1-self.alpha) * dense_s
        # get top rerank_k
        top_idx = np.argsort(hybrid)[::-1][:rerank_k]
        if self.reranker is not None and len(top_idx)>0:
            pairs = [(q, self.doc_texts_raw[i]) for i in top_idx]
            ce_scores = np.array(self.reranker.predict(pairs))
            # combine hybrid + ce (simple weighted)
            # normalize ce
            if ce_scores.max() > ce_scores.min():
                ce_n = (ce_scores - ce_scores.min())/(ce_scores.max()-ce_scores.min())
            else:
                ce_n = np.zeros_like(ce_scores)
            combined = 0.5*hybrid[top_idx] + 0.5*ce_n
            reranked = top_idx[np.argsort(combined)[::-1][:top_k]]
            ce_scores_sorted = ce_scores[np.argsort(combined)[::-1][:top_k]]
            # return with scores
            results = []
            for rank, idx in enumerate(reranked):
                results.append({
                    "doc_id": self.doc_ids[idx],
                    "text": self.doc_texts_raw[idx],
                    "score": float(combined[rank]),
                    "bm25": float(bm25_s[idx]),
                    "dense": float(dense_s[idx]),
                    "ce": float(ce_scores_sorted[rank]),
                })
            return results
        else:
            top = top_idx[:top_k]
            return [
                {"doc_id": self.doc_ids[i], "text": self.doc_texts_raw[i],
                 "score": float(hybrid[i]), "bm25": float(bm25_s[i]), "dense": float(dense_s[i])}
                for i in top
            ]

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "alpha": self.alpha,
                "corpus": self.corpus,
                "doc_ids": self.doc_ids,
                "doc_texts_raw": self.doc_texts_raw,
                "embeddings": self.embeddings,
                "use_dense": self.use_dense,
            }, f)
        print(f"[retriever] saved to {path}")

    def load(self, path: Path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.alpha = data["alpha"]
        self.corpus = data["corpus"]
        self.doc_ids = data["doc_ids"]
        self.doc_texts_raw = data["doc_texts_raw"]
        self.embeddings = data.get("embeddings")
        self.use_dense = data.get("use_dense", False)
        self.bm25 = BM25Okapi(self.corpus)
        if self.use_dense:
            self._try_load_dense()
            # if we loaded embeddings, keep them; else re-encode lazily
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf_vec = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1,2))
            self.tfidf_mat = self.tfidf_vec.fit_transform(self.doc_texts_raw)
        self._try_load_reranker()
        print(f"[retriever] loaded from {path} ({len(self.doc_ids)} docs)")

def build_from_threads(thread_df: pd.DataFrame, out_path: Path = ROOT/"data/index/retriever.pkl"):
    """Helper for scripts/build_index.py"""
    docs = []
    for r in thread_df.itertuples():
        # use brand_reply as doc; fallback to inbound if no reply
        docs.append({
            "tweet_id": r.tweet_id,
            "brand_reply": getattr(r, "brand_reply", ""),
            "inbound_text": getattr(r, "inbound_text", ""),
        })
    # also add all brand replies as separate docs for richer corpus
    ret = Retriever(alpha=0.7)
    ret.build(docs)
    ret.save(out_path)
    return ret
