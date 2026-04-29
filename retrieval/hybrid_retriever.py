try:
    from .base_retriever import BaseRetriever
    from .bm25_retriever import BM25Retriever
    from .dense_retriever import DenseRetriever
except Exception:
    from base_retriever import BaseRetriever
    from bm25_retriever import BM25Retriever
    from dense_retriever import DenseRetriever

import time
import math


class HybridRetriever(BaseRetriever):
    def __init__(self, bm25_retriever, dense_retriever):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever

    # -----------------------------
    # Reciprocal Rank Fusion (RRF)
    # -----------------------------
    def _rrf_score(self, rank, k=60):
        return 1 / (k + rank)

    # -----------------------------
    # Hybrid retrieval
    # -----------------------------
    def retrieve(self, query, k=5, top_n=10):
        start = time.perf_counter()

        # 1. Get results from both systems
        bm25_docs, _ = self.bm25.retrieve(query, k=top_n)
        dense_docs, _ = self.dense.retrieve(query, k=top_n)

        # 2. Score dictionary
        scores = {}

        # 3. BM25 scoring
        for rank, doc in enumerate(bm25_docs):
            doc_id = doc.get("chunk_id", str(doc["text"]))
            scores[doc_id] = scores.get(doc_id, 0) + self._rrf_score(rank)

        # 4. Dense scoring
        for rank, doc in enumerate(dense_docs):
            doc_id = doc.get("chunk_id", str(doc["text"]))
            scores[doc_id] = scores.get(doc_id, 0) + self._rrf_score(rank)

        # 5. Merge documents
        all_docs = {doc.get("chunk_id", str(doc["text"])): doc for doc in bm25_docs + dense_docs}

        # 6. Sort by score
        ranked_docs = sorted(
            all_docs.values(),
            key=lambda doc: scores.get(doc.get("chunk_id", str(doc["text"])), 0),
            reverse=True
        )

        elapsed = time.perf_counter() - start

        return ranked_docs[:k], elapsed

    def retrieve_chunk_id(self, query, k=5, top_n=10):
        start = time.perf_counter()

        bm25_ids, _ = self.bm25.retrieve_chunk_id(query, k=top_n)
        dense_ids, _ = self.dense.retrieve_chunk_id(query, k=top_n)

        scores = {}

        for rank, id_val in enumerate(bm25_ids):
            scores[id_val] = scores.get(id_val, 0) + self._rrf_score(rank)

        for rank, id_val in enumerate(dense_ids):
            scores[id_val] = scores.get(id_val, 0) + self._rrf_score(rank)

        # sort ids by score
        ranked_ids = sorted(scores.keys(), key=lambda i: scores.get(i, 0), reverse=True)

        elapsed = time.perf_counter() - start

        return ranked_ids[:k], elapsed