try:
    from .base_retriever import BaseRetriever
except Exception:
    from base_retriever import BaseRetriever
from rank_bm25 import BM25Okapi
import re
import time  
import numpy as np

class BM25Retriever(BaseRetriever):
    def __init__(self, json_data):
        self.data = json_data
        
        def simple_tokenize(text):
            return re.findall(r"\b\w+\b", text.lower())

        self.corpus = [chunk.get('text', '') for chunk in json_data]
        self.tokenized_corpus = [simple_tokenize(str(doc)) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query, k=5):
        start = time.perf_counter()  

        tokenized_query = re.findall(r"\b\w+\b", query.lower())
        top_docs = self.bm25.get_top_n(tokenized_query, self.data, n=k)

        elapsed = time.perf_counter() - start  

        return top_docs, elapsed 

    def retrieve_chunk_id(self, query, k=5):
        start = time.perf_counter()

        tokenized_query = re.findall(r"\b\w+\b", query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        # get top k indices sorted by score desc
        top_idx = np.argsort(scores)[::-1][:k]

        ids = [int(i) for i in top_idx]

        elapsed = time.perf_counter() - start

        return ids, elapsed