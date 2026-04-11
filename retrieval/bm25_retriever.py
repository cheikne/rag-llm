from .base_retriever import BaseRetriever
from rank_bm25 import BM25Okapi
import re

class BM25Retriever(BaseRetriever):
    def __init__(self, json_data):
        self.data = json_data
        # Tokenize the corpus for BM25 using a simple regex tokenizer
        self.corpus = [chunk.get('text', '') for chunk in json_data]
        def simple_tokenize(text):
            return re.findall(r"\b\w+\b", text.lower())
        self.tokenized_corpus = [simple_tokenize(str(doc)) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query, k=5):
        tokenized_query = re.findall(r"\b\w+\b", query.lower())
        # Get the best documents based on keyword similarity
        top_docs = self.bm25.get_top_n(tokenized_query, self.data, n=k)
        return top_docs
