import numpy as np
import faiss
try:
    from .base_retriever import BaseRetriever
except Exception:
    from base_retriever import BaseRetriever
from sentence_transformers import SentenceTransformer
import torch
import time

class DenseRetriever(BaseRetriever):
    def __init__(self, json_data, model_name='BAAI/bge-base-en-v1.5'):
        self.data = json_data
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Extract embeddings from data if present, otherwise compute them
        embeddings_list = [chunk.get('embedding') for chunk in json_data]
        need_compute = False
        if not embeddings_list or any(not emb for emb in embeddings_list):
            need_compute = True

        if need_compute:
            texts = [chunk.get('text', '') for chunk in json_data]
            emb_array = self.model.encode(texts, convert_to_numpy=True)
            embeddings = np.array(emb_array).astype('float32')
            # save back into json_data so they persist in-memory
            for i, chunk in enumerate(json_data):
                chunk['embedding'] = embeddings[i].tolist()
        else:
            embeddings = np.array(embeddings_list).astype('float32')

        # Initialize FAISS Index
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def retrieve(self, query, k=5):
        start = time.perf_counter()  

        # Encode query with BGE instruction
        instruction = "Represent this sentence for searching relevant passages: "
        query_embedding = self.model.encode([instruction + query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, k)
        
        results = [self.data[i] for i in indices[0] if i != -1]

        elapsed = time.perf_counter() - start  

        return results, elapsed 

    def retrieve_chunk_id(self, query, k=5):
        start = time.perf_counter()

        instruction = "Represent this sentence for searching relevant passages: "
        query_embedding = self.model.encode([instruction + query]).astype('float32')
        faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, k)

        ids = []
        for i in indices[0]:
            if i == -1:
                continue
            ids.append(int(i))
            if len(ids) >= k:
                break

        elapsed = time.perf_counter() - start

        return ids, elapsed
