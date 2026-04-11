import json
import torch
from sentence_transformers import SentenceTransformer

class ChunkEmbedder:
    def __init__(self, model_name='BAAI/bge-base-en-v1.5'):
        # Check if Mac GPU (MPS) is available
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading model '{model_name}' on device: {self.device}")
        
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_chunks(self, json_data):
        """
        Takes the list of chunks and fills the 'embedding' field.
        """
        # Extract all texts to process them in a single batch (faster)
        texts = [chunk["text"] for chunk in json_data]
        
        print(f"Computing embeddings for {len(texts)} chunks...")
        
        # BGE models work best with this specific instruction for retrieval
        instruction = "Represent this sentence for searching relevant passages: "
        texts_with_instructions = [instruction + t for t in texts]
        
        # Generate embeddings
        embeddings = self.model.encode(texts_with_instructions, show_progress_bar=True)
        
        # Update the JSON data with the new embeddings (converted to list for JSON)
        for i, chunk in enumerate(json_data):
            chunk["embedding"] = embeddings[i].tolist()
            
        return json_data

# --- HOW TO INTEGRATE ---
if __name__ == "__main__":
    # 1. Load your previously generated JSON
    with open("processed_data.json", "r") as f:
        data = json.load(f)

    # 2. Run embedding
    embedder = ChunkEmbedder()
    updated_data = embedder.embed_chunks(data)

    # 3. Save the final version
    with open("rag_database_with_vectors.json", "w") as f:
        json.dump(updated_data, f, indent=4)
    print("Done! Your RAG database is now vectorized.")
