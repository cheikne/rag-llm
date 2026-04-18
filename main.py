from model.llm import LLMModel
from rag.pipeline_rag import RAGPipeline
from retrieval.google_search import GoogleSearchAPI
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
import json, os, time

def clean_output(text):
    if "Answer:" in text:
        return text.split("Answer:")[-1].strip()
    return text.strip()
# def main():
#     llm = LLMModel()
#     rag = RAGPipeline(llm)

#     while True:
#         query = input("Enter your question (or 'exit' to quit): ")
#         if query.lower() == "exit":
#             break

#         answer = rag.run(query)
#         print("=========================ANSWER=====================")
#         print(clean_output(answer))

# if __name__ == "__main__":
#     main()

def main():
    # 1. Initialize LLM
    llm = LLMModel()
    rag = RAGPipeline(llm)

    # 2. Load Local Data for BM25 and Dense
    # SURPRISE
    json_path = "data_processor/rag_vectors.json"
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Initialize local retrievers
        bm25_engine = BM25Retriever(data)
        dense_engine = DenseRetriever(data)
    else:
        print(f"Warning: {json_path} not found. Local retrieval will be unavailable.")
        bm25_engine = dense_engine = None

    # 3. Google Search engine
    google_engine = GoogleSearchAPI()

    print("\nAvailable methods: 'bm25', 'dense', 'google', 'hybrid'")
    
    while True:
        method = input("\nSelect method (bm25/dense/google/exit): ").lower()
        if method == "exit":
            break
        
        query = input("Enter your question: ")

        # Selection of the retriever based on user choice
        if method == "bm25" and bm25_engine:
            retriever = bm25_engine
        elif method == "dense" and dense_engine:
            retriever = dense_engine
        elif method == "google":
            retriever = google_engine
        else:
            print("Invalid method or data not loaded.")
            continue

        # Run the pipeline with the selected retriever
        # Note: we pass k=3 to limit context size for the M3 RAM
        # added more lines for time to be calculated
        start_total = time.perf_counter()

        answer = rag.run(query, retriever=retriever, k=3)

        total_time = time.perf_counter() - start_total


        
        print(f"\n========================= ANSWER ({method.upper()}) =====================")
        print(clean_output(answer))
        print("================================================================")
        print(f"\nTotal Time ({method.upper()}): {total_time:.6f} sec")

if __name__ == "__main__":
    main()