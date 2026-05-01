# RAG LLM Project
## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system that combines:
- Retrieval methods (BM25, Dense, Hybrid, Google Search)
- A Large Language Model (Mistral via Hugging Face)
- Hugging Face Transformers (LLM - Mistral)
- SerpAPI (Google Search for retrieval)

---

## 1. Clone the Repository

git clone  https://github.com/cheikne/rag-llm.git
cd rag-llm

## Create Virtual Environment
python -m venv venv
source venv/bin/activate

# Install Dependencies

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install -r requirements.txt


# Set Environment Variables
# SerpAPI Key (Google Search)
export SERPAPI_API_KEY="your_serpapi_key"

# Hugging Face Token (for private models like Gemma 2-b)
export HF_TOKEN="your_hf_token"

# If you wanna make Variables Permanent (Mac/Linux)
echo 'export SERPAPI_API_KEY="your_serpapi_key"' >> ~/.zshrc
echo 'export HF_TOKEN="your_hf_token"' >> ~/.zshrc
source ~/.zshrc

# Run the Project
python main.py


## Architecture

The system follows a standard RAG pipeline:

User Query → Retriever → Context → LLM → Answer

Retrievers implemented:
- BM25 (keyword-based)
- Dense Retrieval (FAISS + embeddings)
- Hybrid Retrieval (RRF)
- Google Search (SerpAPI)


## Retrieval Methods

- BM25: keyword-based retrieval, strong for exact terms
- Dense Retrieval: embedding-based using FAISS, captures semantic similarity
- Hybrid Retrieval: combines BM25 and Dense using Reciprocal Rank Fusion (RRF)
- Google Search: external retrieval using SerpAPI


## Evaluation

We evaluate retrieval performance using:

- Precision@k  
- Recall@k  
- MRR (Mean Reciprocal Rank)  
- NDCG  

We also evaluate generated answers using:

- Exact Match  
- F1 Score  
- ROUGE-L  

The evaluation scripts are already implemented:
- retriver_metrics.py  
- model_metrics.py  

However, due to time constraints, we did not finalize the preparation of the evaluation dataset (ground truth annotations), so these metrics have not yet been fully computed.

## Project Structure

rag-llm/
│
├── main.py
├── model/
│   └── llm.py
├── retrieval/
│   ├── bm25_retriever.py
│   ├── dense_retriever.py
│   ├── hybrid_retriever.py
│   └── google_search.py
|   └── retriver_metrics.py
|   └── model_metrics.py
├── rag/
│   └── pipeline_rag.py
├── data_processor/
│   └── rag_vectors.json
├── evaluation/
│   ├── evaluation_metrics.py
│   └── model_evaluator.py
└── requirements.txt


## Data Source

- MIT OpenCourseWare (Machine Learning lectures)