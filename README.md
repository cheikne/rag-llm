# RAG LLM Project

This project implements a simple Retrieval-Augmented Generation (RAG) system using:
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

# Hugging Face Token (for private models like Mistral)
export HF_TOKEN="your_hf_token"

# If you wanna make Variables Permanent (Mac/Linux)
echo 'export SERPAPI_API_KEY="your_serpapi_key"' >> ~/.zshrc
echo 'export HF_TOKEN="your_hf_token"' >> ~/.zshrc
source ~/.zshrc

# Run the Project
python main.py
<!-- gemma:7b -->