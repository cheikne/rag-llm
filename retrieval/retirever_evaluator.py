import math
from bm25_retriever import BM25Retriever
from dense_retriever import DenseRetriever
from hybrid_retriever import HybridRetriever
from google_search import GoogleSearchAPI
import json, os

# ------------------------------------
# Metrics
# ------------------------------------

def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]

    hits = sum(
        1 for doc_id in retrieved_k
        if doc_id in relevant
    )

    return hits / k if k > 0 else 0


def recall_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]

    hits = sum(
        1 for doc_id in retrieved_k
        if doc_id in relevant
    )

    return hits / len(relevant) if len(relevant) > 0 else 0


def reciprocal_rank(retrieved, relevant):
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1 / i

    return 0


def ndcg_at_k(retrieved, relevant, k):

    dcg = 0

    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            dcg += 1 / math.log2(i + 1)

    ideal_hits = min(len(relevant), k)

    idcg = 0
    for i in range(1, ideal_hits + 1):
        idcg += 1 / math.log2(i + 1)

    if idcg == 0:
        return 0

    return dcg / idcg


# ------------------------------------
# Example dataset
# Replace with your real questions
# ------------------------------------

# dataset = [
#     {
#         "question": "What is the dual form of the SVM optimization problem?",
#         "relevant_chunks": [36, 39, 42]
#     },
#     {
#         "question": "What is boosting and how does it improve classification performance?",
#         "relevant_chunks": [5]
#     },
#     {
#         "question": "How is the linear regression problem solved using the least squares method?",
#         "relevant_chunks": [86, 87]
#     },

#     {
#         "question": "What is Bayes' rule and how is it applied in classification?",
#         "relevant_chunks": [58]
#     },

#      {
#         "question": "What is the role of the learning rate in stochastic gradient descent?",
#         "relevant_chunks": [67]
#     },
#     {
#         "question": "What is the impact of Ridge regularization on the bias-variance tradeoff?",
#         "relevant_chunks": [96, 97]
#     }
# ]

dataset = [
    {
        "question": "What is the dual form of the SVM optimization problem?",
        "relevant_chunks": [70, 71]
    },
    {
        "question": "What is boosting and how does it improve classification performance?",
        "relevant_chunks": [127,113]
    },
    {
        "question": "How is the linear regression problem solved using the least squares method?",
        "relevant_chunks": [17,16]
    },

    {
        "question": "What is Bayes' rule and how is it applied in classification?",
        "relevant_chunks": [228, 165]
    },

     {
        "question": "What is the role of the learning rate in stochastic gradient descent?",
        "relevant_chunks": [31, 136]
    },
    {
        "question": "What is the impact of Ridge regularization on the bias-variance tradeoff?",
        "relevant_chunks": [85, 86, 42, 43]
    }
]

# ------------------------------------
# Replace with YOUR retriever call
# ------------------------------------

def retrieve(question):

    # Example fake retrieval output
    # Replace this with:
    # docs, _ = your_retriever.retrieve(question)
    # return [d["chunk_id"] for d in docs]

    if "overfitting" in question.lower():
        return [12, 4, 19, 2, 9]

    if "gradient" in question.lower():
        return [3, 5, 8, 10, 11]

    if "regularization" in question.lower():
        return [8, 2, 7, 15, 20]

    return []


# ------------------------------------
# Evaluation
# ------------------------------------

def main():

    json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data_processor", "rag_vectors.json"))
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Initialize local retrievers
        bm25_engine = BM25Retriever(data)
        dense_engine = DenseRetriever(data)
        hybrid_engine = HybridRetriever(bm25_engine, dense_engine)
    else:
        print(f"Data file not found: {json_path}")
        print("Place the dataset at data_processor/chunk_embeddings.json (relative to project root) and retry.")
        return

    k = 5

    precision_scores = []
    recall_scores = []
    mrr_scores = []
    ndcg_scores = []

    for item in dataset:

        question = item["question"]
        relevant = item["relevant_chunks"]

        retrieved = bm25_engine.retrieve_chunk_id(question, k=k)[0]  # Example using DenseRetriever

        p = precision_at_k(
            retrieved,
            relevant,
            k
        )

        r = recall_at_k(
            retrieved,
            relevant,
            k
        )

        m = reciprocal_rank(
            retrieved,
            relevant
        )

        n = ndcg_at_k(
            retrieved,
            relevant,
            k
        )

        precision_scores.append(p)
        recall_scores.append(r)
        mrr_scores.append(m)
        ndcg_scores.append(n)

    avg_precision = (
        sum(precision_scores)
        / len(precision_scores)
    )

    avg_recall = (
        sum(recall_scores)
        / len(recall_scores)
    )

    avg_mrr = (
        sum(mrr_scores)
        / len(mrr_scores)
    )

    avg_ndcg = (
        sum(ndcg_scores)
        / len(ndcg_scores)
    )

    print("\n====== Evaluation Results Dense Retrieval======")
    print(f"Avg Precision@{k}: {avg_precision:.4f}")
    print(f"Avg Recall@{k}:    {avg_recall:.4f}")
    print(f"Avg MRR:          {avg_mrr:.4f}")
    print(f"Avg NDCG@{k}:      {avg_ndcg:.4f}")
    print("================================")


if __name__ == "__main__":
    main()