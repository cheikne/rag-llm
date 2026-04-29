import re

# ------------------------------------
# Helpers
# ------------------------------------

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text


def tokenize(text):
    return normalize_text(text).split()


# ------------------------------------
# Exact Match
# ------------------------------------

def exact_match(prediction, reference):
    return int(
        normalize_text(prediction)
        == normalize_text(reference)
    )


# ------------------------------------
# Token-level F1
# ------------------------------------

def f1_score(prediction, reference):

    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    common = {}

    for t in pred_tokens:
        if t in ref_tokens:
            common[t] = min(
                pred_tokens.count(t),
                ref_tokens.count(t)
            )

    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)

    return (
        2 * precision * recall
        / (precision + recall)
    )


# ------------------------------------
# Simple ROUGE-L (approx)
# Longest Common Subsequence
# ------------------------------------

def lcs(a, b):

    a = tokenize(a)
    b = tokenize(b)

    dp = [
        [0]*(len(b)+1)
        for _ in range(len(a)+1)
    ]

    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):

            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1

            else:
                dp[i][j] = max(
                    dp[i-1][j],
                    dp[i][j-1]
                )

    return dp[-1][-1]


def rouge_l(prediction, reference):

    lcs_len = lcs(
        prediction,
        reference
    )

    ref_len = len(tokenize(reference))
    pred_len = len(tokenize(prediction))

    if ref_len == 0 or pred_len == 0:
        return 0

    recall = lcs_len / ref_len
    precision = lcs_len / pred_len

    if precision + recall == 0:
        return 0

    return (
        2 * precision * recall
        / (precision + recall)
    )


# ------------------------------------
# Example evaluation set
# Replace with your real QA set
# ------------------------------------

dataset = [
    {
        "question": "What is overfitting?",
        "reference":
        "Overfitting occurs when a model memorizes training data and generalizes poorly."
    },
    {
        "question": "What is gradient descent?",
        "reference":
        "Gradient descent is an optimization algorithm used to minimize loss."
    }
]


# ------------------------------------
# Replace with YOUR LLM pipeline
# ------------------------------------

def generate_answer(question):

    # Replace with:
    # return rag.run(question, retriever=...)
    
    if "overfitting" in question.lower():
        return (
            "Overfitting happens when a model "
            "fits training data too closely and "
            "fails to generalize."
        )

    if "gradient" in question.lower():
        return (
            "Gradient descent is used to minimize "
            "a loss function during training."
        )

    return ""


# ------------------------------------
# Evaluation
# ------------------------------------

def main():

    em_scores = []
    f1_scores = []
    rouge_scores = []

    for item in dataset:

        q = item["question"]
        ref = item["reference"]

        pred = generate_answer(q)

        em = exact_match(pred, ref)
        f1 = f1_score(pred, ref)
        rg = rouge_l(pred, ref)

        em_scores.append(em)
        f1_scores.append(f1)
        rouge_scores.append(rg)

    avg_em = (
        sum(em_scores)
        / len(em_scores)
    )

    avg_f1 = (
        sum(f1_scores)
        / len(f1_scores)
    )

    avg_rouge = (
        sum(rouge_scores)
        / len(rouge_scores)
    )

    print("\n====== Model Evaluation ======")
    print(f"Avg Exact Match: {avg_em:.4f}")
    print(f"Avg F1 Score:    {avg_f1:.4f}")
    print(f"Avg ROUGE-L:     {avg_rouge:.4f}")
    print("==============================")


if __name__ == "__main__":
    main()