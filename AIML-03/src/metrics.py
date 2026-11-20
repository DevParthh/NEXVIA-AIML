import math


def recall_at_k(ranked, relevant, k):
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def ndcg_at_k(ranked, relevant, k):
    relevant = set(relevant)
    dcg = sum(1.0 / math.log2(i + 2) for i, it in enumerate(ranked[:k]) if it in relevant)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0
