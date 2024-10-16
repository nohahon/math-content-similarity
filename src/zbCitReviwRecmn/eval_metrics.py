import numpy as np

def mean_reciprocal_rank(ideal_recommendations, generated_recommendations):
    """ Mean Reciprocal Rank (MRR) measures how soon the first relevant result is returned."""
    mrr = 0
    for ideal, generated in zip(ideal_recommendations, generated_recommendations):
        for i, rec in enumerate(generated):
            if rec in ideal:
                mrr += 1 / (i + 1)
                break
    return mrr / len(ideal_recommendations)

def dcg_at_k(recommendations, ideal, k):
    """Calculate the Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i in range(min(k, len(recommendations))):
        if recommendations[i] in ideal:
            dcg += 1 / np.log2(i + 2)  # using i+2 to start at log2(2) = 1
    return dcg

def ndcg_at_k(ideal_recommendations, generated_recommendations, k):
    """
    Normalized Discounted Cumulative Gain (nDCG) measures the quality of the recommendations
    by comparing the recommended order with the ideal order.
    """
    ndcg_total = 0
    for ideal, generated in zip(ideal_recommendations, generated_recommendations):
        ideal_dcg = dcg_at_k(ideal, ideal, k)  # DCG for ideal list
        actual_dcg = dcg_at_k(generated, ideal, k)  # DCG for the generated list
        ndcg = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0
        ndcg_total += ndcg
    return ndcg_total / len(ideal_recommendations)


def precision_at_k(ideal_recommendations, generated_recommendations, k):
    """P@k measures how many of the top k recommendations are relevant."""
    precision_total = 0
    for ideal, generated in zip(ideal_recommendations, generated_recommendations):
        relevant = len([rec for rec in generated[:k] if rec in ideal])
        precision_total += relevant / k
    return precision_total / len(ideal_recommendations)

def recall_at_k(ideal_recommendations, generated_recommendations):
    """R@k measures how many of the relevant items are found in the top k recommendations."""
    recall_total = 0
    for ideal, generated in zip(ideal_recommendations, generated_recommendations):
        # remove the following line and add k (constant) to function header 
        k = len(ideal)
        relevant = len([rec for rec in generated[:k] if rec in ideal])
        recall_total += relevant / len(ideal)
    return recall_total / len(ideal_recommendations)

# Sample input
#ideal_recommendations = [['rec1', 'rec2'], ['rec3', 'rec4']]
#generated_recommendations = [['rec5', 'rec6'], ['rec7', 'rec8']]
def main(ideal_recommendations, generated_recommendations):
    # Calculate MRR, nDCG@5, P@3, P@5, and Recall
    mrr = mean_reciprocal_rank(ideal_recommendations, generated_recommendations)
    ndcg_5 = ndcg_at_k(ideal_recommendations, generated_recommendations, 5)
    precision_3 = precision_at_k(ideal_recommendations, generated_recommendations, 3)
    precision_5 = precision_at_k(ideal_recommendations, generated_recommendations, 5)
    recall_ = recall_at_k(ideal_recommendations, generated_recommendations)
    return precision_3, precision_5, recall_, mrr, ndcg_5
