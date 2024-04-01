"""Module containing utility functions."""
import numpy as np
from sklearn.decomposition import PCA


def transform_embedding_mat(embedding_matrix, n_components):
    """
    Applies PCA to reduce the dimensionality of the embedding matrix.

    Args:
        embedding_matrix (numpy.ndarray): The input embedding matrix.
        n_components (int): The number of components to keep after dimensionality reduction.

    Returns:
        numpy.ndarray: The transformed embedding matrix.
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(embedding_matrix)


def get_dataset_name(path_to_dataset: str):
    """
    Returns the name of the dataset given its path.

    Args:
        path_to_dataset (str): The path to the dataset.

    Returns:
        str: The name of the dataset.
    """
    return path_to_dataset.split("/")[-1].split(".")[0]


def P_R_F1_at_k(preds, k):
    """
    Calculates the precision, recall, and F1 score at k for a given set of predictions.

    Args:
        preds (list): A list of tuples containing the ground truth and predicted values.
        k (int): The value of k.

    Returns:
        dict: A dictionary containing the precision, recall, and F1 score at k.
    """
    recalls = []
    precisions = []
    f1s = []
    for truth, predictions in preds:
        truth = [str(t) for t in truth]
        gold_recs = set(truth)
        if len(gold_recs) == 0:
            continue
        predicted = set(predictions[:k])

        hits = len(predicted.intersection(gold_recs))

        recall = hits / len(gold_recs)
        precision = hits / k
        f1 = (
            (2 * (precision * recall) / (precision + recall))
            if (precision + recall) > 0
            else 0
        )

        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)

    return {"p": np.mean(precisions), "r": np.mean(recalls), "f": np.mean(f1s)}
