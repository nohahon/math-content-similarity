"""Module containing utility functions."""
import numpy as np


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

        # not sure about this. If you hit 1 in k=3 and you have 10 gold recs you have recall .33
        # but if you hit 1 in k=10 with gold recs 10 you have recal .1
        # recall = hits / (len(gold_recs) if len(gold_recs) <= k else k)
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
