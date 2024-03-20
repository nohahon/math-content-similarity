import sys
import math
import numpy as np
import jsonlines

def P_R_F1_at_k(k, resultsFile):
    """
    Calculate precision, recall, and F1-score at k for a given results file.

    Parameters:
    k (int): The value of k for calculating precision and recall.
    resultsFile (str): The path to the results file.

    Returns:
    tuple: A tuple containing the mean precision, mean recall, and mean F1-score.
    """

    with jsonlines.open(resultsFile) as reader:
        recalls = []
        precisions = []
        f1s = []
        for line in reader:
            gold_recs = set(line["idealRcmnds"])
            if len(gold_recs) == 0:
                continue
            predicted = line["baselineRcmnds"]
            predicted = set(
                [str(rec[0]) for _, rec in predicted.items() if int(_) > 0][:k],
            )  # {0:[rec_id,similarity_score],1:[rec_id,similarity_score],..}

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

        return np.mean(precisions), np.mean(recalls), np.mean(f1s)

resultsFile = "/beegfs/schubotz/ankit/code/evaluation/hybridApproach/scores/rslts_comBined.jsonl"
print(P_R_F1_at_k(3, resultsFile))
print(P_R_F1_at_k(5, resultsFile))
