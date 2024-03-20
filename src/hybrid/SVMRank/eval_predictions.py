import numpy as np
import pandas as pd
import pickle as pkl


def P_R_F1_at_k_random(k):
    recalls = []
    precisions = []
    f1s = []

    for q_id, recs in rec_dict.items():
        gold_recs = set(recs)
        if len(gold_recs) == 0:
            continue

        ids_lst = [q_id] + ids
        df = pd.DataFrame({"id": ids_lst})

        predicted = set(df["id"].sample(k).tolist())
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

    return np.mean(precisions), np.mean(recalls), np.mean(f1s)


def P_R_F1_at_k(k):
    recalls = []
    precisions = []
    f1s = []
    idx = 0
    ids_len = len(ids) + 1
    for q_id, recs in rec_dict.items():
        gold_recs = set(recs)
        if len(gold_recs) == 0:
            continue

        scores = preds[idx : idx + ids_len]
        idx += ids_len

        ids_lst = [q_id] + ids
        df = pd.DataFrame({"id": ids_lst, "val": scores})
        df = df.sort_values(by="val")

        predicted = set(df[:k]["id"].tolist())
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

    return np.mean(precisions), np.mean(recalls), np.mean(f1s)


with open("svm_predictions", "r") as f:
    preds = f.readlines()
with open("./data/ids.pickle", "rb") as f:
    ids = pkl.load(f)
with open("./data/rec_dict.pickle", "rb") as f:
    rec_dict = pkl.load(f)
preds = [float(el.strip("\n")) for el in preds]


random_p = 0
random_r = 0
random_f1 = 0
for i in range(1000):
    p, r, f1 = P_R_F1_at_k_random(10)
    random_p += p
    random_r += r
    random_f1 += f1

p, r, f1 = P_R_F1_at_k(10)

print(f"Random F1 at 10: {random_f1/1000}")
print(f"Trained F1 at 10: {f1}")
