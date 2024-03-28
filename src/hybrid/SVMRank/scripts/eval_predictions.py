import numpy as np
import pandas as pd
import pickle as pkl
import argparse
from tqdm import tqdm

COS_SIM = lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def P_R_F1_at_k_random(k, rec_dict, ids):
    recalls = []
    precisions = []
    f1s = []

    for q_id, recs in rec_dict.items():
        gold_recs = set(recs)
        if len(gold_recs) == 0:
            continue

        df = pd.DataFrame({"id": ids})

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


def P_R_F1_at_k(k, rec_dict, ids, preds):
    recalls = []
    precisions = []
    f1s = []
    idx = 0
    for q_id, recs in rec_dict.items():
        gold_recs = set(recs)
        if len(gold_recs) == 0:
            continue

        scores = preds[idx : idx + len(ids)]
        idx += len(ids)

        df = pd.DataFrame({"id": ids, "val": scores})
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


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--rand", type=bool, required=False, default=False)
    parser.add_argument("--split", type=str, required=False, default="dev")
    parser.add_argument(
        "--eval_baseline",
        type=bool,
        required=False,
        default=True,
    )

    k_ = 100

    args = parser.parse_args()

    with open("svm_predictions", "r") as f:
        preds = f.readlines()
    with open(f"./data/{args.split}/ids.pickle", "rb") as f:
        ids = pkl.load(f)
    with open(f"./data/{args.split}/rec_dict.pickle", "rb") as f:
        rec_dict = pkl.load(f)
    preds = [float(el.strip("\n")) for el in preds]

    if args.rand:
        random_p = 0
        random_r = 0
        random_f1 = 0
        for i in tqdm(range(10000)):
            p, r, f1 = P_R_F1_at_k_random(k_, rec_dict, ids)
            random_p += p
            random_r += r
            random_f1 += f1
        print(random_f1 / 10000)

    else:
        p, r, f1 = P_R_F1_at_k(k_, rec_dict, ids, preds)
        if args.eval_baseline:
            with open(f"./data/zbmath/seedToembed.pkl", "rb") as f:
                embeddings = pkl.load(f)

            scores = []
            for k in list(rec_dict.keys()):
                for id in ids:
                    scores.append(COS_SIM(embeddings[k], embeddings[id]))
            p_b, r_b, f1_b = P_R_F1_at_k(k_, rec_dict, ids, scores)

        print(f"baseline: {f1_b}")
        print(f"SVMrank: {f1}")
