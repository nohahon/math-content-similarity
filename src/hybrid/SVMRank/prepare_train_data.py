# %%
import pickle
import pandas as pd
import random
from sklearn.decomposition import PCA
import numpy as np
import argparse


def transform_embedding_mat(embedding_matrix, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(embedding_matrix)


def make_single_query_string(q, idx, rec, neg, embeddings):
    q_emb = embeddings[q]
    rec_emb = embeddings[rec]
    neg_emb = embeddings[neg]

    pos_emb = np.mean([q_emb, rec_emb], axis=0)
    neg_emb = np.mean([q_emb, neg_emb], axis=0)

    lines = []
    # positive
    lines.append(
        f"1 qid:{idx} "
        + " ".join([f"{j+1}:{pos_emb[j]}" for j in range(0, len(pos_emb))]),
    )
    # negative
    lines.append(
        f"2 qid:{idx} "
        + " ".join([f"{j+1}:{neg_emb[j]}" for j in range(0, len(neg_emb))]),
    )

    random.shuffle(lines)
    return "\n".join(lines)


def make_all_queries_string(pos_dict, neg_dict, embeddings):
    idx = 1
    queries = []
    for q in pos_dict.keys():
        if q not in neg_dict.keys():
            continue

        recs = pos_dict[q]
        negs = neg_dict[q][: len(recs)]

        for rec, neg in zip(recs, negs):
            queries.append(
                make_single_query_string(q, idx, rec, neg, embeddings),
            )
            idx += 1

    return "\n".join(queries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_dim", type=int, default=200)
    args = parser.parse_args()
    n_dim = args.n_dim

    # load data
    with open("./data/zbmath/train_posandnegPairs.pkl", "rb") as f:
        pos, neg = pickle.load(f)
    with open("./data/zbmath/seedToembed.pkl", "rb") as f:
        feature_map = pickle.load(f)

    # create an embedding map
    ids = set()
    for k, v in pos:
        ids.add(k)
        ids.add(v)
    for k, v in neg:
        ids.add(k)
        ids.add(v)

    embeddings = {id: feature_map[id] for id in ids}
    transformed = transform_embedding_mat(
        np.array(list(embeddings.values())),
        n_dim,
    )
    embeddings_transformed = {
        id: emb for id, emb in zip(list(embeddings.keys()), transformed)
    }

    pos_df = pd.DataFrame(pos)
    neg_df = pd.DataFrame(neg)

    # build dictionaries
    positive = {}
    for query in pos_df[0].unique():
        positive[query] = pos_df[pos_df[0] == query][1].tolist()

    negative = {}
    for query in neg_df[0].unique():
        negative[query] = neg_df[neg_df[0] == query][1].tolist()

    s = make_all_queries_string(positive, negative, embeddings_transformed)

    with open("./data/train/train.dat", "w") as f:
        f.write(s)
