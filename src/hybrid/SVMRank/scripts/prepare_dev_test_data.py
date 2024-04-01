# %%
import pickle
import pandas as pd
from tqdm import tqdm
from src.myutils import transform_embedding_mat
import numpy as np
import argparse


def make_single_test_query_string(q, q_id, ids, embeddings):
    """
    Creates a single test query string for ranking.

    Args:
        q (str): The query.
        q_id (int): The query ID.
        ids (list): The list of unique IDs.
        embeddings (dict): The dictionary of embeddings.

    Returns:
        str: The test query string.
    """
    q_emb = embeddings[q]

    lines = []

    all_ = ids

    for rank, id in enumerate(all_):
        emb = embeddings[id]
        emb = np.mean([emb, q_emb], axis=0)
        lines.append(
            f"{rank+1} qid:{q_id} "
            + " ".join([f"{i+1}:{emb[i]}" for i in range(0, len(emb))])
            + f" #{id}",
        )

    return "\n".join(lines)


def make_all_queries_string(rec_dict, ids, embeddings):
    """
    Creates a string containing all the test queries for ranking.

    Args:
        rec_dict (dict): The dictionary of positive queries and their corresponding recommendations.
        ids (list): The list of unique IDs.
        embeddings (dict): The dictionary of embeddings.

    Returns:
        str: The string containing all the test queries.
    """
    idx = 153
    queries = []
    for pos_query in tqdm(rec_dict.keys()):
        queries.append(
            make_single_test_query_string(pos_query, idx, ids, embeddings),
        )
        idx += 1

    return "\n".join(queries)


def create_data(pos, neg, feature_map, split: str, n_dim: int):
    """
    Creates the data for training or testing the SVMRank model.

    Args:
        pos (list): The list of positive query-recommendation pairs.
        neg (list): The list of negative query-recommendation pairs.
        feature_map (dict): The dictionary mapping IDs to embeddings.
        split (str): The split name (e.g., "train", "test", "dev").
        n_dim (int): The number of dimensions for the transformed embeddings.
    """
    # create unique id list
    ids = []
    for k, v in pos:
        if k not in ids:
            ids.append(k)
        if v not in ids:
            ids.append(v)
    for k, v in neg:
        if k not in ids:
            ids.append(k)
        if v not in ids:
            ids.append(v)
    ids = list(ids)
    ids = pd.DataFrame(ids).sample(len(ids), random_state=42)[0].tolist()
    embeddings = {id: feature_map[id] for id in ids}
    transformed = transform_embedding_mat(
        np.array(list(embeddings.values())),
        n_dim,
    )
    embeddings_transformed = {
        id: emb for id, emb in zip(list(embeddings.keys()), transformed)
    }

    pos_df = pd.DataFrame(pos)
    rec_dict = {}
    for query in pos_df[0].unique():
        rec_dict[query] = pos_df[pos_df[0] == query][1].tolist()

    with open(f"./data/{split}/rec_dict.pickle", "wb") as f:
        pickle.dump(rec_dict, f)
    with open(f"./data/{split}/ids.pickle", "wb") as f:
        pickle.dump(ids, f)

    s = make_all_queries_string(
        rec_dict=rec_dict,
        ids=ids,
        embeddings=embeddings_transformed,
    )
    with open(f"./data/{split}/{split}.dat", "w") as f:
        f.write(s)


if __name__ == "__main__":
    # load data

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_dim", type=int, default=100)
    args = parser.parse_args()

    with open("./data/zbmath/test_posandnegPairs.pkl", "rb") as f:
        pos_t, neg_t = pickle.load(f)
    with open("./data/zbmath/valid_posandnegPairs.pkl", "rb") as f:
        pos_v, neg_v = pickle.load(f)
    with open("./data/zbmath/seedToembed.pkl", "rb") as f:
        feature_map = pickle.load(f)

    create_data(pos_t, neg_t, feature_map, split="test", n_dim=args.n_dim)
    create_data(pos_v, neg_v, feature_map, split="dev", n_dim=args.n_dim)
