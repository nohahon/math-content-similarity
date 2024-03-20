# %%
import pickle
import pandas as pd
from tqdm import tqdm


def make_single_test_query_string(q, idx):
    q_emb = embeddings[q]

    lines = []
    lines.append(
        f"1 qid:{idx} "
        + " ".join([f"{i}:{q_emb[i]}" for i in range(1, len(q_emb))]),
    )

    all_ = ids

    for id in all_:
        emb_ = embeddings[id]
        lines.append(
            f"2 qid:{idx} "
            + " ".join([f"{i}:{emb_[i]}" for i in range(1, len(emb_))]),
        )

    return "\n".join(lines)


def make_all_queries_string():
    idx = 0
    queries = []
    for pos_query in tqdm(rec_dict.keys()):
        queries.append(make_single_test_query_string(pos_query, idx))
        idx += 1

    return "\n".join(queries)


if __name__ == "__main__":
    # load data
    with open("./data/zbmath/test_posandnegPairs.pkl", "rb") as f:
        pos, neg = pickle.load(f)
    with open("./data/zbmath/seedToembed.pkl", "rb") as f:
        feature_map = pickle.load(f)

    # create an embedding map
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

    pos_df = pd.DataFrame(pos)
    rec_dict = {}
    for query in pos_df[0].unique():
        rec_dict[query] = pos_df[pos_df[0] == query][1].tolist()

    with open("./data/rec_dict.pickle", "wb") as f:
        pickle.dump(rec_dict, f)
    with open("./data/ids.pickle", "wb") as f:
        pickle.dump(ids, f)

    s = make_all_queries_string()
    with open("./data/test.dat", "w") as f:
        f.write(s)
