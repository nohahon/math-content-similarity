import pickle
import pandas as pd
import random

# load data
with open("zbmath/test_posandnegPairs.pkl", "rb") as f:
    pos, neg = pickle.load(f)
with open("zbmath/seedToembed.pkl", "rb") as f:
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
pos_df = pd.DataFrame(pos)
neg_df = pd.DataFrame(neg)

# build dictionaries
positive = {}
for query in pos_df[0].unique():
    positive[query] = pos_df[pos_df[0] == query][1].tolist()

negative = {}
for query in neg_df[0].unique():
    negative[query] = neg_df[neg_df[0] == query][1].tolist()


def make_single_query_string(q, doc, idx):
    q_emb = embeddings[q]
    doc_emb = embeddings[doc]

    lines = []
    lines.append(
        f"1 qid:{idx} "
        + " ".join([f"{i}:{q_emb[i]}" for i in range(1, len(q_emb))]),
    )
    lines.append(
        f"2 qid:{idx} "
        + " ".join([f"{i}:{doc_emb[i]}" for i in range(1, len(doc_emb))]),
    )

    negatives = negative[query]

    for neg_example in negatives:
        neg_emb = embeddings[neg_example]
        lines.append(
            f"3 qid:{idx} "
            + " ".join([f"{i}:{neg_emb[i]}" for i in range(1, len(neg_emb))]),
        )

    random.shuffle(lines)
    return "\n".join(lines)


def make_all_queries_string():
    idx = 1
    queries = []
    for pos_query in positive.keys():
        if pos_query not in negative.keys():
            continue

        for doc_id in positive[pos_query]:
            queries.append(make_single_query_string(pos_query, doc_id, idx))
            idx += 1

    return "\n".join(queries)


# s = make_all_queries_string()
# with open("test.dat", "w") as f:
#     f.write(s)
