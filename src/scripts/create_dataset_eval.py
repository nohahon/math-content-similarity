# %%
import pandas as pd
from tqdm import tqdm
import numpy as np

np.random.seed(42)

# %%
conts = pd.read_csv("../data/zbmath_doc_contents.csv")
header = ["seed"] + ["rec" + str(i) for i in range(14)]
recs = pd.read_csv(
    "../data/recommendationPairs.csv",
    names=header,
    index_col=0,
)
full = pd.read_csv("../data/out.csv")

# %%
conts = conts.rename(columns={"Abstract/Review/Summarry": "text"})
conts = conts[~conts.text.isna()]
conts["text"] = conts.text.apply(lambda x: x[2:-2])
conts["text_len"] = conts.text.apply(lambda x: len(x))
conts = conts[conts.text_len > 150]
conts["zbMATH_ID"] = conts["zbMATH_ID"].astype(int)
golden_lookup = conts.set_index("zbMATH_ID")

# %%
lookup = full.rename(columns={"de": "id"})[["id", "text"]]
lookup["text_len"] = lookup.text.apply(lambda x: len(str(x)))
lookup = lookup[lookup.text_len > 150]
lookup["id"] = lookup["id"].astype(int)
lookup = lookup.set_index("id")

# %%
lookup


# %%
def create_pairs(df):
    relevant_docs = {}
    for seed, row in tqdm(df.iterrows()):
        rec_list = row[~row.isna()].astype(int).to_list()
        relevant_docs[seed] = set(rec_list)
    return relevant_docs


# %%
relevant_docs = create_pairs(recs)

# %%
relevant_docs


# %%
def get_contents_queries(df, lookup):
    queries = {}
    for seed, recs in df.items():
        if seed not in lookup.index:
            print(seed)
            continue
        query = lookup.loc[seed].text
        if isinstance(query, pd.Series):
            query = query.iloc[0]
        queries[seed] = query
    return queries


# %%
def get_contents_recs(df, lookup):
    queries = {}
    for _, recs in df.items():
        for rec_id in recs:
            if rec_id not in lookup.index:
                continue
            rec = lookup.loc[rec_id].text
            if isinstance(rec, pd.Series):
                rec = rec.iloc[0]
            queries[rec_id] = rec
    return queries


# %%
corpus_golden = get_contents_recs(relevant_docs, golden_lookup)

# %%
queries = get_contents_queries(relevant_docs, golden_lookup)


# %%
def get_contents_corpus(n, lookup):
    a = lookup.sample(n).reset_index()
    return dict(zip(a["id"], a["text"]))


# %%
corpus = get_contents_corpus(50000, lookup)

# %%
corpus.update(corpus_golden)

# %%
len(corpus.keys())

# %%
import pickle

with open("zbmath.pickle", "wb") as file:
    pickle.dump(corpus, file)

with open("relevant_docs.pickle", "wb") as file:
    pickle.dump(relevant_docs, file)

with open("queries.pickle", "wb") as file:
    pickle.dump(queries, file)
