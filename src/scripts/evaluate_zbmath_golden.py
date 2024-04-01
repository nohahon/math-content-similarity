# %%
import pandas as pd
import pickle as pkl
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm


def load_pkl(path):
    with open(path, "rb") as f:
        return pkl.load(f)


# %%
# setup database
chroma_client = chromadb.PersistentClient(path="data/vecdb")
# creating embedding function
sentence_transformer_ef = (
    embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="mixedbread-ai/mxbai-embed-large-v1",
    )
)
# %%
table = chroma_client.get_collection(
    "golden_vecs",
    embedding_function=sentence_transformer_ef,
)

# %%
recs = load_pkl("data/zbmath_golden_recs.pickle")
lookup = pd.read_csv("data/zbmath_golden_lookup.csv")
lookup = lookup.set_index("id")

predictions = []
for seed, recs in tqdm(recs.items()):
    seed_text = lookup.loc[seed].text
    preds = table.query(
        query_texts=seed_text,
        n_results=11,
    )["ids"]
    predictions.append((recs, preds[0][1:]))


from src.myutils import P_R_F1_at_k

print(P_R_F1_at_k(predictions, 10))


with open("data/predictions/preds.pickle", "wb") as f:
    pkl.dump(predictions, f)
