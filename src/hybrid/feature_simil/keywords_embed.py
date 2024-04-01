import csv
import torch
import pickle
from transformers import AutoTokenizer, AutoModel

csv.field_size_limit(100000000)

INSTRUCTIONS = {
    "qa": {
        "query": "Represent this query for retrieving relevant documents: ",
        "key": "Represent this document for retrieval: ",
    },
}


def getAlltitles(filename):
    """retrurns dict with key as zbMATH ID and value as abstract/review/summarry"""
    dataWhole = dict()
    with open(filename, "r", encoding="utf-8", errors="ignore") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for eachro in csvreader:
            dataWhole[eachro[1]] = eachro[0]
    return dataWhole


def genEmbeddingsBatch():
    alltitles = getAlltitles("/beegfs/schubotz/ankit/data/zbMATH_keywords.csv")
    instruction = INSTRUCTIONS["qa"]
    tokenizer = AutoTokenizer.from_pretrained("BAAI/llm-embedder")
    model = AutoModel.from_pretrained("BAAI/llm-embedder")
    queries = [
        instruction["query"] + alltitles[query] for query in getSEEDIds()
    ]
    query_inputs = tokenizer(
        queries,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    for i in range(0, len(alltitles) - 1, 5000):
        print("Doing for batch: ", i)
        keys = [
            instruction["key"] + alltitles[key]
            for key in list(alltitles.keys())[i : i + 5000]
        ]
        key_inputs = tokenizer(
            keys,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            query_outputs = model(**query_inputs)
            key_outputs = model(**key_inputs)
            query_embeddings = query_outputs.last_hidden_state[:, 0]
            key_embeddings = key_outputs.last_hidden_state[:, 0]
            query_embeddings = torch.nn.functional.normalize(
                query_embeddings,
                p=2,
                dim=1,
            )
            key_embeddings = torch.nn.functional.normalize(
                key_embeddings,
                p=2,
                dim=1,
            )
        similarity = query_embeddings @ key_embeddings.T

        with open("data_ne/keywords/key_" + str(i) + "_.pkl", "wb") as f:
            pickle.dump(similarity, f)


genEmbeddingsBatch()
