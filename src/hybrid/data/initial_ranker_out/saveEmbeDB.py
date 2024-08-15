import sys
import csv
import torch
import pickle
import sqlite3
import numpy as np
from transformers import AutoTokenizer, AutoModel
csv.field_size_limit(100000000)

INSTRUCTIONS = {
    "qa": {
        "query": "Represent this query for retrieving relevant documents: ",
        "key": "Represent this document for retrieval: ",
    },
}

def insert_embeddingDB(keys, embeddings):
    # can we adapt such a that it saves multiple key and mebeddings
    data_to_insert = [(key, sqlite3.Binary(embedding.detach().cpu().numpy().tobytes())) for key, embedding in zip(keys, embeddings)]
    conn = sqlite3.connect('sqlLite_DB/rfrncsEmbeddings.db')
    cursor = conn.cursor()
    insert_query = 'INSERT OR REPLACE INTO embeddings (id, embedding) VALUES (?, ?)'
    cursor.executemany(insert_query, data_to_insert)
    conn.commit()
    conn.close()

def genEmbeddingsBatch():
    """ Saves embeddings of all initail ranked documents """
    kwrddocfeat = pickle.load(open("dataFeatures/initRank_featRefrnc.pkl", "rb"))
    print(len(kwrddocfeat))
    instruction = INSTRUCTIONS["qa"]
    tokenizer = AutoTokenizer.from_pretrained("BAAI/llm-embedder")
    model = AutoModel.from_pretrained("BAAI/llm-embedder")
    docIds = list(kwrddocfeat.keys())
    for i in range(0, len(docIds) - 1, 5000):
        docs_ = [instruction["query"] + kwrddocfeat[que] for que in docIds[i:i+5000]]
        query_inputs = tokenizer(docs_, padding=True, truncation=True, return_tensors="pt", )
        with torch.no_grad():
            query_outputs = model(**query_inputs)
            query_embeddings = query_outputs.last_hidden_state[:, 0]
            query_embeddings = torch.nn.functional.normalize( query_embeddings, p=2, dim=1, )
        print(docIds[i:i+5000])
        insert_embeddingDB(docIds[i:i+5000], query_embeddings)

def retrievEmbed(doc_id):
    conn = sqlite3.connect('sqlLite_DB/featDBs/abstrEmbeddings.db')
    cursor = conn.cursor()
    cursor.execute('SELECT embedding FROM embeddings WHERE id = ?', (doc_id,))
    data = cursor.fetchone()
    # Convert bytes back to numpy array
    if data:
        embedding = np.frombuffer(data[0], dtype=np.float64)  # Adjust dtype according to how you store it
        return embedding
    return None

#genEmbeddingsBatch()
#print(retrievEmbed("1138672").size)