import os
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize
import sys
import pickle
import faiss
import json
import numpy as np
import pandas as pd
sys.path.append('../../tf_idf_algrthmn/')
import zbCitData_st
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INSTRUCTIONS = {
    "qa": {
        "query": "Instruct: Retrieve semantically similar text.\nQuery: ",
        "key": "",
    },
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load model
model_dir = "../contra_stella/trained_models/trained_model_5e_6/"
vector_dim = 1024
vector_linear_directory = f"2_Dense_{vector_dim}"
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
vector_linear = torch.nn.Linear(in_features=model.config.hidden_size, out_features=vector_dim)
vector_linear_dict = {
    k.replace("linear.", ""): v for k, v in
    torch.load(os.path.join("/beegfs/schubotz/.cache/huggingface/hub/models--dunzhang--stella_en_400M_v5/snapshots/24e2e1ffe95e95d807989938a5f3b8c18ee651f5", f"{vector_linear_directory}/pytorch_model.bin")).items()
}
vector_linear.load_state_dict(vector_linear_dict)
vector_linear.to(device)
print("Model loaded!")

def createIndex(main_data_, model_):
    instruction = INSTRUCTIONS["qa"]
    embedding_dim = model_.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(embedding_dim)
    batch_size = 5000 
    for start_idx in range(0, len(main_data_), batch_size):
        end_idx = min(start_idx + batch_size, len(main_data_))
        titles_batch = main_data_['text'].iloc[start_idx:end_idx].tolist()
        embeddings_batch = model_.encode(titles_batch, convert_to_numpy=True, device='cuda')
        faiss.normalize_L2(embeddings_batch)
        index.add(embeddings_batch.astype('float32'))
    faiss.write_index(index, "data/trained_models/abstrs/rand_negs_10/ab_stella_randneg_10.index")

def candRecTest(seed_id, test_df ,n_rec):
    # get shuffllled cand rec for a seed ID seed_id, n_rec
    print(test_df.columns, test_df['document_id'].dtype, type(seed_id))
    print("This is what inside DF looks like: ", type(test_df['document_id'].tolist()[0]))
    idelRec = test_df.loc[test_df['document_id'] == str(seed_id), 'citation_de'].iloc[0]
    idelRec = idelRec.split(';')
    getRecCand = test_['document_id'].sample(n=10, random_state=42)
    getRecCand = [cand for cand in getRecCand if cand not in idelRec]
    getRecCand.append(seed_id)
    #print(getRecCand)
    return getRecCand

def main_faiss():
    instruction = INSTRUCTIONS["qa"]
    data_ = "/beegfs/schubotz/ankit/data/zbReviewCitData/citation_dataset.csv"
    dataFr = zbCitData_st.load_csv_to_dataframe(data_)  # Load main dataset
    main_df = zbCitData_st.getMainData(usenans=True)[['document_id','title','text']].dropna()
    train_, test_, valid_ = zbCitData_st.split_dataframe(dataFr)
    print(train_.shape[0], test_.shape[0], valid_.shape[0])
    main_data = zbCitData_st.getMainData()
    missing_document_ids = test_[~test_['document_id'].isin(main_data['document_id'])]['document_id']
    missing_data = pd.DataFrame({'document_id': missing_document_ids, 'text': 'No Abstract'})
    main_data = pd.concat([main_data, missing_data], ignore_index=True)
    #createIndex(main_data, model)
    #sys.exit(0)
    # Loop through each document_id in the test_ dataframe
    #index = faiss.read_index("data/trained_models/abstrs/rand_negs_10/ab_stella_randneg_10.index")
    batch_size = 50
    getseedCands = test_['document_id'].sample(n=10, random_state=42)
    for eachTestSD in getseedCands:
        getCandRec = candRecTest(getseedCands, test_,50)
        print("getCandRec: ", getCandRec)
        seed_text = main_data.set_index('document_id').loc[[eachTestSD]]['text'].values
        batch_titles = main_data.set_index('document_id').loc[getCandRec]['text'].values
        print(batch_titles)
        # Embed the queries
        with torch.no_grad():
            input_data = tokenizer(seed_text, padding="longest", truncation=True, max_length=512, return_tensors="pt")
            input_data = {k: v.cuda() for k, v in input_data.items()}
            attention_mask = input_data["attention_mask"]
            last_hidden_state = model(**input_data)[0]
            last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            query_vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            query_vectors = normalize(vector_linear(query_vectors).cpu().numpy())
        # Embed the documents
        with torch.no_grad():
            input_data = tokenizer(batch_titles, padding="longest", truncation=True, max_length=512, return_tensors="pt")
            input_data = {k: v.cuda() for k, v in input_data.items()}
            attention_mask = input_data["attention_mask"]
            last_hidden_state = model(**input_data)[0]
            last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            docs_vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            docs_vectors = normalize(vector_linear(docs_vectors).cpu().numpy())
        print(query_vectors.shape, docs_vectors.shape)
        similarities = query_vectors @ docs_vectors.T
        print(similarities)
        sys.exit(0)


    for batch_start in range(0, len(test_['document_id']), batch_size):
        results = {}
        batch_doc_ids = test_['document_id'][batch_start:batch_start + batch_size]
        #batch_titles = main_data[main_data['document_id'].isin(batch_doc_ids)]['title'].values
        batch_titles = main_data.set_index('document_id').loc[batch_doc_ids]['text'].values
        embeddings_batch = model.encode(batch_titles, convert_to_numpy=True, device='cuda')
        faiss.normalize_L2(embeddings_batch)
        _, ranked_indices = index.search(embeddings_batch, 1000)
        for i, docu_id in enumerate(batch_doc_ids):
            try:
                ranked_doc_ids = [main_data['document_id'].iloc[idx_] for idx_ in ranked_indices[i]]
                results[docu_id] = ranked_doc_ids
            except:
                results[docu_id] = []
        #print(results)
        results = {int(doc_id): [int(idx) for idx in ranked_doc_ids] for doc_id, ranked_doc_ids in results.items()}
        #print(results)
        with open(f'data/trained_models/abstrs/rand_negs_10/scores_/abs_rndneg_10_{batch_start}.json', 'w') as json_file:
            json.dump(results, json_file)
        #sys.exit(0)

if __name__ == "__main__":
    #main_tfidf()
    main_faiss()

