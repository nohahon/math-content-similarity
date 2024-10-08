import sys
import pickle
import faiss
import json
import numpy as np
import pandas as pd
sys.path.append('../../zbCitReviwRecmn/')
import datasplits
from sentence_transformers import SentenceTransformer

INSTRUCTIONS = {
    "qa": {
        "query": "Instruct: Retrieve semantically similar text.\nQuery: ",
        "key": "",
    },
}

def createIndex(main_data_):
    """ Creating FAISS vector DB """
    instruction = INSTRUCTIONS["qa"]
    model = SentenceTransformer("dunzhang/stella_en_400M_v5", device='cuda', trust_remote_code=True)
    embedding_dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(embedding_dim)
    batch_size = 5000
    for start_idx in range(0, len(main_data_), batch_size):
        end_idx = min(start_idx + batch_size, len(main_data_))
        titles_batch = main_data_['title'].iloc[start_idx:end_idx].tolist()
        embeddings_batch = model.encode(titles_batch, convert_to_numpy=True, device='cuda')
        faiss.normalize_L2(embeddings_batch)
        index.add(embeddings_batch.astype('float32'))
    faiss.write_index(index, "data/base_/title/tit_stella_basemebd.index")

def main_faiss():
    instruction = INSTRUCTIONS["qa"]
    model = SentenceTransformer("dunzhang/stella_en_400M_v5", device='cuda', trust_remote_code=True)
    data_ = "/beegfs/schubotz/ankit/data/zbReviewCitData/citation_dataset.csv"
    dataFr = datasplits.load_csv_to_dataframe(data_)  # Load main dataset
    train_, test_, valid_ = datasplits.split_dataframe(dataFr)# Split data into train, test, valid
    main_data = datasplits.getMainData()
    missing_document_ids = test_[~test_['document_id'].isin(main_data['document_id'])]['document_id']
    missing_data = pd.DataFrame({'document_id': missing_document_ids, 'title': 'No title'})
    main_data = pd.concat([main_data, missing_data], ignore_index=True)
    #createIndex(main_data)  #only need to run once for creating index 
    #sys.exit(0)
    # Loop through each document_id in the test_ dataframe
    index = faiss.read_index("data/base_/title/tit_stella_basemebd.index")
    batch_size = 5000
    for batch_start in range(0, len(test_['document_id']), batch_size):
        results = {}
        batch_doc_ids = test_['document_id'][batch_start:batch_start + batch_size]
        #batch_titles = main_data[main_data['document_id'].isin(batch_doc_ids)]['title'].values
        batch_titles = main_data.set_index('document_id').loc[batch_doc_ids]['title'].values
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
        with open(f'data/base_/title/scores_/tit_base_{batch_start}.json', 'w') as json_file:
            json.dump(results, json_file)
        #sys.exit(0)

if __name__ == "__main__":
    #main_tfidf()
    main_faiss()