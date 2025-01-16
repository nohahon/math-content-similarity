import os
import sys
import pickle
import faiss
import json
import numpy as np
import pandas as pd
sys.path.append('../tf_idf_algrthmn/')
from FlagEmbedding import FlagICLModel
import zbCitData_st

def getGPUproperties():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_id = torch.cuda.current_device()
    print("GPU ID: ", gpu_id)
    total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
    reserved_memory = torch.cuda.memory_reserved(gpu_id)
    allocated_memory = torch.cuda.memory_allocated(gpu_id)
    free_memory = reserved_memory - allocated_memory
    print(f"Total memory: {total_memory / (1024 ** 3):.2f} GB")
    print(f"Reserved memory: {reserved_memory / (1024 ** 3):.2f} GB")
    print(f"Allocated memory: {allocated_memory / (1024 ** 3):.2f} GB")
    print(f"Free memory: {free_memory / (1024 ** 3):.2f} GB")

queries = ["how much protein should a female eat"]
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day."]
examples = [
  {'instruct': 'Given a title of research paper from pure and applied mathematics find out titles that are similar.',
   'query': 'Projective varieties multiply covered by rational normal curves.',
   'response': "On varieties X\subset \mathbb P^N such that a curve of X of given degree passes through n points of X."},
  {'instruct': 'Given a title of research paper from pure and applied mathematics find out titles that are similar.',
   'query': 'On the Hilbert scheme of curves of degree d and genus frac{(d-3)(d-4)}{2}',
   'response': "The Hilbert schemes of locally Cohen-Macaulay curves in P3 may after all be connected."}
]
model = FlagICLModel('BAAI/bge-en-icl', 
                     query_instruction_for_retrieval="Given a title of pure and applied mathematics research paper, retrieve relevant titles that are similar.",
                     examples_for_task=examples,  # set `examples_for_task=None` to use model without examples
                     use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

def createIndex(main_data_):
    #getGPUproperties()
    query_embeddings = model.encode_queries(queries)
    embedding_dim = query_embeddings.shape[1]
    print("Dimension of embedding: ", embedding_dim)
    # Initialize the FAISS index with the embedding dimension
    index = faiss.IndexFlatIP(embedding_dim)
    batch_size_ = 100
    # Loop through the dataset and encode the titles in batches
    for start_idx in range(0, len(main_data_), batch_size_):
        end_idx = min(start_idx + batch_size_, len(main_data_))
        titles_batch = main_data_['title'].iloc[start_idx:end_idx].tolist()
        query_embeddings = model.encode_corpus(titles_batch)
        #query_embeddings = query_embeddings.cpu().numpy()
        faiss.normalize_L2(query_embeddings)
        index.add(query_embeddings.astype('float32'))
    # Save the index to a file
    faiss.write_index(index, "data/base_/title/tit_nvmembv2_basemebd.index")

def main_faiss():
    data_ = "/beegfs/schubotz/ankit/data/zbReviewCitData/citation_dataset.csv"
    dataFr = zbCitData_st.load_csv_to_dataframe(data_)  # Load main dataset
    train_, test_, valid_ = zbCitData_st.split_dataframe(dataFr)# Split data into train, test, valid
    main_data = zbCitData_st.getMainData()
    missing_document_ids = test_[~test_['document_id'].isin(main_data['document_id'])]['document_id']
    missing_data = pd.DataFrame({'document_id': missing_document_ids, 'title': 'No title'})
    main_data = pd.concat([main_data, missing_data], ignore_index=True)
    #createIndex(main_data)
    #sys.exit(0)
    # Loop through each document_id in the test_ dataframe
    index = faiss.read_index("data/base_/title/tit_nvmembv2_basemebd.index")
    batch_size = 100
    for batch_start in range(0, len(test_['document_id']), batch_size):
        results = {}
        batch_doc_ids = test_['document_id'][batch_start:batch_start + batch_size]
        #batch_titles = main_data[main_data['document_id'].isin(batch_doc_ids)]['title'].values
        batch_titles = main_data.set_index('document_id').loc[batch_doc_ids]['title'].values
        embeddings_batch = model.encode_queries(batch_titles)
        faiss.normalize_L2(embeddings_batch)
        _, ranked_indices = index.search(embeddings_batch, 1000)
        for i, docu_id in enumerate(batch_doc_ids):
            ranked_doc_ids = [main_data['document_id'].iloc[idx_] for idx_ in ranked_indices[i]]
            results[docu_id] = ranked_doc_ids
        #print(results)
        results = {int(doc_id): [int(idx) for idx in ranked_doc_ids] for doc_id, ranked_doc_ids in results.items()}
        #print(results)
        with open(f'data/base_/title/scores_/tit_base_{batch_start}.json', 'w') as json_file:
            json.dump(results, json_file)
        #sys.exit(0)

if __name__ == "__main__":
    #main_tfidf()
    main_faiss()

