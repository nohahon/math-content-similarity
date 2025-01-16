import os
import sys
import pickle
import faiss
import json
import numpy as np
import pandas as pd
sys.path.append('../../tf_idf_algrthmn/')
sys.path.append('/beegfs/schubotz/ankit/')
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

def getMSCSubsetMain(main_data,msc=''):
    has_msc_df = main_data['classification'].map(lambda x: any([x.startswith(msc) for x in x.split()]))
    return main_data[has_msc_df==True]

def getMSCSubsetSplit(main_data,split,msc=''):
    main_data = getMSCSubsetMain(main_data,msc)
    if not msc:
        missing_document_ids = subset_df[~subset_df['document_id'].isin(main_data['document_id'])]['document_id']
        print("Number of rows:", missing_document_ids.shape[0])
        #sys.exit(0)
        missing_data = pd.DataFrame({'document_id': missing_document_ids, used_data: ''})
        main_data = pd.concat([main_data, missing_data], ignore_index=True)
        #createIndex(main_data)
    all_doc_ids = main_data['document_id'].tolist()
    return split[split['document_id'].isin(all_doc_ids)==True], main_data
    
    

def createIndex(main_data_,used_data,msc=''):
    print(used_data)
    if isinstance(used_data,str):
        used_data_str = used_data.replace(' ','_')
    else:
        used_data_str = "_x_".join([feat.replace(' ','_') for feat in used_data])
    main_data_ = getMSCSubsetMain(main_data_,msc)
    instruction = INSTRUCTIONS["qa"]
    model = SentenceTransformer("dunzhang/stella_en_400M_v5", device='cuda', trust_remote_code=True)
    embedding_dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(embedding_dim)
    batch_size = 5000
    for start_idx in range(0, len(main_data_), batch_size):
    #for start_idx in range(0, 10, batch_size):
        print(start_idx)
        end_idx = min(start_idx + batch_size, len(main_data_))
        if isinstance(used_data,str):
            titles_batch = main_data_[used_data].iloc[start_idx:end_idx].tolist()
        else:
            titles_batch = list(main_data_[used_data].iloc[start_idx:end_idx].itertuples(index=False, name=None))
            titles_batch = ["| ".join([used_data[i]+": "+tup[i] for i in range(len(used_data))]) for tup in titles_batch]
        print(titles_batch[:3])
        embeddings_batch = model.encode(titles_batch, convert_to_numpy=True, device='cuda')
        faiss.normalize_L2(embeddings_batch)
        index.add(embeddings_batch.astype('float32'))
    os.makedirs(f"data/base_{msc}/{used_data_str}", exist_ok=True)
    faiss.write_index(index, f"data/base_{msc}/{used_data_str}/stella_basemebd.index")

def main_faiss(main_data, used_data, cross_features=False, subset='test', msc=''):
    if isinstance(used_data,str):
        used_data_str = used_data.replace(' ','_')
    else:
        used_data_str = "_x_".join([feat.replace(' ','_') for feat in used_data])
    instruction = INSTRUCTIONS["qa"]
    model = SentenceTransformer("dunzhang/stella_en_400M_v5", device='cuda', trust_remote_code=True)
    data_ = "/beegfs/schubotz/ankit/data/zbReviewCitData/citation_dataset.csv"
    dataFr = zbCitData_st.load_csv_to_dataframe(data_)  # Load main dataset
    train_, test_, valid_ = zbCitData_st.split_dataframe(dataFr)
    if subset=='test':
        subset_df = test_
    elif subset=='train':
        subset_df = train_
    else:
        subset_df = test_ 
    subset_df, main_data = getMSCSubsetSplit(main_data,subset_df,msc)
    num_scores = 1000 if subset=='test' else 100 
    # Split data into train, test, valid
    #main_data = zbCitData_st.getMainData().fillna('')
    # Loop through each document_id in the test_ dataframe
    batch_size = 5000
    for batch_start in range(0, len(subset_df['document_id']), batch_size):
        embedding_path = f'data/base_{msc}/{used_data_str}_{subset}/stella_queries_{batch_start}_embed.pkl' 
        if os.path.exists(embedding_path):
            continue                
        batch_doc_ids = subset_df['document_id'][batch_start:batch_start + batch_size]
        #batch_titles = main_data[main_data['document_id'].isin(batch_doc_ids)]['title'].values
        if isinstance(used_data,str):
            batch_titles = main_data.set_index('document_id').loc[batch_doc_ids][used_data].values
        else:
            batch_titles = list(main_data.set_index('document_id').loc[batch_doc_ids][used_data].itertuples(index=False, name=None))
            batch_titles = ["| ".join([used_data[i]+": "+tup[i] for i in range(len(used_data))]) for tup in titles_batch]
        embeddings_batch = model.encode(batch_titles, convert_to_numpy=True, device='cuda')
        faiss.normalize_L2(embeddings_batch)
        with open(f'data/base_{msc}/{used_data_str}_{subset}/stella_queries_{batch_start}_embed.pkl','wb') as fh:
            pickle.dump(embeddings_batch,fh)
    if cross_features == False:
        cross_features = [used_data]
    for feature in cross_features:
        if isinstance(feature,str):
            feature_str = feature.replace(' ','_')
        else:
            feature_str = "_x_".join([feat.replace(' ','_') for feat in feature])
        print("feature: ",feature) 
        print("loading index")
        index_dir = f"data/base_{msc}/{feature_str}/"
        for file_name in os.listdir(index_dir):
            if file_name.endswith(".index"):
                index = faiss.read_index(os.path.join(index_dir,file_name))
                break
        print("index loaded")
        for batch_start in range(0, len(subset_df['document_id']), batch_size):
            if os.path.exists(f'data/base_{msc}/{used_data_str}_vs_{feature_str}/scores_{subset}/base_{batch_start}.json'):
                continue
            results = {}
            batch_doc_ids = subset_df['document_id'][batch_start:batch_start + batch_size]
            with open(f'data/base_{msc}/{used_data_str}_{subset}/stella_queries_{batch_start}_embed.pkl','rb') as fh:
                embeddings_batch = pickle.load(fh)
            scores, ranked_indices = index.search(embeddings_batch, num_scores)
            for i, docu_id in enumerate(batch_doc_ids):
                try:
                    ranked_doc_ids = [(main_data['document_id'].iloc[idx_],score) for score,idx_ in zip(scores[i],ranked_indices[i])]
                    results[docu_id] = ranked_doc_ids
                except:
                    results[docu_id] = []
                if i<10:
                    print(results[docu_id][:10])
            #print(results)
            results = {int(doc_id): [(int(idx),str(score)) for idx,score in scores] for doc_id, scores in results.items()}
            #print(results)
            os.makedirs(f"data/base_{msc}/{used_data_str}_vs_{feature_str}/scores_{subset}/", exist_ok=True)
            with open(f'data/base_{msc}/{used_data_str}_vs_{feature_str}/scores_{subset}/base_{batch_start}.json', 'w') as json_file:
                json.dump(results, json_file)
            #sys.exit(0)

if __name__ == "__main__":
    #main_tfidf()
    main_data_ = pd.read_csv('/beegfs/schubotz/ankit/data/zbmath_complete.csv')
    co_data_ = pd.read_csv('/beegfs/schubotz/ankit/data/zbmath_extkw.csv')
    main_data_ = pd.merge(main_data_,co_data_, on='document_id')
    main_data_ = main_data_.fillna('')
    #createIndex(main_data_,'keyword')
    #main_faiss(main_data_,'keyword')
    #print("next task")
    #createIndex(main_data_,'mscs in text')
    #main_faiss(main_data_,'mscs in text')
    #createIndex(main_data_,'extended_keywords')
    #main_faiss(main_data_,'extended_keywords')
    #createIndex(main_data_,['keyword','mscs in text'])
    main_faiss(main_data_,'extended_keywords',cross_features=['extended_keywords'],subset='train',msc='14')
import os
import sys
import pickle
import faiss
import json
import numpy as np
