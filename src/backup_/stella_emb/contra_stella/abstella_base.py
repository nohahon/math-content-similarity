import os
import sys
import pickle
import faiss
import json
import numpy as np
import pandas as pd
sys.path.append('noah/tf_idf_algrthmn/')
sys.path.append('/beegfs/schubotz/ankit/')
sys.path.append('/beegfs/schubotz/ankit/code/zbReviewCit/')
import zbCitData_st
from createResultsNoah import read_results_file
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INSTRUCTIONS = {
    "qa": {
        "query": "Instruct: Retrieve semantically similar text.\nQuery: ",
        "key": "",
    },
}

def getMainDataOld():
    data_absr = "/beegfs/schubotz/noah/arxMLiv/zbmath_abstracts.csv"
    data_titles = "/beegfs/schubotz/ankit/data/zbMATH_titles.csv"
    df1 = pd.read_csv(data_titles)
    df1 = df1.dropna(subset=['title'])
    df2 = pd.read_csv(data_absr)
    df2 = df2[['document_id', 'text']]
    df2 = df2.dropna(subset=['text'])
    merged_df = pd.merge(df1, df2, on='document_id')
    return merged_df


def getMSCSubsetMain(main_data,msc=''):
    if not msc:
        return main_data
    has_msc_df = main_data['classification'].map(lambda x: any([x.startswith(msc) for x in x.split()]))
    return main_data[has_msc_df==True]

def getMSCSubsetSplit(main_data,split,msc=''):
    main_data = getMSCSubsetMain(main_data,msc)
    print(msc)
    if not msc:
        missing_document_ids = split[~split['document_id'].isin(main_data['document_id'])]['document_id']
        print("Number of rows:", missing_document_ids.shape[0])
        #sys.exit(0)
        missing_data_dict = {key:'' for key in main_data.keys()}
        missing_data_dict['document_id']=missing_document_ids
        missing_data = pd.DataFrame(missing_data_dict)
        main_data = pd.concat([main_data, missing_data], ignore_index=True)
        #createIndex(main_data)
        return split, main_data
    all_doc_ids = main_data['document_id'].tolist()
    return split[split['document_id'].isin(all_doc_ids)==True], main_data


def get_test_data(main_data,subset,msc):
    data_ = "/beegfs/schubotz/ankit/data/zbReviewCitData/citation_dataset.csv"
    dataFr = zbCitData_st.load_csv_to_dataframe(data_)  # Load main dataset
    train_, test_, valid_ = zbCitData_st.split_dataframe(dataFr)
    if subset=='test':
        subset_df = test_
    elif subset=='train':
        subset_df = train_
    else:
        subset_df = valid_ 
    subset_df, main_data = getMSCSubsetSplit(main_data,subset_df,msc)
    return subset_df, main_data

    
def fix_jsons_backward(main_data,scores_dir):
    #import multiprocessing
    #def fix_scores
    main_data_old = getMainDataOld()
    join_old_new = main_data_.join(main_data_old,rsuffix='_old')[['document_id','document_id_old']]
    fix_dict = pd.Series(join_old_new.document_id_old.values,index=join_old_new.document_id).to_dict()
    os.makedirs(os.path.join(scores_dir,'corrected_scores'),exist_ok=True)
    for scores_name in os.listdir(scores_dir):
        print(scores_name)
        if os.path.isfile(os.path.join(scores_dir,scores_name)):
            with open(os.path.join(scores_dir,scores_name)) as scores_file:
                scores_dict = json.load(scores_file)
                for i,key in enumerate(scores_dict.keys()):
                    if i%100==0:
                        print(i)
                    wrong_ids = [doc_id for doc_id,score in scores_dict[key]]
                    #wrong_ids_data = pd.DataFrame({'document_id':wrong_ids})
                    #sorted_matches = wrong_ids_data.join(join_old_new.set_index('document_id'),on='document_id',how='inner')
                    scores = [score for doc_id,score in scores_dict[key]]
                    fixed_ids = [int(fix_dict[doc_id]) for doc_id in wrong_ids]
                    fixed_scores = list(zip(fixed_ids,scores))
                    scores_dict[key]= fixed_scores
            with open(os.path.join(scores_dir,'corrected_scores',scores_name),'w') as scores_out:
                json.dump(scores_dict,scores_out)

def createIndex(main_data_,used_data,msc='',model_name_or_path='base'):
    print(used_data)
    if isinstance(used_data,str):
        used_data_str = used_data.replace(' ','_')
    else:
        used_data_str = "_x_".join([feat.replace(' ','_') for feat in used_data])
    main_data_ = getMSCSubsetMain(main_data_,msc)
    instruction = INSTRUCTIONS["qa"]
    model_path = "dunzhang/stella_en_400M_v5" if model_name_or_path=='base' else model_name_or_path
    if model_name_or_path.endswith('/'):
        model_name_or_path = model_name_or_path[:-1]
    if model_name_or_path=='base':
        model_name = 'base' 
    elif model_name_or_path.endswith('final'):
        model_name = model_name_or_path.split("/")[-2]
    else:
        model_name = model_name_or_path.split("/")[-1]
    model = SentenceTransformer(model_path, device='cuda', trust_remote_code=True)
    embedding_dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(embedding_dim)
    #batch_size = 5000
    batch_size = 64
    for start_idx in range(0, len(main_data_), batch_size):
    #for start_idx in range(0, 10, batch_size):
        print(start_idx,flush=True)
        end_idx = min(start_idx + batch_size, len(main_data_))
        if isinstance(used_data,str):
            titles_batch = main_data_[used_data].iloc[start_idx:end_idx].tolist()
        else:
            titles_batch = list(main_data_[used_data].iloc[start_idx:end_idx].itertuples(index=False, name=None))
            titles_batch = ["\n".join([used_data[i]+": "+tup[i] for i in range(len(used_data))]) for tup in titles_batch]
        print(titles_batch[:3],flush=True)
        embeddings_batch = model.encode(titles_batch, convert_to_numpy=True, device='cuda')
        faiss.normalize_L2(embeddings_batch)
        index.add(embeddings_batch.astype('float32'))
    os.makedirs(f"data/{model_name}_{msc}/{used_data_str}", exist_ok=True)
    faiss.write_index(index, f"data/{model_name}_{msc}/{used_data_str}/stella_{model_name}mebd.index")



def main_faiss(main_data, used_data, cross_features=False, subset='test', msc='', model_name_or_path='base'):
    if isinstance(used_data,str):
        used_data_str = used_data.replace(' ','_')
    else:
        used_data_str = "_x_".join([feat.replace(' ','_') for feat in used_data])
    instruction = INSTRUCTIONS["qa"]
    model_path = "dunzhang/stella_en_400M_v5" if model_name_or_path=='base' else model_name_or_path
    if model_name_or_path.endswith('/'):
        model_name_or_path = model_name_or_path[:-1]
    if model_name_or_path=='base':
        model_name = 'base' 
    elif model_name_or_path.endswith('final'):
        model_name = model_name_or_path.split("/")[-2]
    else:
        model_name = model_name_or_path.split("/")[-1]
    model = SentenceTransformer(model_path, device='cuda', trust_remote_code=True)
    subset_df, main_data = get_test_data(main_data,subset,msc)
    num_scores = 1000 if subset=='test' else 100 
    # Split data into train, test, valid
    #main_data = zbCitData_st.getMainData().fillna('')
    # Loop through each document_id in the test_ dataframe
    #batch_size = 5000
    batch_size = 64
    for batch_start in range(0, len(subset_df['document_id']), batch_size):
        embedding_path = f'data/{model_name}_{msc}/{used_data_str}/{subset}/stella_queries_{batch_start}_embed.pkl' 
        if os.path.exists(embedding_path):
            continue                
        batch_doc_ids = subset_df['document_id'][batch_start:batch_start + batch_size]
        #batch_titles = main_data[main_data['document_id'].isin(batch_doc_ids)]['title'].values
        if isinstance(used_data,str):
            batch_titles = main_data.set_index('document_id').loc[batch_doc_ids][used_data].values
        else:
            batch_titles = list(main_data.set_index('document_id').loc[batch_doc_ids][used_data].itertuples(index=False, name=None))
            batch_titles = ["| ".join([used_data[i]+": "+tup[i] for i in range(len(used_data))]) for tup in batch_titles]
        embeddings_batch = model.encode(batch_titles, convert_to_numpy=True, device='cuda')
        faiss.normalize_L2(embeddings_batch)
        os.makedirs(f'data/{model_name}_{msc}/{used_data_str}/{subset}/',exist_ok=True)
        with open(f'data/{model_name}_{msc}/{used_data_str}/{subset}/stella_queries_{batch_start}_embed.pkl','wb') as fh:
            pickle.dump(embeddings_batch,fh)
    if cross_features == False:
        cross_features = [used_data]
    for feature in cross_features:
        if isinstance(feature,str):
            feature_str = feature.replace(' ','_')
        else:
            feature_str = "_x_".join([feat.replace(' ','_') for feat in feature])
        print("feature: ",feature,flush=True) 
        print("loading index",flush=True)
        index_dir = f"data/{model_name}_{msc}/{feature_str}/"
        for file_name in os.listdir(index_dir):
            if file_name.endswith(".index"):
                index = faiss.read_index(os.path.join(index_dir,file_name))
                break
        print("index loaded", flush=True)
        for batch_start in range(0, len(subset_df['document_id']), batch_size):
            print("jsons_path: ",f'data/{model_name}_{msc}/{used_data_str}_vs_{feature_str}/scores/{subset}/{model_name}_{batch_start}.json')
            if os.path.exists(f'data/{model_name}_{msc}/{used_data_str}_vs_{feature_str}/scores/{subset}/{model_name}_{batch_start}.json'):
                continue
            print(batch_start)
            results = {}
            batch_doc_ids = subset_df['document_id'][batch_start:batch_start + batch_size]
            with open(f'data/{model_name}_{msc}/{used_data_str}/{subset}/stella_queries_{batch_start}_embed.pkl','rb') as fh:
                embeddings_batch = pickle.load(fh)
            scores, ranked_indices = index.search(embeddings_batch, num_scores)
            for i, docu_id in enumerate(batch_doc_ids):
                try:
                    ranked_doc_ids = [(main_data['document_id'].iloc[idx_],score) for score,idx_ in zip(scores[i],ranked_indices[i])]
                    results[docu_id] = ranked_doc_ids
                except:
                    results[docu_id] = []
                if i<10:
                    print(results[docu_id][:10],flush=True)
            #print(results)
            results = {int(doc_id): [(int(idx),str(score)) for idx,score in scores] for doc_id, scores in results.items()}
            #print(results)
            jsons_path = f"data/{model_name}_{msc}/{used_data_str}_vs_{feature_str}/scores/{subset}/"
            os.makedirs(jsons_path, exist_ok=True)
            with open(f'data/{model_name}_{msc}/{used_data_str}_vs_{feature_str}/scores/{subset}/{model_name}_{batch_start}.json', 'w') as json_file:
                json.dump(results, json_file)
            #sys.exit(0)
            json_data = read_results_file(jsons_path,split=subset)

def load_main_data():
    main_data_ = pd.read_csv('/beegfs/schubotz/ankit/data/zbmath_complete_cleaned.csv')
    co_data_ = pd.read_csv('/beegfs/schubotz/ankit/data/zbmath_extkw.csv')
    main_data_ = pd.merge(main_data_,co_data_, on='document_id')
    main_data_ = main_data_.fillna('')
    return main_data_

main_data_ = load_main_data()

if __name__ == "__main__":
    #main_tfidf()
    #model_dir = '/beegfs/schubotz/ankit/code/zbReviewCit/stella_emb/contra_stella/noah/models/stella_400_5e6_256_20241213-185254/final'
    model_dir = '/beegfs/schubotz/ankit/code/zbReviewCit/stella_emb/contra_stella/noah/models/stella_400_5e6_2048_20241215-155348/checkpoint-100'
    createIndex(main_data_,['title','text'],msc='14',model_name_or_path=model_dir)
    main_faiss(main_data_,['title','text'],msc='14',subset='valid',model_name_or_path=model_dir)
    #main_faiss(main_data_,'text',subset='valid')
    #print("next task")
    #createIndex(main_data_,'mscs in text')
    #main_faiss(main_data_,'mscs in text')
    #createIndex(main_data_,'extended_keywords')
    #main_faiss(main_data_,'extended_keywords')
    #createIndex(main_data_,['keyword','mscs in text'])
    #main_faiss(main_data_,'extended_keywords',cross_features=['extended_keywords'],subset='train',msc='14')
