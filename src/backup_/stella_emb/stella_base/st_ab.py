import csv
import sys
import pickle
import faiss
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import eval_metrics

INSTRUCTIONS = {
    "qa": {
        "query": "Instruct: Retrieve semantically similar text.\nQuery: ",
        "key": "",
    },
}

file_path = "/beegfs/schubotz/ankit/data/recommendationPairs.csv"

def getidealrecommendations():
    listDocs = dict()
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader: 
            IdsandRec = list(filter(None, row))
            listDocs[int(IdsandRec[0])] = [int(el_) for el_ in IdsandRec[1:]]
    return listDocs

def getOlafSeedIDs():
    first_elements = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row:  # Ensure the row is not empty
                first_elements.append(int(row[0]))
    return first_elements

def createIndex(main_data_):
    """ Creating FAISS vector DB """
    instruction = INSTRUCTIONS["qa"]
    model = SentenceTransformer("trained_models/trained_model", device='cuda', trust_remote_code=True)
    embedding_dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(embedding_dim)
    batch_size = 5000
    for start_idx in range(0, len(main_data_), batch_size):
        end_idx = min(start_idx + batch_size, len(main_data_))
        titles_batch = main_data_['text'].iloc[start_idx:end_idx].tolist()
        embeddings_batch = model.encode(titles_batch, convert_to_numpy=True, device='cuda')
        faiss.normalize_L2(embeddings_batch)
        index.add(embeddings_batch.astype('float32'))
    faiss.write_index(index, "data/trained_models/abstrs/rand_negs/ab_stella_randneg.index")

def main_faiss(indexes_h):
    instruction = INSTRUCTIONS["qa"]
    model_dir = '/beegfs/schubotz/ankit/code/zbReviewCit/stella_emb/contra_stella/noah/models/stella_400_5e6_{batch_size}/'
    model = SentenceTransformer(model_dir, device='cuda', trust_remote_code=True)
    main_data_ = load_main_data()
    idealRecmnds, genRecmnds = {}, defaultdict(lambda:list())
    for index_ in indexes_h:
        index = faiss.read_index(index_)
        results = {}
        batch_doc_ids = getOlafSeedIDs()
        seedidlRcmnds = getidealrecommendations()
        batch_titles = main_data_.set_index('document_id').loc[batch_doc_ids]['text'].values
        embeddings_batch = model.encode(batch_titles, convert_to_numpy=True, device='cuda')
        faiss.normalize_L2(embeddings_batch)
        _, ranked_indices = index.search(embeddings_batch, 1000)
        #print(type(_))
        #print(type(ranked_indices))
        for i, docu_id in enumerate(batch_doc_ids):
            #try:
            listofscores = []
            #print("Current seed IDs:", docu_id)
            for mn_id, idx_ in enumerate(ranked_indices[i]):
                listofscores.append([main_data_['document_id'].iloc[idx_], _[i][mn_id]])
                #print([main_data_['document_id'].iloc[idx_], _[i][mn_id]])
                #sys.exit(0)
            #ranked_doc_ids = [main_data_['document_id'].iloc[idx_] for idx_ in ranked_indices[i]]
            results[docu_id] = listofscores
            #except:
            #    results[docu_id] = []
        results = {int(doc_id): [[int(idx[0]), idx[1]] for idx in ranked_doc_ids] for doc_id, ranked_doc_ids in results.items()}
        for EachRes in results.keys():
            #print("Doing for seed: ", EachRes)
            #print(type(EachRes))
            genRecmnds[EachRes] += results[EachRes][1:1001]
            #print("Generated recmnds: ", results[EachRes][1:11])
            if EachRes not in idealRecmnds.keys():
                #print(seedidlRcmnds.keys())
                idealRecmnds[EachRes] = [str(star_) for star_ in seedidlRcmnds[int(EachRes)]]
                #print(type(list(seedidlRcmnds.keys())[0]))
                #sys.exit(0)
    sortedGenRec = dict()
    for eackKud in genRecmnds.keys():
        sortedGenRec[eackKud] = sorted(genRecmnds[eackKud], key=lambda x: float(x[1]), reverse=True)
    genRec, idealRec = [], []
    for eachK in idealRecmnds.keys():
        idealRec.append(idealRecmnds[eachK])
        seen = set()
        uniq_list = []
        for x_ in sortedGenRec[eachK]:
            if str(x_[0]) not in seen:
                uniq_list.append(str(x_[0]))
                seen.add(str(x_[0]))
        genRec.append(uniq_list)
    #print("sample sir: ", idealRec[0], genRec[0][0])
    #print("sample sir type: ", type(idealRec[0][0]), type(genRec[0][0])) #int, str
    p3, p5, r_,r_1k, mrr_, ndcg_ = eval_metrics.main(idealRec, genRec)
    print(p3, p5, r_,r_1k, mrr_, ndcg_)

def load_main_data():
    main_data_ = pd.read_csv('/beegfs/schubotz/ankit/data/zbmath_complete_cleaned.csv')
    co_data_ = pd.read_csv('/beegfs/schubotz/ankit/data/zbmath_extkw.csv')
    main_data_ = pd.merge(main_data_,co_data_, on='document_id')
    main_data_ = main_data_.fillna('')
    print(main_data_.columns, main_data_.shape)
    return main_data_

if __name__ == "__main__":
    #main_tfidf()
    ind_a = ["/beegfs/schubotz/ankit/code/zbReviewCit/stella_emb/stella_base/data/stella_400_5e6_{batch_size}_/keyword/stella_stella_400_5e6_{batch_size}mebd.index"]
    indexes_ = ["/beegfs/schubotz/ankit/code/zbReviewCit/stella_emb/stella_base/data/stella_400_5e6_{batch_size}_/keyword/stella_stella_400_5e6_{batch_size}mebd.index", "/beegfs/schubotz/ankit/code/zbReviewCit/stella_emb/stella_base/data/stella_400_5e6_{batch_size}_/text/stella_stella_400_5e6_{batch_size}mebd.index", "/beegfs/schubotz/ankit/code/zbReviewCit/stella_emb/stella_base/data/stella_400_5e6_{batch_size}_/title/stella_stella_400_5e6_{batch_size}mebd.index", "/beegfs/schubotz/ankit/code/zbReviewCit/stella_emb/stella_base/data/stella_400_5e6_{batch_size}_/mscs_in_text/stella_stella_400_5e6_{batch_size}mebd.index"]
    main_faiss(indexes_)
    #main_data_ = load_main_data()
