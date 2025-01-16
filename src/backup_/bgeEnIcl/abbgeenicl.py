import os
import csv
import sys
import pickle
import faiss
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from FlagEmbedding import FlagICLModel

def load_main_data():
    main_data_ = pd.read_csv('/beegfs/schubotz/ankit/data/zbmath_complete_cleaned.csv')
    co_data_ = pd.read_csv('/beegfs/schubotz/ankit/data/zbmath_extkw.csv')
    main_data_ = pd.merge(main_data_,co_data_, on='document_id')
    main_data_ = main_data_.fillna('')
    print(main_data_.columns, main_data_.shape)
    return main_data_

main_data_ = load_main_data()
seedab1 = " ".join(main_data_.loc[main_data_['document_id'] == 5152710, 'text'].iloc[0].split("\n"))
idlrec1 = " ".join(main_data_.loc[main_data_['document_id'] == 6161945, 'text'].iloc[0].split("\n"))
seedab2 = " ".join(main_data_.loc[main_data_['document_id'] == 1222049, 'text'].iloc[0].split("\n"))
idlrec2 = " ".join(main_data_.loc[main_data_['document_id'] == 6224865, 'text'].iloc[0].split("\n"))
queries = ["how much protein should a female eat"]
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day."]
examples = [
  {'instruct': 'Given a abstract of a research paper from pure and applied mathematics find out abstracts that are similar.',
   'query': seedab1,
   'response': idlrec1},
  {'instruct': 'Given a abstract of a research paper from pure and applied mathematics find out abstracts that are similar.',
   'query': seedab2,
   'response': idlrec2}
]

model = FlagICLModel('BAAI/bge-en-icl', 
                     query_instruction_for_retrieval="Given a title of pure and applied mathematics research paper, retrieve relevant titles that are similar.",
                     examples_for_task=examples,  # set `examples_for_task=None` to use model without examples
                     use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

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

def main_faiss():
    #createIndex(main_data)
    #sys.exit(0)
    # Loop through each document_id in the test_ dataframe
    index = faiss.read_index("/beegfs/schubotz/ankit/code/zbReviewCit/bgeEnIcl/data/base_/abstracts/abs_nvmembv2_basemebd.index")
    #main_data_ = load_main_data()
    results = {}
    idealRecmnds, genRecmnds = {}, defaultdict(lambda:list())
    batch_doc_ids = getOlafSeedIDs()
    seedidlRcmnds = getidealrecommendations()
    #batch_titles = main_data[main_data['document_id'].isin(batch_doc_ids)]['title'].values
    batch_titles = main_data_.set_index('document_id').loc[batch_doc_ids]['text'].values
    #print(batch_titles)
    embeddings_batch = model.encode_queries(batch_titles)
    embeddings_batch = np.array(embeddings_batch, dtype='float32')
    faiss.normalize_L2(embeddings_batch)
    _, ranked_indices = index.search(embeddings_batch, 1000)
    for i, docu_id in enumerate(batch_doc_ids):
        listofscores = []
        for mn_id, idx_ in enumerate(ranked_indices[i]):
            listofscores.append([main_data_['document_id'].iloc[idx_], _[i][mn_id]])
        #ranked_doc_ids = [main_data['document_id'].iloc[idx_] for idx_ in ranked_indices[i]]
        results[docu_id] = listofscores
    #print(results)
    results = {int(doc_id): [[int(idx[0]), idx[1]] for idx in ranked_doc_ids] for doc_id, ranked_doc_ids in results.items()}
    #print(results)
    for EachRes in results.keys():
        genRecmnds[EachRes] += results[EachRes][1:1001]
        if EachRes not in idealRecmnds.keys():
            idealRecmnds[EachRes] = [str(star_) for star_ in seedidlRcmnds[int(EachRes)]]
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
    p3, p5, r_,r_1k, mrr_, ndcg_ = eval_metrics.main(idealRec, genRec)
    print(p3, p5, r_,r_1k, mrr_, ndcg_)

if __name__ == "__main__":
    #main_tfidf()
    main_faiss()

