import os
import sys
import csv
import json
import random
import pandas as pd
import zbCitData_st
sys.path.append("../")
import eval_metrics
from collections import defaultdict
from ast import literal_eval

data_ = "/beegfs/schubotz/ankit/data/zbReviewCitData/citation_dataset_mod.csv"
#dataFr = zbCitData_st.load_csv_to_dataframe(data_)  # Load main dataset
dataFr = pd.read_csv(data_,converters={"citations_list": literal_eval})
train_, test_, valid_ = zbCitData_st.split_dataframe(dataFr)  # Split data into train, test, valid
#print(train_.shape[0] ,test_.shape[0], valid_.shape[0])
#sys.exit(0)

def idlRecommendations(doc_id,split):
    val_ret = split[split['document_id'] == int(doc_id)]['citation_de'].values[0]
    return val_ret.split("; ")

def getIdlandGenrecs(random_eles,split,file_path):
    dictrm = dict()
    for eachRE in random_eles:
        idl_recmnds = idlRecommendations(eachRE,split)
        dictrm[eachRE] = [idl_recmnds]
    getallfls = os.listdir(file_path)
    for eachF in getallfls:
        with open(file_path+eachF, 'r') as json_file:
            # Load the content of the file into a Python dictionary
            data = json.load(json_file)
        for eackDoc in data.keys():
            if eackDoc in random_eles:
                dictrm[eackDoc].append([str(ea_) for ea_ in data[eackDoc][:4]])
    print(dictrm)

def results_json_read(file_path,split='test',maxk=10):
    scores_dict = {}
    for i,eachF in enumerate(getallfls):
        print(eachF)
        with open(file_path+eachF, 'r') as json_file:
            # Load the content of the file into a Python dictionary
            data = json.load(json_file)
            try:
                list_true = isinstance(data[list(data.keys())[0]][0],list)
            except:
                list_true = isinstance(data[list(data.keys())[1]][0],list)
        for eackDoc in data.keys():
            if list_true:
                gen_recmnds = [{'corpus_id':str(ea_[0]),'score':ea_[1]} for ea_ in data[eackDoc][1:max_k+1]]
            else:
                gen_recmnds = [{'corpus_id':str(ea_),'score':1/i} for i,ea_ in enumerate(data[eackDoc][1:max_k+1])]
            scores_dict[str(eackDoc)]=gen_recmnds
        return scores_dict            

def read_results_file(file_path,split='test'):
    #from abstella_base import get_test_data, load_main_data
    if split=='test':
        split_df = test_
    elif split=='train':
        split_df = train_
    else:
        split_df = valid_
    seed_ids = split_df['document_id'].tolist()
    seed_ids = [str(seed_id) for seed_id in seed_ids]
    getallfls = os.listdir(file_path)
    print(getallfls)
    genRecmnds, idealRecmnds = [], []
    seeddocids = list()
    countDocs = 0
    with open("example_recmnds.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["seed", "recommendation"])
        for i,eachF in enumerate(getallfls):
            print(eachF)
            with open(file_path+eachF, 'r') as json_file:
                # Load the content of the file into a Python dictionary
                data = json.load(json_file)
                try:
                    list_true = isinstance(data[list(data.keys())[0]][0],list)
                except:
                    list_true = isinstance(data[list(data.keys())[1]][0],list)
            for eackDoc in data.keys():
                if not str(eackDoc) in seed_ids:
                    continue
                countDocs += 1
                seeddocids.append(int(eackDoc))
                if list_true:
                    gen_recmnds = [str(ea_[0]) for ea_ in data[eackDoc][1:11]]
                else:
                    gen_recmnds = [str(ea_) for ea_ in data[eackDoc][1:11]]
                genRecmnds.append(gen_recmnds)
                for eachGenRec in gen_recmnds:
                    writer.writerow([eackDoc, eachGenRec])
                    #if countDocs > 20:
                    #    sys.exit(0)
    gen_rec_df = pd.DataFrame({'document_id':seeddocids, 'generated recs':genRecmnds})
    split_df = gen_rec_df.join(split_df.set_index('document_id'), on='document_id',how='inner')
    return split_df

def get_metrics(file_path,split='test'):
    split_df = read_results_file(file_path,split='test')
    #print(split_df)
    #random.seed(42)#
    #rand_eles = random.sample(seeddocids, 3)
    #print(rand_eles)
    #getIdlandGenrecs(rand_eles,split_df,file_path)
    #sys.exit(0)
    #dictcits = defaultdict(lambda: 0)
    #for eachIdl in idl_recmnds:
    #    dictcits[len(eachIdl)] += 1
    #print("Distribution of citations: ", dictcits.keys())
    #print(dictcits)
    #print(len(genRecmnds), len(idealRecmnds))
    #sys.exit(0)
    
    idealRecmnds = split_df['citations_list'].tolist()
    genRecmnds = split_df['generated recs'].tolist()
    p3, p5, r_, mrr_, ndcg_ = eval_metrics.main(idealRecmnds, genRecmnds)
    print(p3, p5, r_, mrr_, ndcg_)

def create_results_alt(split='valid',msc='',used_data=['title','text']):
    from contra_stella_gino import construct_ir_evaluator
    evaluator = construct_ir_evaluator(split,msc,used_data)
    split_df = read_results_file(file_path,split)
    
    

# Example usage
if __name__ == "__main__":
    #file_path = "/beegfs/schubotz/ankit/code/zbReviewCit/stella_emb/stella_base/data/base_/extended_keywords_vs_extended_keywords/scores_/"  
    file_path = "/beegfs/schubotz/ankit/code/zbReviewCit/stella_emb/stella_base/data/checkpoint-50_14/title_x_text_vs_title_x_text/scores/valid/"
    json_data = get_metrics(file_path,split='valid')

