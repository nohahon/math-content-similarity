import os
import sys
import csv
import json
import random
import pandas as pd
from tf_idf_algrthmn import zbCitData_st
import eval_metrics
from collections import defaultdict

data_ = "/beegfs/schubotz/ankit/data/zbReviewCitData/citation_dataset.csv"
dataFr = zbCitData_st.load_csv_to_dataframe(data_)  # Load main dataset
train_, test_, valid_ = zbCitData_st.split_dataframe(dataFr)  # Split data into train, test, valid
print("This is what train, text and validation sample looks like: ", train_.shape[0] ,test_.shape[0], valid_.shape[0])
main_df = zbCitData_st.getMainData(usenans=True)[['document_id', 'title', 'text']].dropna()
main_df["document_id"] = pd.to_numeric(main_df["document_id"], errors="coerce")
main_df["document_id"] = main_df["document_id"].fillna(0).astype("string")
intersection_ = pd.merge(test_.astype(str), main_df, on='document_id')
intersection_ = set(intersection_["document_id"]) # seeds from test for whch we have data in main_df
test_citations = set(
    elem.strip()
    for citations in test_['citation_de'].dropna()
    for elem in citations.split('; ')
)
print("Total citation in test DF: ", len(test_citations))
#print("Datatype set ele: ", type(list(test_citations)[0]))
intersection_cit = main_df[main_df["document_id"].isin(test_citations)]
intersection_cit = set(intersection_cit["document_id"]) # recmnds from test for which we have data in main_df
print("Cits for which we have data available in Df: ", len(intersection_cit))
print("Seeds for whcih we have data avaibal in Df: ", len(intersection_))

def idlRecommendations(doc_id,split):
    try:
        val_ret = split[split['document_id'] == int(doc_id)]['citation_de'].values[0]
    except:
        print(split[split['document_id'] == int(doc_id)]['citation_de'].values)
    return val_ret.split("; ")

def read_results_file(file_path,split='test'):
    if split=='test':
        split_df = test_
    elif split=='train':
        split_df = train_
    else:
        split_df = valid_
    getallfls = os.listdir(file_path)
    genRecmnds, idealRecmnds = [], []
    for i,eachF in enumerate(getallfls):
        with open(file_path+eachF, 'r') as json_file:
            # Load the content of the file into a Python dictionary
            data = json.load(json_file)
            try:
                list_true = isinstance(data[list(data.keys())[0]][0],list)
            except:
                list_true = isinstance(data[list(data.keys())[1]][0],list)
        for eackDoc in data.keys():
            if eackDoc in intersection_: # check if we have data for this seed
                idl_recmnds = idlRecommendations(eackDoc,split_df)
                if set(idl_recmnds).issubset(intersection_cit): #check if we have data for both recmnds
                    if len(idl_recmnds) == 0:
                        print(eackDoc)
                    idealRecmnds.append(idl_recmnds)
                    if list_true:
                        gen_recmnds = [str(ea_[0]) for ea_ in data[eackDoc][:1000]]
                    else:
                        gen_recmnds = [str(ea_) for ea_ in data[eackDoc][:1000]] 
                    genRecmnds.append(gen_recmnds)
    p3, p5, r_, r_1k, mrr_, ndcg_ = eval_metrics.main(idealRecmnds, genRecmnds)
    print(p3, p5, r_, r_1k, mrr_, ndcg_)

def getR_at_1000forcombinedFeat(file_path,split='test'):
    if split=='test':
        split_df = test_
    elif split=='train':
        split_df = train_
    else:
        split_df = valid_
    genRecmnds, idealRecmnds = defaultdict(lambda:list()), {}
    for eachFpath in file_path:
        getallfls = os.listdir(eachFpath)
        for i,eachF in enumerate(getallfls):
            with open(eachFpath+eachF, 'r') as json_file:
                data = json.load(json_file)
                try:
                    list_true = isinstance(data[list(data.keys())[0]][0],list)
                except:
                    list_true = isinstance(data[list(data.keys())[1]][0],list)
            for eackDoc in data.keys():
                if eackDoc in intersection_:
                    idl_recmnds = idlRecommendations(eackDoc,split_df)
                    if set(idl_recmnds).issubset(intersection_cit):
                        if len(idl_recmnds) == 0:
                            print(eackDoc)
                        if eackDoc not in idealRecmnds.keys():
                            idealRecmnds[eackDoc] = idl_recmnds
                        if list_true:
                            gen_recmnds = [[ea_[0], float(ea_[1])] for ea_ in data[eackDoc][1:1000]]
                            #gen_recmnds = [str(ea_[0]) for ea_ in data[eackDoc][:1000]]
                        else:
                            #gen_recmnds = [str(ea_) for ea_ in data[eackDoc][:1000]]
                            gen_recmnds = data[eackDoc][:1000]
                            print("Am I even going to this loop")
                        genRecmnds[eackDoc] += gen_recmnds
    sortedGenRec = dict()
    for eackKud in genRecmnds.keys():
        #print(eackKud , genRecmnds[eackKud][:20])
        sortedGenRec[eackKud] = sorted(genRecmnds[eackKud], key=lambda x: float(x[1]), reverse=True)
        #print(sortedGenRec[eackKud][:20])
        #sys.exit(0)
    print("Len of ideal rec and gen rec: ", len(idealRecmnds), len(sortedGenRec))
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
    print("should be 4000: ", len(genRec[0]))
    p3, p5, r_, r_1k, mrr_, ndcg_ = eval_metrics.main(idealRec, genRec)
    print(p3, p5, r_, r_1k, mrr_, ndcg_)

# Example usage
if __name__ == "__main__":
    file_path = ["/beegfs/schubotz/ankit/code/zbReviewCit/stella_emb/stella_base/data/stella_400_5e6_{batch_size}_/text_vs_text/scores/test/"]
    #read_results_file(file_path,split='test')
    file_paths = ["/beegfs/schubotz/ankit/code/zbReviewCit/stella_emb/stella_base/data/stella_400_5e6_{batch_size}_/text_vs_text/scores/test/", "/beegfs/schubotz/ankit/code/zbReviewCit/stella_emb/stella_base/data/stella_400_5e6_{batch_size}_/title_vs_title/scores/test/", "/beegfs/schubotz/ankit/code/zbReviewCit/stella_emb/stella_base/data/stella_400_5e6_{batch_size}_/keyword_vs_keyword/scores/test/", "/beegfs/schubotz/ankit/code/zbReviewCit/stella_emb/stella_base/data/stella_400_5e6_{batch_size}_/mscs_in_text_vs_mscs_in_text/scores/test/"]
    getR_at_1000forcombinedFeat(file_paths,split='test')

