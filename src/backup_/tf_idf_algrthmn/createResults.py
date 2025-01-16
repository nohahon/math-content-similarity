import os
import sys
import csv
import json
import random
import zbCitData_st
sys.path.append("../")
import eval_metrics
from collections import defaultdict

data_ = "/beegfs/schubotz/ankit/data/zbReviewCitData/citation_dataset.csv"
dataFr = zbCitData_st.load_csv_to_dataframe(data_)  # Load main dataset
train_, test_, valid_ = zbCitData_st.split_dataframe(dataFr)  # Split data into train, test, valid

def idlRecommendations(doc_id):
    val_ret = test_[test_['document_id'] == int(doc_id)]['citation_de'].values
    print("Ideal recommendations: ", val_ret)
    sys.exit(0)
    return val_ret

def getIdlandGenrecs(random_eles):
    dictrm = dict()
    random_eles = ['1313667', '1364152', '907268']
    for eachRE in random_eles:
        idl_recmnds = idlRecommendations(eachRE)
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

def read_results_file(file_path):
    getallfls = os.listdir(file_path)
    genRecmnds, idealRecmnds = [], []
    seeddocids = list()
    countDocs = 0
    countSeedCr = 0
    with open("example_recmnds.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["seed", "recommendation", "similarity_score", "version_identifier"])
        for i,eachF in enumerate(getallfls):
            print(eachF)
            with open(file_path+eachF, 'r') as json_file:
                data = json.load(json_file)
                list_true = isinstance(data[list(data.keys())[0]][0],list)
            for eackDoc in data.keys():
                countDocs += 1
                seeddocids.append(eackDoc)
                #print("Type of seed: ", type(eackDoc))
                idl_recmnds = idlRecommendations(eackDoc)
                #print("idl rcmnds type: ", type(idl_recmnds[0]))
                idealRecmnds.append(idl_recmnds)
                print("do e have score? ", data[eackDoc][:10])
                sys.exit(0)
                if list_true:
                    gen_recmnds = [str(ea_[0]) for ea_ in data[eackDoc][:10]]
                else:
                    gen_recmnds = [str(ea_) for ea_ in data[eackDoc][:10]]
                #print("gen rcmnds type: ", type(gen_recmnds[0]))
                for eachGenRec in gen_recmnds:
                    writer.writerow(["seed", "recommendation", "similarity_score", 1])
                if eackDoc == gen_recmnds[0]:
                    countSeedCr += 1
                genRecmnds.append(gen_recmnds)
    print("Number of seeds with first idl rcmnd as seed: ", countSeedCr)
    p3, p5, r_, mrr_, ndcg_ = eval_metrics.main(idealRecmnds, genRecmnds)
    print(p3, p5, r_, mrr_, ndcg_)

# Example usage
if __name__ == "__main__":
    #file_path = "/beegfs/schubotz/ankit/code/zbReviewCit/stella_emb/stella_base/data/base_/extended_keywords_vs_extended_keywords/scores_/"  
    #file_path = "/beegfs/schubotz/ankit/code/zbReviewCit/stella_emb/stella_base/data/base_/title/scores_/"
    file_path = "/beegfs/schubotz/ankit/code/zbReviewCit/tf_idf_algrthmn/data/base_/title/"
    json_data = read_results_file(file_path)

