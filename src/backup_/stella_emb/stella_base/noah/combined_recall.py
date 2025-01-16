import os
import sys
import json
import random
sys.path.append("../../../tf_idf_algrthmn")
import zbCitData_st
from collections import defaultdict

data_ = "/beegfs/schubotz/ankit/data/zbReviewCitData/citation_dataset.csv"
dataFr = zbCitData_st.load_csv_to_dataframe(data_)  # Load main dataset
train_, test_, valid_ = zbCitData_st.split_dataframe(dataFr)  # Split data into train, test, valid
#print(train_.shape[0] ,test_.shape[0], valid_.shape[0])
#sys.exit(0)

def recall_at_k(ideal_recommendations, generated_recommendations):
    """R@k measures how many of the relevant items are found in the top k recommendations."""
    recall_total = 0
    num_feats = int(len(list(generated_recommendations.values())[1])/10)
    print(num_feats)
    recall_total_sub = [0 for i in range(1,num_feats)]
    for seed in generated_recommendations.keys():
        ideal = ideal_recommendations[seed]
        generated  = generated_recommendations[seed]
        relevant_recs = {rec for rec in generated if rec in ideal}
        relevant_recs_sub = [{rec for rec in generated[10*k:10*(k+1)] if rec in ideal} for k in range(num_feats) ]
        relevant_recs_comp = [recs.difference(relevant_recs_sub[0]) for recs in relevant_recs_sub[1:]]
        relevant = len(relevant_recs)
        recall_total += relevant / len(ideal)
        for i in range(num_feats-1):
            recall_total_sub[i] += len(relevant_recs_comp[i]) / len(ideal)
    return (recall_total / len(generated_recommendations.keys()),
            [reca/len(generated_recommendations.keys()) for reca in recall_total_sub])


def idlRecommendations(doc_id):
    val_ret = test_[test_['document_id'] == int(doc_id)]['citation_de'].values
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

def read_results_file(file_path, features,k=10):
    genRecmnds, idealRecmnds = defaultdict(list), defaultdict(list)
    seeddocids = list()
    for feat in features:
        getallfls = os.listdir(os.path.join(file_path,feat,"scores_"))
        for i,eachF in enumerate(getallfls):
            with open(os.path.join(file_path,feat,"scores_",eachF), 'r') as json_file:
                # Load the content of the file into a Python dictionary
                data = json.load(json_file)
                try:
                    list_true = isinstance(data[list(data.keys())[0]][0],list)
                except:
                    list_true = isinstance(data[list(data.keys())[1]][0],list)
            for eackDoc in data.keys():
                seeddocids.append(eackDoc)
                idl_recmnds = idlRecommendations(eackDoc)
                idealRecmnds[eackDoc] = idl_recmnds
                if list_true:
                    gen_recmnds = [str(ea_[0]) for ea_ in data[eackDoc][:k]]
                else:
                    gen_recmnds = [str(ea_) for ea_ in data[eackDoc][:k]]
                genRecmnds[eackDoc] += gen_recmnds
    print(recall_at_k(idealRecmnds, genRecmnds))
    #rand_eles = random.sample(seeddocids, 3)
    #print(rand_eles)
    #getIdlandGenrecs(rand_eles)
    #sys.exit(0)
    #dictcits = defaultdict(lambda: 0)
    #for eachIdl in idl_recmnds:
    #    dictcits[len(eachIdl)] += 1
    #print("Distribution of citations: ", dictcits.keys())
    #print(dictcits)
    #print(len(genRecmnds), len(idealRecmnds))
    #sys.exit(0)
    #p3, p5, r_, mrr_, ndcg_ = eval_metrics.main(idealRecmnds, genRecmnds)
    #print(p3, p5, r_, mrr_, ndcg_)

# Example usage
if __name__ == "__main__":
    file_path = "/beegfs/schubotz/ankit/code/zbReviewCit/stella_emb/stella_base/data/base_/"
    features = ['text','text_vs_title','text_vs_mscs_in_text','text_vs_keyword']
    #features = ['text']
    json_data = read_results_file(file_path,features, k=10)

