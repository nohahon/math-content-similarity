import sys
import pickle
import math
import numpy as np
import jsonlines

def getTestData():
    dataset = "/beegfs/schubotz/ankit/code/evaluation/hybridApproach/re-ranker/LibRerank/Data/zbmath/test_posandnegPairs.pkl"
    possamp, negsamp = pickle.load(open(dataset, 'rb'))
    return set([ele[0] for ele in possamp])

def R_at_n(n, resultsFile):
    """
    n : value of k in P@k
    ToDos: replace baselineRcmnds wit generatedRec
    """
    tst_samps = getTestData()
    with jsonlines.open(resultsFile) as reader:
        allseeds_R = 0 #will add individual recall values here
        for obj in reader:
            if str(obj['seed']) in tst_samps: #only consider s seed is part of test set
                get_K_generatedRec = list() # get top k generated recmnds
                for intr, eachRank in enumerate(obj["baselineRcmnds"].keys()): #loping thrgh gentd recmnds
                    if intr == 0:
                        if obj["baselineRcmnds"][eachRank][0] != int(obj["seed"]): # if its not then we consider it
                            get_K_generatedRec.append(obj["baselineRcmnds"][eachRank][0])
                    elif intr > n:
                        break
                    else:
                        get_K_generatedRec.append(obj["baselineRcmnds"][eachRank][0])
                count = 0 # count ideal rcmnds in k generated rcmnds
                count += sum(1 for eachGenRec in get_K_generatedRec if str(eachGenRec) in obj["idealRcmnds"])
                allseeds_R += count/len(obj["idealRcmnds"])#R = ideal recmnds in k/total idl rcmnds
    return allseeds_R/len(tst_samps)

def P_at_n(n, resultsFile):
    """n : value of k in P@k
    ToDos: replace baselineRcmnds wit generatedRec"""
    tst_samps= getTestData()
    with jsonlines.open(resultsFile) as reader:
        allseeds_P = 0 #recallfortotseeds = 0
        for obj in reader:
            if str(obj['seed']) in tst_samps:
                get_K_generatedRec = list() # get top k generated rec
                totalRel_items = len(obj["idealRcmnds"])
                for intr, eachRank in enumerate(obj["baselineRcmnds"].keys()):
                    if intr == 0: # In most cases seeddoc is part of cand rcmnds so it is first expcted rcmnd
                        if obj["baselineRcmnds"][eachRank][0] != int(obj["seed"]): # if its not then we consider it
                            get_K_generatedRec.append(obj["baselineRcmnds"][eachRank][0])
                    elif intr > n:
                        break
                    else:
                        get_K_generatedRec.append(obj["baselineRcmnds"][eachRank][0])
                count = 0 # count nr of gen rec in ideal rec
                count += sum(1 for eachGenRec in get_K_generatedRec if str(eachGenRec) in obj["idealRcmnds"])
                allseeds_P += count/n
    return allseeds_P/len(tst_samps)

def get_nDCG_real(resultsFile):
    score = 0
    tst_samps= getTestData()
    with jsonlines.open(resultsFile) as reader:
        for obj in reader:
            if str(obj['seed']) in tst_samps:
                bslineIds = [str(every[0]) for every in list(obj["baselineRcmnds"].values())[1:11]]
                idealRcmnds = obj["idealRcmnds"]
                score += ndcg(idealRcmnds, bslineIds, len(idealRcmnds))
    return score/len(tst_samps)

def calculate_relevance_scores(ideal_list, retrieved_list):
    """ Generate relevance scores for retrieved_list based on their position in ideal_list """
    ideal_ranking_dict = {doc: len(ideal_list) - idx for idx, doc in enumerate(ideal_list)}
    return [ideal_ranking_dict.get(doc, 0) for doc in retrieved_list]

def dcg(relevances, k=None):
    """ Compute the Discounted Cumulative Gain. """
    relevances = relevances[:k]
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))

def ndcg(ideal_list, retrieved_list, k=None):
    """ Compute the Normalized Discounted Cumulative Gain. """
    ideal_relevances = calculate_relevance_scores(ideal_list, ideal_list)
    retrieved_relevances = calculate_relevance_scores(ideal_list, retrieved_list)
    idcg = dcg(ideal_relevances, k)
    if idcg == 0:
        return 0
    return dcg(retrieved_relevances, k) / idcg

def getF1(prec, rec):
    """ Compute F1 score from precision and recall. """
    return 2 * (prec * rec) / (prec + rec)

def metric_MRR(resultsFile):
    """ Compute Mean Retrieval Rank for all seeds. """
    recipr_rank = 0 # add reciprocal rank of each seed
    tst_samps= getTestData() # get test seeds
    with jsonlines.open(resultsFile) as reader:
        for obj in reader:
            if str(obj['seed']) in tst_samps: # only calculate rank if its test doc seed
                bslineIds = [str(every[0]) for every in list(obj["baselineRcmnds"].values())[1:11]] #get top 10 gen rcmnds
                for eachbsrc in bslineIds:
                    if eachbsrc in obj["idealRcmnds"]:
                        rank = bslineIds.index(eachbsrc)+1
                        recipr_rank += 1/rank # calculate reciprocl rank
                        break # only have to get first ideal rcmnd in generated rcmnds
    return recipr_rank/len(tst_samps) # divided by total queries

def getAll_scores(scoresFor, resultsFile):
    print("Evaluation scores for: ",scoresFor)
    for k in [3,5,10]:
        prec = P_at_n(k, resultsFile)
        rec = R_at_n(k, resultsFile)
        print(f'Evaluation scores : Precision: {prec} Recall: {rec} F1 {getF1(prec, rec)}')
    print(f'MRR: {metric_MRR(resultsFile)} N´nDCG: {get_nDCG_real(resultsFile)}')

resultsFile = "/beegfs/schubotz/ankit/code/evaluation/hybridApproach/re-ranker/mpnn/outputData/rslts_mpnn_15_10_hdl.jsonl"
getAll_scores("kwrds: ", resultsFile)