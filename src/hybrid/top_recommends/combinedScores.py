import os
import sys
import csv
import pickle
import jsonlines
from collections import defaultdict
from msc_keyw_similarity import getidealRecommendations
csv.field_size_limit(100000000)

def checkCombinedCOmpleteness(pickle_f1, pickle_f2, pickle_f3):
    idRec = getidealRecommendations()
    with open(pickle_f1, 'rb') as f:
        scores_1 = pickle.load(f)
    with open(pickle_f2, 'rb') as f:
        scores_2 = pickle.load(f)
    with open(pickle_f3, 'rb') as f:
        scores_3 = pickle.load(f)
    totalIdeal = 0
    lenOfretrrec = 0
    for eaach in scores_1.keys():
        totalIdeal += len(idRec[eaach])
        idealRec = idRec[eaach]
        topIds_1 = [lst[0] for lst in scores_1[eaach][:10]]
        topIds_2 = [lst[0] for lst in scores_2[eaach][:10]]
        topIds_3 = [lst[0] for lst in scores_3[eaach][:10]]
        lenOfretrrec += len(set(topIds_1).union(set(topIds_2).union(set(topIds_3))).intersection(set(idealRec)))
    print("Total ideal recommendations: ",totalIdeal)
    print("Ideal recommendations in all test set: ", lenOfretrrec)

def createResults(pickle_f1, pickle_f2, pickle_f3):
    with open(pickle_f1, 'rb') as f:
        scores_1 = pickle.load(f)
    #print(scores_1.keys())
    with open(pickle_f2, 'rb') as f:
        scores_2 = pickle.load(f)
    with open(pickle_f3, 'rb') as f:
        scores_3 = pickle.load(f)
    resultsdict = {"seed": None, "idealRcmnds": None, "baselineRcmnds":None}
    seed_idlrecmnds = getidealRecommendations()
    with jsonlines.open('rslts_comBined.jsonl', mode='w') as writer:
        for eachSeed in scores_1.keys():
            baselinercmnds = dict()
            #print(scores_3[eachSeed][1:5])
            combined = [scores_1[eachSeed][0]]
            for is_,el in enumerate(scores_1[eachSeed][1:5]):
                combined += [scores_1[eachSeed][1:5][is_], scores_2[eachSeed][1:5][is_], scores_3[eachSeed][1:5][is_]]
            #print(len(combined))
            for id_,pot_rcmnds in enumerate(combined):
                baselinercmnds[str(id_)] = [int(pot_rcmnds[0]), pot_rcmnds[1]]
            resultsdict["seed"] = eachSeed
            resultsdict["idealRcmnds"] = seed_idlrecmnds[eachSeed]
            resultsdict["baselineRcmnds"] = baselinercmnds
            writer.write(resultsdict)

def checkCompleteness(pickle_file):
    #print(getidealRecommendations())
    idRec = getidealRecommendations()
    with open(pickle_file, 'rb') as f:
        scores = pickle.load(f)
    lenOfretrrec = 0
    totalIdeal = 0
    for eaach in scores.keys():
        topIds = [lst[0] for lst in scores[eaach][:1000]]
        idealRec = idRec[eaach]
        totalIdeal += len(idRec[eaach])
        lenOfretrrec += len(set(topIds).intersection(set(idealRec)))
    print("Total ideal recommendations: ",totalIdeal)
    print("Ideal recommendations in all test set: ", lenOfretrrec)

createResults("llembOrig_seed1.pkl", "mscSimil_scores.pkl","KeywordsSimil_scores.pkl")
#checkCompleteness("llembOrig_seed1.pkl")
#checkCompleteness("mscSimil_scores.pkl")
#checkCompleteness("KeywordsSimil_scores.pkl")
#checkCombinedCOmpleteness("llembOrig_seed1.pkl", "mscSimil_scores.pkl","KeywordsSimil_scores.pkl")