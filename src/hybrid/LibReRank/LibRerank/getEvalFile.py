import os
import csv
import sys
import pickle
import jsonlines
from collections import defaultdict

def getidealRecommendations():
    """get Ideal Recmnds in a list"""
    listDocs = dict()
    with open("/beegfs/schubotz/ankit/data/recommendationPairs.csv", mode ='r') as csvfile:
        csvFile = csv.reader(csvfile)
        for lines in csvFile:
            IdsandRec = list(filter(None, lines))
            listDocs[IdsandRec[0]] = IdsandRec[1:]
    return listDocs

def getScoresinFile():
    resultsdict = {"seed": None, "idealRcmnds": None, "baselineRcmnds":None}
    with open("lambdaMART_pred.pkl", "rb") as f:
        scores_ = pickle.load(f)
    print("Len of keys: ", len(scores_.keys()))
    with jsonlines.open('rslts_lambdaMART.jsonl', mode='w') as writer:
        seed_idlrecmnds = getidealRecommendations()
        for each_ in scores_.keys():
            baselinercmnds = dict()
            baselinercmnds[str(0)] = [int(each_), 1.0]
            id_h = 0
            for id_,eachRcmnds in enumerate(scores_[each_]):
                if eachRcmnds[2] == 1.0:
                    id_h += 1
                    baselinercmnds[str(id_h)] = [int(eachRcmnds[0]),eachRcmnds[1] ]
            print("Seed Id: ", each_)
            print("baselinercmnds", baselinercmnds)
            resultsdict["seed"] = each_
            resultsdict["idealRcmnds"] = seed_idlrecmnds[each_]
            resultsdict["baselineRcmnds"] = baselinercmnds
            writer.write(resultsdict)

def getEvalScoresRanker():
    predictions_dir = "initial_ranker/lgb/pred_titles_lgb_lambdMART.pkl"
    seedrecpairs_dir = "initial_ranker/seed_potRec_titles.pkl"
    with open(predictions_dir, "rb") as prdf:
        predictions = pickle.load(prdf)
    with open(seedrecpairs_dir, "rb") as prda:
        seedrecpairs = pickle.load(prda)
    dictScores = defaultdict(lambda:list()) #for storing seed and their predictions
    print(len(predictions), len(seedrecpairs))
    for id_,eachPred in enumerate(predictions):
        dictScores[seedrecpairs[id_][0]].append([seedrecpairs[id_][1],eachPred])
    print(len(dictScores))
    dictSortedScores = dict()
    for eachSeed in dictScores.keys():
        sorted_list = sorted(dictScores[eachSeed], key=lambda x: x[1], reverse=True)
        dictSortedScores[eachSeed] = sorted_list
    #with open("evalRsl_titl_skl_lambdMART.pkl", "wb") as wrf:
    #    pickle.dump(dictSortedScores ,wrf)
    resultsdict = {"seed": None, "idealRcmnds": None, "baselineRcmnds":None}
    with jsonlines.open('rslts_lambdaMART_titles_lgb_1500.jsonl', mode='w') as writer:
        seed_idlrecmnds = getidealRecommendations()
        for each_ in dictSortedScores.keys():
            baselinercmnds = dict()
            baselinercmnds[str(0)] = [int(each_), 1.0]
            id_h = 0
            for id_,eachRcmnds in enumerate(dictSortedScores[each_][:1500]):
                id_h += 1
                baselinercmnds[str(id_h)] = [int(eachRcmnds[0]),eachRcmnds[1]]
            resultsdict["seed"] = each_
            resultsdict["idealRcmnds"] = seed_idlrecmnds[each_]
            resultsdict["baselineRcmnds"] = baselinercmnds
            writer.write(resultsdict)

getEvalScoresRanker()
#getScoresinFile()
