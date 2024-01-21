import os
import sys
import csv
import torch
import random
import itertools
import pickle
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
csv.field_size_limit(100000000)

def getidealrecommendations():
    listDocs = dict()
    with open("/beegfs/schubotz/ankit/data/recommendationPairs.csv", mode ='r') as csvfile:
        csvFile = csv.reader(csvfile)
        for lines in csvFile:
            IdsandRec = list(filter(None, lines))
            listDocs[IdsandRec[0]]= IdsandRec[1:]
    return listDocs

def getSEEDIds():
    """
    get seed IDS in a list
    """
    listDocs = list()
    with open("/beegfs/schubotz/ankit/data/recommendationPairs.csv", mode ='r') as csvfile:
        csvFile = csv.reader(csvfile)
        for lines in csvFile:
            IdsandRec = list(filter(None, lines))
            listDocs.append(IdsandRec[0])
    return listDocs

def getAlltitles(filename):
    """ retrurns dict with key as zbMATH ID and value as titles"""
    dataWhole = dict()
    # impIDs = getIDs41() #Only for 14
    with open(filename, 'r', encoding="utf-8", errors='ignore') as csvfile:
        csvreader = csv.reader(csvfile)
        first_row = next(csvreader)
        for eachro in csvreader:
            dataWhole[eachro[0]]= eachro[1]
    return  dataWhole

def readRankedScores():
    pickle_dir = "/beegfs/schubotz/ankit/code/evaluation/hybridApproach/scores/titles_LLMemb.pkl"
    idealRecs = getidealrecommendations()
    posvePairs = list()
    negvePairs = list()
    with open(pickle_dir, 'rb') as f:
        scores = pickle.load(f)    
    for eaach in scores.keys():
        localPos = list()
        localNeg = list()
        for eachDet in scores[eaach][1:]:
            if eachDet[0] in idealRecs[eaach]:
                localPos.append((eaach,eachDet[0]))
            else:
                if len(localNeg) < 20:
                    localNeg.append((eaach,eachDet[0]))
        posvePairs += localPos
        negvePairs += localNeg
        #if len(posvePairs) > 160:
        #    break
    return posvePairs,negvePairs

def genEMbeddings():
    pvePrs, nvPrs = readRankedScores()
    with open("train_posandnegPairs.pkl", 'wb') as f:
        pickle.dump([pvePrs[:156], nvPrs[:640]],f)
    with open("test_posandnegPairs.pkl", 'wb') as f:
        pickle.dump([pvePrs[156:312], nvPrs[640:1280]],f)
    with open("valid_posandnegPairs.pkl", 'wb') as f:
        pickle.dump([pvePrs[312:], nvPrs[1280:]],f)
    allTitles = getAlltitles("/beegfs/schubotz/ankit/data/zbMATH_titles.csv")
    tokenizer = AutoTokenizer.from_pretrained('BAAI/llm-embedder')
    model = AutoModel.from_pretrained('BAAI/llm-embedder')
    refSet = set()
    for eachEle in pvePrs:
        refSet.add(eachEle[0])
        refSet.add(eachEle[1])
    for eachEle in nvPrs:
        refSet.add(eachEle[0])
        refSet.add(eachEle[1])
    refSet = list(refSet)
    queries = [allTitles[eachK] for eachK in refSet]
    query_inputs = tokenizer(queries, padding=True,truncation=True, return_tensors='pt')
    with torch.no_grad():
        query_outputs = model(**query_inputs)
        query_embeddings = query_outputs.last_hidden_state[:, 0]
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        print(len(query_embeddings))
    dictFinalEMbed = dict()
    for id_,eachEl in enumerate(query_embeddings):
        dictFinalEMbed[refSet[id_]] = eachEl.tolist()
    with open("seedToembed.pkl", 'wb') as f:
        pickle.dump(dictFinalEMbed,f)

genEMbeddings()
