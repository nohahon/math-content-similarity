import sys
import csv
import torch
import pickle
from transformers import AutoTokenizer, AutoModel
import os
from collections import defaultdict
csv.field_size_limit(100000000)

INSTRUCTIONS = {
    "qa": {
        "query": "Represent this query for retrieving relevant documents: ",
        "key": "Represent this document for retrieval: ",
    },
}


def _get_msc_dict(dictfile):
    msc_dict = {}
    with open(dictfile, "r", encoding="utf-8", errors="ignore") as csvfile:
        csvreader = csv.reader(csvfile,delimiter='\t')
        #next(csvreader)
        for row in csvreader:
            try:
                mscyear = int(row[-2][3:])
            except:
                mscyear = 0
            if mscyear in msc_dict.keys():
                try:
                    msc_dict[mscyear][row[0].upper()]=row[2]
                except:
                    print(row)
                    raise
            else:
                msc_dict[mscyear]={}
                try:
                    msc_dict[mscyear][row[0].upper()]=row[2]
                except:
                    print(row)
                    raise
    return msc_dict

def _msc_to_text(msc,msc_dict):
    mscyears = list(msc_dict.keys())
    mscyears.sort(reverse=True)
    for year in mscyears:
        if msc in msc_dict[year].keys():
            return msc+": "+msc_dict[year][msc]
        else:
            return msc

def _msc_row_to_text(mscs, msc_dict):
    mscs = mscs.upper().split()
    mscs = [msc if len(msc)==5 else  msc+("-XX")[len(msc)-2:] for msc in mscs]
    mscs = "; ".join([_msc_to_text(msc, msc_dict) for msc in mscs])
    return mscs


def getMSCs(filename,mscdictfile):
    """retrurns dict with key as zbMATH ID and value as MSCs"""
    seedrecs = getSEEDRecIds()
    msc_dict = _get_msc_dict(mscdictfile)

    dataWhole = dict()
    with open(filename, "r", encoding="utf-8", errors="ignore") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        i = 0
        for eachro in csvreader:
            mscs = eachro[1].upper().split()
            mscs = [msc if len(msc)==5 else  msc+("-XX")[len(msc)-2:] for msc in mscs]
            #print(mscs)
            try:
                mscs = "; ".join([_msc_to_text(msc, msc_dict) for msc in mscs])
                dataWhole[eachro[0]] = mscs
            except:
                i+=1
                print(i,eachro)
                raise

    return dataWhole
    
def getSEEDRecIds():
    """get seed IDS in a list"""
    listDocs = list()
    with open("./data/recommendationPairs.csv", mode ='r') as csvfile:
        csvFile = csv.reader(csvfile)
        for lines in csvFile:
            IdsandRec = list(filter(None, lines))
            listDocs += IdsandRec
    return listDocs

def getidealrecommendations():
    """get seed IDS in a list"""
    listDocs = dict()
    with open("./data/recommendationPairs.csv", mode ='r') as csvfile:
        csvFile = csv.reader(csvfile)
        for lines in csvFile:
            IdsandRec = list(filter(None, lines))
            listDocs[IdsandRec[0]]=IdsandRec[1:]
    return listDocs


def getSEEDIds():
    """get seed IDS in a list"""
    listDocs = list()
    with open("./data/recommendationPairs.csv", mode ='r') as csvfile:
        csvFile = csv.reader(csvfile)
        for lines in csvFile:
            IdsandRec = list(filter(None, lines))
            listDocs.append(IdsandRec[0])
    return listDocs

def getAlltitles(filename):
    """retrurns dict with key as zbMATH ID and value as keywords"""
    dataWhole = dict()
    with open(filename, "r", encoding="utf-8", errors="ignore") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for eachro in csvreader:
            dataWhole[eachro[1]] = eachro[0]
    return dataWhole

def genEmbeddingsBatch(batch_size):
    """ Saves cosine scores of seeds vs all candidate recommendations """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    alltitles = getMSCs("zbmath_abstracts.csv","msccode_to_text_detailed.csv")
    instruction = INSTRUCTIONS["qa"]
    print("executing2")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/llm-embedder")
    model = AutoModel.from_pretrained("BAAI/llm-embedder",device_map=device)
    docIDs = list()
    #Error!! Some seed is not in the abstracts file!
    queries = [
        instruction["query"] + alltitles[query] for query in getSEEDIds()
    ]
    query_inputs = tokenizer(
        queries,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    query_inputs.to(device)
    query_outputs = model(**query_inputs)
    query_embeddings = query_outputs.last_hidden_state[:, 0]
    query_embeddings = torch.nn.functional.normalize(
        query_embeddings,
        p=2,
        dim=1,
    )
    print("executing3")
    for i in range(0, len(alltitles) - 1, batch_size):
        if i<4772000:
            continue
        print("Doing for batch: ", i)
        keys = [
            instruction["key"] + alltitles[key]
            for key in list(alltitles.keys())[i : i + batch_size]
        ]
        docIDs += list(alltitles.keys())[i:i+batch_size]
        key_inputs = tokenizer(
            keys,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        key_inputs.to(device)
        with torch.no_grad():
            key_outputs = model(**key_inputs)
            key_embeddings = key_outputs.last_hidden_state[:, 0]
            key_embeddings = torch.nn.functional.normalize(
                key_embeddings,
                p=2,
                dim=1,
            )
        similarity = query_embeddings @ key_embeddings.T
        similarity = similarity.cpu().detach().numpy()
        similarity = [[tensor.item() for tensor in tensors] for tensors in similarity]
        with open("./data_ne/mscs/key_" + str(i) + "_.pkl", "wb") as f:
            pickle.dump(similarity, f)
        #if i>=10*batch_size:
        #    sys.exit(0)
    with open('./data_ne/docIDs_msc.pkl', 'wb') as fa:
        pickle.dump(docIDs,fa)

def createDictScores(dir_here):
    """ Combines all cosine scores to one pickle """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    getAllscores = sorted(os.listdir(dir_here),key=lambda x:int(x[4:-5]))
    allSeeds = getSEEDIds()
    seed_to_scores = defaultdict(lambda:list())
    for pick in getAllscores:
        with open(os.path.join(dir_here, pick), 'rb') as f:
            scores = pickle.load(f)
            #scores = [[tensor.item() for tensor in tensors] for tensors in scores]
            #scores.half()
            #scores = torch.load(f,map_location=device)
        for id_,ele in enumerate(scores):
            seed_to_scores[id_] += ele
    with open("./data_ne/docIDs_msc.pkl", 'rb') as fa:
        #docIds = torch.load(fa,map_location=device)
        docIds = pickle.load(fa)
    dictSeedRec = dict()
    for seed in seed_to_scores.keys():
        dictOfscores = dict()
        for id_h, eachScore in enumerate(seed_to_scores[seed]):
            dictOfscores[docIds[id_h]] = eachScore
        dictSeedRec[allSeeds[seed]] = dictOfscores
    sorted_dict = dict()
    for each_ in dictSeedRec.keys():
        sorted_dict[each_] = sorted(dictSeedRec[each_].items(), key=lambda x: x[1], reverse=True)
    with open('mscs_LLMemb.pkl', 'wb') as f:
        pickle.dump(sorted_dict, f)

def createResults(pickle_f1):
    """ Create resultfiles for evaluation """
    with open(pickle_f1, 'rb') as f:
        scores_1 = pickle.load(f)
    resultsdict = []
    seed_idlrecmnds = getidealrecommendations()
    #with jsonlines.open('rslts_keyowrds.jsonl', mode='w') as writer:
    for eachSeed in scores_1.keys():
        baselinercmnds = dict()
        for id_,pot_rcmnds in enumerate(scores_1[eachSeed][:15]):
            baselinercmnds[str(id_)] = [int(pot_rcmnds[0]), pot_rcmnds[1]]
        tempdict = {}
        tempdict["seed"] = eachSeed
        tempdict["idealRcmnds"] = seed_idlrecmnds[eachSeed]
        tempdict["baselineRcmnds"] = baselinercmnds
        resultsdict.append(tempdict)
    with open('mscs_results.pkl', 'wb') as f:
        pickle.dump(resultsdict, f)

def main():
    print("executing1")
    #dic = print(getMSCs("zbmath_abstracts.csv","msccode_to_text_detailed.csv"))
    genEmbeddingsBatch(1000) #generate cosine scores
    createDictScores("./data_ne/mscs/") # combine pickle files to single dict
    createResults("mscs_LLMemb.pkl") # get results file for evaluation

if __name__ == "__main__":
    main()
