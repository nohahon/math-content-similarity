import os
import csv
import torch
import random
import jsonlines
import pickle
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel

INSTRUCTIONS = {
    "qa": {
        "query": "Represent this query for retrieving relevant documents: ",
        "key": "Represent this document for retrieval: ",
    },
}


def createDocIdAbPairs(filename):
    datadict = dict()
    with open(filename, "r", encoding="utf-8", errors="ignore") as csvfile:
        csvreader = csv.reader(csvfile)
        first_row = next(csvreader)  # Read the first row
        for eachro in csvreader:
            datadict[eachro[0]] = eachro[1]
    return datadict


def genEmbeddingsBatch():
    alldocs = createDocIdAbPairs(
        "citeulike-a/raw-data.csv",
    )
    instruction = INSTRUCTIONS["qa"]
    tokenizer = AutoTokenizer.from_pretrained("BAAI/llm-embedder")
    model = AutoModel.from_pretrained("BAAI/llm-embedder")
    for i_ in range(0, len(alldocs), 1000):
        queries = [
            instruction["query"] + alldocs[query]
            for query in list(alldocs.keys())[i_ : i_ + 1000]
        ]
        query_inputs = tokenizer(
            queries,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            query_outputs = model(**query_inputs)
            query_embeddings = query_outputs.last_hidden_state[:, 0]
            query_embeddings = torch.nn.functional.normalize(
                query_embeddings,
                p=2,
                dim=1,
            )
        with open("data/seed_" + str(i_) + "_.pkl", "wb") as f:
            pickle.dump(query_embeddings, f)


def calculateSimilarity(dir_storeddata):
    alldocs = createDocIdAbPairs(
        "citeulike-a/raw-data.csv",
    )
    os.listdir(dir_storeddata)
    recordScores = defaultdict(lambda: list())
    for i_ in range(0, len(alldocs), 1000):
        with open("data/seed_" + str(i_) + "_.pkl", "rb") as f:
            scores = pickle.load(f)
        for j_ in range(0, len(alldocs), 1000):
            with open("data/seed_" + str(j_) + "_.pkl", "rb") as f:
                scores_ = pickle.load(f)
            similarity_ = scores @ scores_.T
            for score in similarity_:
                recordScores[i_].append([it.item() for it in score])
    with open("abSimil_cUL_scores.pkl", "wb") as f:
        pickle.dump(dict(recordScores), f)


def getUserPairs(filename):
    userInterest = defaultdict(lambda: list())
    userCount = 0
    with open(filename, "r", encoding="utf-8", errors="ignore") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for eachro in csvreader:
            userCount += 1
            for eachInt in eachro[0].split()[1:]:
                userInterest[int(eachInt) + 1].append(userCount)
    userCombs = defaultdict(lambda: 0)
    for eachke in userInterest.keys():
        if len(userInterest[eachke]) > 3:
            userCombs[tuple(set(userInterest[eachke]))] += 1
    listOfComvs = list()
    for eachK in userCombs.keys():
        if userCombs[eachK] > 2:
            listOfComvs.append(eachK)
    seedIds = list()
    recomendsID = set()
    seed_and_rcmnds = dict()
    for eachCombi in listOfComvs:
        sd = random.choice(eachCombi)
        rcmnds = set()
        seedIds.append(sd)
        for eachP in eachCombi:
            rcmnds.add(eachP)
            recomendsID.add(eachP)
        seed_and_rcmnds[sd] = rcmnds
    instruction = INSTRUCTIONS["qa"]
    alldocs = createDocIdAbPairs(
        "citeulike-a/raw-data.csv",
    )
    tokenizer = AutoTokenizer.from_pretrained("BAAI/llm-embedder")
    model = AutoModel.from_pretrained("BAAI/llm-embedder")
    queries = [instruction["query"] + alldocs[str(query)] for query in seedIds]
    qu_ = [query for query in seedIds]
    query_inputs = tokenizer(
        queries,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    keys = [instruction["key"] + alldocs[str(key)] for key in recomendsID]
    ke_ = [key for key in recomendsID]
    key_inputs = tokenizer(
        keys,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        query_outputs = model(**query_inputs)
        key_outputs = model(**key_inputs)
        query_embeddings = query_outputs.last_hidden_state[:, 0]
        key_embeddings = key_outputs.last_hidden_state[:, 0]
        query_embeddings = torch.nn.functional.normalize(
            query_embeddings,
            p=2,
            dim=1,
        )
        key_embeddings = torch.nn.functional.normalize(
            key_embeddings,
            p=2,
            dim=1,
        )
    similarity = query_embeddings @ key_embeddings.T
    dictScores = defaultdict(lambda: dict())
    for id_, eachSim in enumerate(similarity):
        for id_h, eachSc in enumerate(eachSim):
            dictScores[qu_[id_]][ke_[id_h]] = eachSc.item()
    sorted_dict = dict()
    for each_ in dictScores.keys():
        sorted_dict[each_] = sorted(
            dictScores[each_].items(),
            key=lambda x: x[1],
            reverse=True,
        )
    with open("citeUlik_title.pkl", "wb") as f:
        pickle.dump(sorted_dict, f)
    return seed_and_rcmnds


def getTopScores():
    seedandrsmnds = getUserPairs(
        "citeulike-a/users.dat",
    )
    with open("citeUlik_title.pkl", "rb") as f:
        scores = pickle.load(f)
    resultsdict = {"seed": None, "idealRcmnds": None, "baselineRcmnds": None}
    with jsonlines.open("rslts_cituLike.jsonl", mode="w") as writer:
        for eachSeed in scores.keys():
            baselinercmnds = dict()
            for id_, pot_rcmnds in enumerate(scores[eachSeed][:11]):
                baselinercmnds[str(id_)] = [int(pot_rcmnds[0]), pot_rcmnds[1]]
            resultsdict["seed"] = str(eachSeed)
            resultsdict["idealRcmnds"] = [
                str(ele) for ele in seedandrsmnds[eachSeed]
            ]
            resultsdict["baselineRcmnds"] = baselinercmnds
            writer.write(resultsdict)
