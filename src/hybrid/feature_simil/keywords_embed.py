import csv
import torch
import pickle
from transformers import AutoTokenizer, AutoModel
csv.field_size_limit(100000000)

INSTRUCTIONS = {
    "qa": {
        "query": "Represent this query for retrieving relevant documents: ",
        "key": "Represent this document for retrieval: ",
    },
}

def getSEEDIds():
    """get seed IDS in a list"""
    listDocs = list()
    with open("/data/recommendationPairs.csv", mode ='r') as csvfile:
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

def genEmbeddingsBatch():
    """ Saves cosine scores of seeds vs all candidate recommendations """
    alltitles = getAlltitles("data/zbMATH_keywords.csv")
    instruction = INSTRUCTIONS["qa"]
    tokenizer = AutoTokenizer.from_pretrained("BAAI/llm-embedder")
    model = AutoModel.from_pretrained("BAAI/llm-embedder")
    docIDs = list()
    queries = [
        instruction["query"] + alltitles[query] for query in getSEEDIds()
    ]
    query_inputs = tokenizer(
        queries,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    for i in range(0, len(alltitles) - 1, 5000):
        print("Doing for batch: ", i)
        keys = [
            instruction["key"] + alltitles[key]
            for key in list(alltitles.keys())[i : i + 5000]
        ]
        docIDs += list(alltitles.keys())[i:i+5000]
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
        with open("data_ne/keywords/key_" + str(i) + "_.pkl", "wb") as f:
            pickle.dump(similarity, f)
    with open('data_ne/docIDs.pkl', 'wb') as fa:
        pickle.dump(docIDs,fa)

def createDictScores(dir_here):
    """ Combines all cosine scores to one pickle """
    getAllscores = os.listdir(dir_here)
    allSeeds = getSEEDIds()
    seed_to_scores = defaultdict(lambda:list())
    for pick in getAllscores:
        with open(os.path.join(dir_here, pick), 'rb') as f:
            scores = pickle.load(f)
        for id_,ele in enumerate(scores):
            seed_to_scores[id_] += ele
    with open("data_ne/docIDs.pkl", 'rb') as fa:
        docIds = pickle.load(fa)
    dictSeedRec = dict()
    for seed in seed_to_scores.keys():
        dictOfscores = dict()
        for id_h, eachScore in enumerate(seed_to_scores[seed]):
            dictOfscores[docIds[id_h]] = eachScore.item()
        dictSeedRec[allSeeds[seed]] = dictOfscores
    sorted_dict = dict()
    for each_ in dictSeedRec.keys():
        sorted_dict[each_] = sorted(dictSeedRec[each_].items(), key=lambda x: x[1], reverse=True)
    with open('keywords_LLMemb.pkl', 'wb') as f:
        pickle.dump(sorted_dict, f)

def createResults(pickle_f1):
    """ Create resultfiles for evaluation """
    with open(pickle_f1, 'rb') as f:
        scores_1 = pickle.load(f)
    resultsdict = {"seed": None, "idealRcmnds": None, "baselineRcmnds":None}
    seed_idlrecmnds = getidealrecommendations()
    with jsonlines.open('rslts_keyowrds.jsonl', mode='w') as writer:
        for eachSeed in scores_1.keys():
            baselinercmnds = dict()
            for id_,pot_rcmnds in enumerate(scores_1[eachSeed][:15]):
                baselinercmnds[str(id_)] = [int(pot_rcmnds[0]), pot_rcmnds[1]]
            resultsdict["seed"] = eachSeed
            resultsdict["idealRcmnds"] = seed_idlrecmnds[eachSeed]
            resultsdict["baselineRcmnds"] = baselinercmnds
            writer.write(resultsdict)

def main():
    genEmbeddingsBatch() #generate cosine scores
    createDictScores("/data_ne/keywords/") # combine pickle files to single dict
    createResults("keywords_LLMemb.pkl") # get results file for evaluation

if __name__ == "__main__":
    main()