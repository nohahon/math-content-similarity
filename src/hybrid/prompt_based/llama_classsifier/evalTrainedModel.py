import sys
import pickle
import torch
import jsonlines
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
sys.path.append('../../../longformer/')
import longformer_embed

#model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_name = "/beegfs/schubotz/ankit/code/evaluation/hybridApproach/re-ranker/promptbased/sentiment_analysis/saved_model_tit_test/"
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)
model.to("cuda:0")

def makePredictions(batch_):
    inputs = tokenizer(batch_, return_tensors="pt", padding=True)
    inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    lable = outputs['logits'].argmax()
    probs = torch.softmax(outputs['logits'], dim=-1).cpu().numpy()
    if lable == 0:
        return lable, probs[0][0]
    else:
        return lable, probs[0][1]

def testPredict():
    dataset_dir = "/beegfs/schubotz/ankit/code/evaluation/hybridApproach/re-ranker/LibRerank/Data/zbmath/"
    possamp, negsamp = pickle.load(open(dataset_dir+"test_posandnegPairs.pkl", 'rb'))
    features = longformer_embed.load_features()
    pos_seeds = set([eachEle[0] for eachEle in possamp])
    dictScores = defaultdict(lambda:list())
    for eachSeed in pos_seeds:
        seedStr = longformer_embed.getdocString(eachSeed, features)
        abs_, cits, kwr, mth, msc, ttle =  longformer_embed.getFeatuResRe(eachSeed)
        for id_,eachF in enumerate([abs_, cits, kwr, mth, msc, ttle]):
            for eachEle in eachF:
                potrecStr = longformer_embed.getdocString(eachEle[0], features)
                if potrecStr != "":
                    labels_, probs_ = makePredictions(seedStr+potrecStr)
                    dictScores[eachSeed].append([eachEle[0], labels_, probs_])
    normalDict = dict()
    for eachS in dictScores.keys():
        normalDict[eachS] = sorted(dictScores[eachS], key=lambda x: (x[1],x[2]), reverse=True)
    with open('llama_test_tit.pkl', 'wb') as file:
        pickle.dump(normalDict, file)

def createResultsFile():
    seed_toRec = pickle.load(open("llama_test_tit.pkl", 'rb'))
    seed_idlrecmnds = longformer_embed.getIdealRec()
    with jsonlines.open('rslts_llama_tit_test.jsonl', mode='w') as writer:
        resultsdict = {"seed": None, "idealRcmnds": None, "baselineRcmnds":None}
        for eachSeed in seed_toRec.keys():
            baselinercmnds = dict()
            baselinercmnds[str(0)] = [int(eachSeed), 1.0]
            id_h = 0
            for id_,eachRcmnds in enumerate(seed_toRec[eachSeed][:1500]):
                id_h += 1
                baselinercmnds[str(id_h)] = [int(eachRcmnds[0]),str(eachRcmnds[2])]
            resultsdict["seed"] = eachSeed
            resultsdict["idealRcmnds"] = seed_idlrecmnds[eachSeed]
            resultsdict["baselineRcmnds"] = baselinercmnds
            writer.write(resultsdict)

testPredict()
createResultsFile()