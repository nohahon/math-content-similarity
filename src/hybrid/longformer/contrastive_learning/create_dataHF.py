import sys
import pickle
from datasets import load_dataset, Dataset
sys.path.append('../../../longformer/')
import longformer_embed

def get_data(dataset):
    """return two lists first with samples and second with lables """
    possamp, negsamp = pickle.load(open(dataset, 'rb'))
    records, labels = [], []
    seedToidlrcmnds = longformer_embed.getIdealRec()
    features = longformer_embed.load_features() #need to get string
    pos_seeds = set([eachEle[0] for eachEle in possamp])
    for eachSeed in pos_seeds:
        seedStr = longformer_embed.getdocString(eachSeed, features)
        abs_, cits, kwr, mth, msc, ttle =  longformer_embed.getFeatuResRe(eachSeed)
        for id_,eachF in enumerate([abs_, cits, kwr, mth, msc, ttle]):
            for eachEle in eachF:
                potrecStr = longformer_embed.getdocString(eachEle[0], features)
                records.append(seedStr + potrecStr)
                if eachEle[0] in seedToidlrcmnds[eachSeed]:
                    labels.append(1)
                else:
                    labels.append(0)
    return records, labels

def dataon_hf():
    """ Put train test and validation data on huggingface """
    dataset_dir = "/beegfs/schubotz/ankit/code/evaluation/hybridApproach/re-ranker/LibRerank/Data/zbmath/"
    list_trainsamp, train_labl = get_data(dataset_dir+"train_posandnegPairs.pkl")
    data_dict = {"text": list_trainsamp, "label": train_labl}
    ds_ = Dataset.from_dict(data_dict)
    ds_.push_to_hub("AnkitSatpute/zbMathCombinedSamps", split="train")

    list_trainsamp, train_labl = get_data(dataset_dir+"test_posandnegPairs.pkl")
    data_dict = {"text": list_trainsamp, "label": train_labl}
    ds_ = Dataset.from_dict(data_dict)
    ds_.push_to_hub("AnkitSatpute/zbMathCombinedSamps", split="test")

    list_trainsamp, train_labl = get_data(dataset_dir+"valid_posandnegPairs.pkl")
    data_dict = {"text": list_trainsamp, "label": train_labl}
    ds_ = Dataset.from_dict(data_dict)
    ds_.push_to_hub("AnkitSatpute/zbMathCombinedSamps", split="validation")

dataon_hf()