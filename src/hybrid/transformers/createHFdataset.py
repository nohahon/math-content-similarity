import sys
import pickle
from datasets import Dataset
sys.path.append('/mabowdor')
import mabowdorScores

def getFeatuResRe(seed_):
    """
    lod all top 1k recommendations and retrun seeds
    """
    locati_ = "evaluation/hybridApproach/scores/top1krec/onlyPotRec/"
    abstr = pickle.load(open(locati_+"abstract.pkl", 'rb'))
    citn = pickle.load(open(locati_+"cits_withstdvl.pkl", 'rb'))
    keywrd = pickle.load(open(locati_+"kywrd_withstdvl.pkl", 'rb'))
    math = pickle.load(open(locati_+"math_withstdvl.pkl", 'rb'))
    msc = pickle.load(open(locati_+"msc_withstdvl.pkl", 'rb'))
    titls = pickle.load(open(locati_+"title.pkl", 'rb'))
    if seed_ in citn.keys():
        citn_val = citn[seed_][:1000]
    else:
        citn_val = []
    return abstr[seed_][:1000],citn_val, keywrd[seed_][:1000], math[seed_][:1000], msc[seed_][:1000], titls[seed_][:1000]

def get_data_r(dataset):
    # getting re-ranker training data and testing data
    possamp, negsamp = pickle.load(open(dataset, 'rb'))
    records, labels = [], []
    seedToidlrcmnds = mabowdorScores.getidealRecommendations()
    seedrecPrs = list()
    pos_seeds = set([eachEle[0] for eachEle in possamp])
    for eachSeed in pos_seeds:
        abs_, cits, kwr, mth, msc, ttle =  getFeatuResRe(eachSeed)
        #for id_,eachF in enumerate([abs_, kwr, mth, msc, ttle]):
        for id_,eachF in enumerate([abs_, cits, kwr, mth, msc, ttle]):
            for eachEle in eachF:
                if eachEle[0] in seedToidlrcmnds[eachSeed]:
                    #records.append(str([eachEle[1],id_]))
                    records.append("A data point with a siilarity score of "+str(eachEle[1])+", and feature ID "+str(id_)+".")
                    labels.append(1)
                    seedrecPrs.append([eachSeed,eachEle[0]])
                else:
                    #records.append(str([eachEle[1],id_]))
                    records.append("A data point with a siilarity score of "+str(eachEle[1])+", and feature ID "+str(id_)+".")
                    labels.append(0)
                    seedrecPrs.append([eachSeed,eachEle[0]])
    with open("BERT_rerankdata_textformat.pkl", "wb") as wpf:
        pickle.dump(seedrecPrs, wpf)
    return records, labels

def createTFdata():
    maindict = dict()
    #maindict = {"text": ["tt", "ttn"], "label": [0, 1]}
    #maindict_ = {"text": ["aa", "aan"], "label": [1, 0]}
    #dataset_train = "/beegfs/schubotz/ankit/code/evaluation/hybridApproach/re-ranker/LibRerank/Data/zbmath/train_posandnegPairs.pkl"
    #list_of_list_samps, labels = get_data_r(dataset_train)
    #maindict = {"text": list_of_list_samps, "label": labels}
    dataset_test = "/evaluation/hybridApproach/re-ranker/LibRerank/Data/zbmath/test_posandnegPairs.pkl"
    list_of_test, labels_test = get_data_r(dataset_test)
    maindict_ = {"text": list_of_test, "label": labels_test}
    sys.exit(0)

    dataset_valid = "/evaluation/hybridApproach/re-ranker/LibRerank/Data/zbmath/valid_posandnegPairs.pkl"
    list_of_vlid, labels_vlid = get_data_r(dataset_valid)
    maindict__ = {"text": list_of_vlid, "label": labels_vlid}
    #print(len(maindict))
    #print(len(maindict["train"]["text"]), len(maindict["train"]["label"]))
    #print(len(maindict["test"]["text"]), len(maindict["test"]["label"]))
    ds_ = Dataset.from_dict(maindict)
    ds_.push_to_hub("zbm_top1000_ttv_textformat", split="train")
    ds__ = Dataset.from_dict(maindict_)
    ds__.push_to_hub("zbm_top1000_ttv_textformat", split="test")
    ds___ = Dataset.from_dict(maindict__)
    ds___.push_to_hub("zbm_top1000_ttv_textformat", split="validation")

createTFdata()
