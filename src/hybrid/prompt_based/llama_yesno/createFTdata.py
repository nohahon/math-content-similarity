import sys
import json
import pickle
sys.path.append('../../../longformer/')
import longformer_embed

def getdocString(doc_id, features):
    """given docID and feature return One Hot String """
    oneHotstr = ""
    for key in features.keys():
        feature_dict = features[key]
        if doc_id in feature_dict:
            oneHotstr += " "+str(key)+": "+ feature_dict[doc_id]+"."
            oneHotstr += " "
    return oneHotstr

def getPrompt():
    prmpStr = "You are working as a recommender system in pure and applied mathematics, i.e., with research articles from zbMATHOpen. I provide you with a Seed document and a Candidate document. Both documents have features such as title, abstract, keywords, and MSCs (Math Subject Classification codes describing what topics are contained in a document). However, it is possible that a feature for either seed or candidate document might be missing. You must answer only Yes or No to determine whether the candidate document can act as a recommendation for the seed document. "
    return prmpStr

def getRandRec(seed_id, ideal_rec):
    """Return random recs as size of ideal rec that are nit ideal rec """
    locati_ = "/beegfs/schubotz/ankit/code/evaluation/hybridApproach/scores/top1krec/onlyPotRec/"
    abstr = pickle.load(open(locati_+"abstract.pkl", 'rb'))
    cand_ = abstr[seed_id][10000:11000]
    cand_rev = [itm[0] for itm in cand_ if itm[0] not in ideal_rec]
    return cand_rev[:len(ideal_rec)]

def creatDataFile():
    data_h = list()
    initPrmpt = getPrompt()
    endPrompt = " Please answer only in either Yes or No whether the candidate document can act as recommendation for the seed docuemnt. Do not explain your answer."
    feat_ = longformer_embed.load_features()
    seedidlrec = longformer_embed.getIdealRec()
    dataset_train = "/beegfs/schubotz/ankit/code/evaluation/hybridApproach/re-ranker/LibRerank/Data/zbmath/train_posandnegPairs.pkl"
    possamp, negsamp = pickle.load(open(dataset_train, 'rb'))
    dataset_eval = "/beegfs/schubotz/ankit/code/evaluation/hybridApproach/re-ranker/LibRerank/Data/zbmath/valid_posandnegPairs.pkl"
    possamp_, negsamp_ = pickle.load(open(dataset_eval, 'rb'))
    pos_seeds_ = set([eachEle[0] for eachEle in possamp_])
    pos_seeds = set([eachEle[0] for eachEle in possamp])
    for id_,eachSeed in enumerate(pos_seeds.union(pos_seeds_)):
        seedText = "Seed document: ["+getdocString(eachSeed, feat_)+"]."
        if seedText != "":
            seed_idlRec = seedidlrec[eachSeed]
            seed_randRec = getRandRec(eachSeed, seed_idlRec)
            for idlrec in seed_idlRec:
                idlrecText = "Candidate document: ["+getdocString(idlrec, feat_)+" ]."
                if idlrecText != "":
                    idl_output = "Yes"
                    data_h.append({"instruction": initPrmpt+seedText+idlrecText+endPrompt,"input":"", "output": idl_output})

            for idlrec in seed_randRec:
                randRecText = "Candidate document: ["+getdocString(idlrec, feat_)+" ]."
                if randRecText != "":
                    idl_output = "No"
                    data_h.append({"instruction": initPrmpt+seedText+randRecText+endPrompt,"input":"", "output": idl_output})
    with open('zbPrompGen_forlogit.json', 'w') as json_file:
        json.dump(data_h, json_file, indent=2)

creatDataFile()