import sys
import json
import pickle
from random import shuffle
sys.path.append('/beegfs/schubotz/ankit/code/mabowdor')
import mabowdorScores

def getText(listOfdocs_, titles_):
    text_ = list()
    for eachDoc in listOfdocs_:
        if eachDoc in titles_:
            text_.append(" "+str(eachDoc)+" :"+titles_[eachDoc]+".")
    return text_

def getInstruction(seed_, potrec_):
    loc_title = "/beegfs/schubotz/ankit/code/SeqRecExp/zbDataReRanker/dataFeatures/"
    titls = pickle.load(open(loc_title+"initRank_featKwrd.pkl", 'rb'))
    seedText = getText([seed_], titls)[0]
    potentialdocText = getText(potrec_, titls)
    prmpt = "You are working as a Recommender System in scientific library with research papers in pure and applied mathematics. The library you are working with is zbMATHOpen. I will give you a seed document and a list of documents. You have to rank the documents from the list in ranked order considering first document in the ranked list as most likely recommendation and last as least. You can consider conceptual similarity, interlinked concepts to rank, etc. Along with document I also give assigned ID followed by document title. Seed document"
    prmpt += seedText
    prmpt += "Potential Recommendations:"
    prmpt += " ".join(potentialdocText)
    prmpt += " Please only output ranked list of IDs wihtout any text in the response and do not explain the output."
    return prmpt

def getTopRanked(seed_):
    """Get top 100 documents from initial ranker """
    locati_ = "/beegfs/schubotz/ankit/code/evaluation/hybridApproach/scores/top1krec/onlyPotRec/"
    keywrd = pickle.load(open(locati_+"title.pkl", 'rb'))#using titles as they have top recall
    return set(keywrd[seed_][:100])

def getRankedIdeal(seed_id, seed_idealrec):
    """Get ranked list for training data with ideal rec + rec from ranked titles """
    rankedList = list()
    for eachIdlrec in seed_idealrec[seed_id]:
        rankedList.append(eachIdlrec)
    get100top = getTopRanked(seed_id)
    for eachPotRec in get100top:
        if eachPotRec not in rankedList:
            rankedList.append(eachPotRec[0])
    return rankedList[:90]

def getOutput(seed_, idealRankedlist):
    """ Get ideal output prompt """
    prmpt_id = f"Following are the ranked recommendations for the seed id {seed_} : "
    for id_, eachIdlrec in enumerate(idealRankedlist):
        prmpt_id += f" Rank{id_+1}: {eachIdlrec}. "
    return prmpt_id

def creatDataFile():
    data_h = list()
    seedidlrec = mabowdorScores.getidealRecommendations()
    dataset_train = "/beegfs/schubotz/ankit/code/evaluation/hybridApproach/re-ranker/LibRerank/Data/zbmath/train_posandnegPairs.pkl"
    possamp, negsamp = pickle.load(open(dataset_train, 'rb'))
    dataset_eval = "/beegfs/schubotz/ankit/code/evaluation/hybridApproach/re-ranker/LibRerank/Data/zbmath/valid_posandnegPairs.pkl"
    possamp_, negsamp_ = pickle.load(open(dataset_eval, 'rb'))
    pos_seeds_ = set([eachEle[0] for eachEle in possamp_])
    #print(pos_seeds_)
    pos_seeds = set([eachEle[0] for eachEle in possamp])
    #print(pos_seeds)
    for id_,eachSeed in enumerate(pos_seeds.union(pos_seeds_)):
        try:
            top90rec = getRankedIdeal(eachSeed, seedidlrec)
            idl_output = getOutput(eachSeed, top90rec)
            shuffle(top90rec)
            instruct_ = getInstruction(eachSeed, top90rec)
            data_h.append({"instruction": instruct_,"input":"", "output": idl_output})
        except:
            print("did not work for seed: ", eachSeed)
    with open('zbPrompGen.json', 'w') as json_file:
        json.dump(data_h, json_file, indent=2)

creatDataFile()