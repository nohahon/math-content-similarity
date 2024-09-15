import csv
import sys
import jsonlines
import pickle
import transformers
from transformers import AutoTokenizer
from transformers import pipeline
from collections import defaultdict
sys.path.append('/beegfs/schubotz/ankit/code/mabowdor')
import mabowdorScores
default_ = "Title does not exists"

def getaAllTitles():
    dataWhole = defaultdict()
    filename = '/beegfs/schubotz/ankit/data/zbMATH_titles.csv'
    with open(filename, 'r', encoding="utf-8", errors='ignore') as csvfile:
        csvreader = csv.reader(csvfile)
        first_row = next(csvreader)
        for eachro in csvreader:
            dataWhole[eachro[0]]= eachro[1]
    return dataWhole

def getFeatuResRe(seed_):
    """lod all top 1k recommendations and retrun seeds"""
    locati_ = "/beegfs/schubotz/ankit/code/evaluation/hybridApproach/scores/top1krec/onlyPotRec/"
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
    return abstr[seed_][:1000],citn_val,keywrd[seed_][:1000],math[seed_][:1000],msc[seed_][:1000],titls[seed_][:1000]

def getseedpotrec(dataset):
    # return dict with key as seed and val as set of pot recmnds
    possamp, negsamp = pickle.load(open(dataset, 'rb'))
    seedpotrec = dict()
    pos_seeds = set([eachEle[0] for eachEle in possamp])
    for eachSeed in pos_seeds:
        abs_, cits, kwr, mth, msc, ttle =  getFeatuResRe(eachSeed)
        candidateRec = set()
        for eachF in [abs_, cits, kwr, mth, msc, ttle]: #all features [abs_, cits, kwr, mth, msc, ttle]
            for eachEle in eachF:
                candidateRec.add(eachEle[0])
        seedpotrec[eachSeed] = candidateRec
    return seedpotrec

def gettrainDataprmpt():
    seedrecSeq = ""
    dir_feat = "/beegfs/schubotz/ankit/code/SeqRecExp/zbDataReRanker/dataFeatures/"
    dataset_train = "/beegfs/schubotz/ankit/code/evaluation/hybridApproach/re-ranker/LibRerank/Data/zbmath/train_posandnegPairs.pkl"
    titls = getaAllTitles()
    seedidlrec = mabowdorScores.getidealRecommendations()
    seedPotrec = getseedpotrec(dataset_train)
    for id_,eachSeed in enumerate(seedPotrec.keys()):
        seedrecSeq += f'Seed {id_+1}: {titls.get(eachSeed,default_)}.'
        for id__,eachIdealRec in enumerate(seedidlrec[eachSeed]):
            seedrecSeq += f" Recommendation {id__+1}: {titls.get(eachIdealRec,default_)}"
    return seedrecSeq

def gettestDataDict():
    dataset_test = "/beegfs/schubotz/ankit/code/evaluation/hybridApproach/re-ranker/LibRerank/Data/zbmath/test_posandnegPairs.pkl"
    seedPotrec = getseedpotrec(dataset_test)
    return seedPotrec

def testHere():
    titls = pickle.load(open("title_file.pkl", 'rb')) #getaAllTitles()
    print(titls)
    sys.exit(0)
    prmpStr = "You are working as a recommender system in the domain of pure and applied mathematics, i.e., with research articles from zbMATHopen. First, I provide you seed docuemnts and their ranked recommendations in order of relevance collected by an expert. Second, I will give you seed and a potential document, you will have to judge how likely this potential document can be a recommendation to the seed and assign a score between 0 and 1 with 1 being fully relevant and 0 being not at all. You can also choose intermdeiate values upto 4 decimal digits. I want to collect such scores for many documents so the more disrete values you give easier it gets to finally rank all potential recommendationsi. "
    traindataPrompt = gettrainDataprmpt()
    prmpStr += traindataPrompt
    testsdrec = {'1308161':['1356576','5083606', '4193896', '5007259','847806'],
                '1621343':['2709733', '1915690','1663809','1943338', '4099459']}
    for eachSeed in testsdrec.keys():#testdataPrmpt.keys()
        for eachPotrec in testsdrec[eachSeed]:#testdataPrmpt[eachSeed]
            prmpStr += f"Based on the previous seed and recommendationn please assign a score in between 0 and 1 to the follwoing seed and potential recommendation. Seed: {titls.get(eachSeed,default_)}. Potential recommendation: {titls.get(eachPotrec, default_)}."
            #reponse = llamaM(prmpStr)
            print(titls.get(eachSeed,default_))
            print(titls.get(eachPotrec, default_))
            #sys.exit(0)

def getPrompts():
    titls = pickle.load(open("title_file.pkl", 'rb')) #getaAllTitles()
    prmpStr = "You are working as a recommender system in the domain of pure and applied mathematics, i.e., with research articles from zbMATHopen. First, I provide you seed docuemnts and their ranked recommendations in order of relevance collected by an expert. Second, I will give you seed and a potential document, you will have to judge how likely this potential document can be a recommendation to the seed and assign a score between 0 and 1 with 1 being fully relevant and 0 being not at all. You can also choose intermdeiate values upto 4 decimal digits. I want to collect such scores for many documents so the more disrete values you give easier it gets to finally rank all potential recommendationsi. "
    traindataPrompt = gettrainDataprmpt()
    prmpStr += traindataPrompt
    resultsdict = {"seed": None, "pot_rec": None, "response":None}
    with jsonlines.open('test_outputs/latest_mammoth_textGen.jsonl', mode='w') as writer:
        for eachSeed in testdataPrmpt.keys():
            for eachPotrec in testdataPrmpt[eachSeed]:
                potrecstr = prmpStr
                potrecstr += f" Based on the previous seed and recommendationn please assign a score in between 0 and 1 to the follwoing seed and potential recommendation. Seed: {titls.get(eachSeed,default_)}. Potential recommendation: {titls.get(eachPotrec, default_)}."
                reponse = mammoth(potrecstr)
                resultsdict["seed"] = eachSeed
                resultsdict["pot_rec"] = eachPotrec
                resultsdict["response"] = reponse
                writer.write(resultsdict)

def chocolatine(msg_):
    messages = [{"role": "user", "content": msg_},]
    pipe = pipeline("text-generation", model="jpacifico/Chocolatine-14B-Instruct-4k-DPO", trust_remote_code=True, max_length=5024, device_map="auto")
    return pipe(messages)

def phiSmall(msg_):
    messages = [{"role": "user", "content": msg_},]
    pipe = pipeline("text-generation", model="microsoft/Phi-3-small-8k-instruct", trust_remote_code=True)
    return pipe(messages)

def llamaM(msg_):
    messages = [{"role": "user", "content": msg_},]
    pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct",device=0, max_length=40960)
    return pipe(messages)

def qwen(msg_):
    messages = [{"role": "user", "content": msg_},]
    pipe = pipeline("text-generation", model="MaziyarPanahi/calme-2.1-qwen2-7b", device=0)
    return pipe(messages)

def mammoth(msg_):
    messages = [
        {"role": "user", "content": msg_},
    ]
    pipe = pipeline("text-generation", model="TIGER-Lab/MAmmoTH2-7B-Plus", max_length=40960, device=0)
    return pipe(messages)

#testHere()
#qwen()
#mammoth()
#gettrainDataprmpt()
getPrompts()