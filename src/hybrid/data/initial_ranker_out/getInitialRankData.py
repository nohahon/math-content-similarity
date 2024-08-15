import sys
import csv
import pickle
csv.field_size_limit(100000000)

def getFeatuResRe(seed_):
    """
    lod all top 1k recommendations and retrun seeds
    """
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
    return abstr[seed_][:1000],citn_val, keywrd[seed_][:1000], math[seed_][:1000], msc[seed_][:1000], titls[seed_][:1000]

def get_data_r(dataset_train, initial_len):
    possamp, negsamp = pickle.load(open(dataset_train, 'rb'))
    seedrecPrs = set()
    pos_seeds = set([eachEle[0] for eachEle in possamp])
    for eachSeed in pos_seeds:
        abs_, cits, kwr, mth, msc, ttle =  getFeatuResRe(eachSeed)
        for eachEle in [abs_, cits, kwr, mth, msc, ttle]:
            for eachIndvfeat in eachEle:
                seedrecPrs.add(eachIndvfeat[0])
    return seedrecPrs.union(initial_len.union(pos_seeds))

def countdocIDsunique():
    dataset_train = "/beegfs/schubotz/ankit/code/evaluation/hybridApproach/re-ranker/LibRerank/Data/zbmath/train_posandnegPairs.pkl"
    list_of_list_samps = get_data_r(dataset_train, set())
    dataset_test = "/beegfs/schubotz/ankit/code/evaluation/hybridApproach/re-ranker/LibRerank/Data/zbmath/test_posandnegPairs.pkl"
    list_of_list_samps_new = get_data_r(dataset_train, list_of_list_samps)
    return list_of_list_samps_new

def getKeyWordFeat(kwrdFile , un_ique_docs):#store id to keyword mappings
    dataWhole = dict()
    with open(kwrdFile, "r", encoding="utf-8", errors="ignore") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for eachro in csvreader:
            if eachro[1] in un_ique_docs:
                dataWhole[eachro[1]] = eachro[0]
    with open('initRank_featKwrd.pkl', 'wb') as fa:
        pickle.dump(dataWhole,fa)

def getTitleFeat(titleFile, un_ique_docs):#store id to keyword mappings
    dataWhole = dict()
    with open(titleFile, "r", encoding="utf-8", errors="ignore") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for eachro in csvreader:
            if eachro[0] in un_ique_docs:
                dataWhole[eachro[0]] = eachro[1]
    with open('initRank_featTitle.pkl', 'wb') as fa:
        pickle.dump(dataWhole,fa)

def getAbstrFeat(absFile, un_ique_docs):
    dataWhole = dict()
    with open(absFile, "r", encoding="utf-8", errors="ignore") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for eachro in csvreader:
            if eachro[0] in un_ique_docs:
                dataWhole[eachro[0]] = eachro[3]
    with open('initRank_featAbstr.pkl', 'wb') as fa:
        pickle.dump(dataWhole,fa)

def getDocID_to_zbl():
    dictRef = dict()
    with open("/beegfs/schubotz/ankit/data/zbMATH_id_to_ZBL.csv", mode ='r') as csvfile:
        csvFile = csv.reader(csvfile)
        first_row = next(csvFile)
        for lines in csvFile:
            dictRef[lines[0]] = lines[1]
    return dictRef

def getDocandRefstyle(file_loc):#get id to ref style matchings
    dictRef = dict()
    zbl_to_de = getDocID_to_zbl()
    with open(file_loc, 'r', encoding="utf-8", errors='ignore') as csvfile:
        csvreader = csv.reader(csvfile)
        first_row = next(csvreader)  # Read the first row
        for eachro in csvreader:
            if eachro[2] in zbl_to_de.keys():
                dictRef[zbl_to_de[eachro[2]]] = eachro[1]
    return dictRef

def getRefrnFeat(ref_fileloc, un_ique_docs):
    dataWhole = dict()
    idToRef = getDocandRefstyle(ref_fileloc)
    for eachDoc in idToRef.keys():
        if eachDoc in un_ique_docs:
            dataWhole[eachDoc] = idToRef[eachDoc]
    with open('initRank_refrncMSCs.pkl', 'wb') as fa:
        pickle.dump(dataWhole,fa)

def getMathfeat():
    pass

def getMscsFeat(mscCodeFile, mscCodeTotxt, un_ique_docs):
    mscCodetoTxt = dict()
    with open(mscCodeTotxt, "r", encoding="utf-8", errors="ignore") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for eachro in csvreader:
            mscCodetoTxt[eachro[0]] = eachro[1]
    dataWhole = dict()
    with open(mscCodeFile, "r", encoding="utf-8", errors="ignore") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for eachro in csvreader:
            if eachro[0] in un_ique_docs:
                msc_coeds = eachro[1].split(" ")
                mcstxtcodes = [mscCodetoTxt[k] for k in msc_coeds if k in mscCodetoTxt]
                mscs_string = ', '.join(mcstxtcodes)
                dataWhole[eachro[0]] = mscs_string
    with open('initRank_featMSCs.pkl', 'wb') as fa:
        pickle.dump(dataWhole,fa)

def main():
    uniqDocs = countdocIDsunique() #to get all unique docs from initial ranker stage
    print(len(uniqDocs))
    kwrdFile_zbmath = "/beegfs/schubotz/ankit/data/zbMATH_keywords.csv"
    getKeyWordFeat(kwrdFile_zbmath, uniqDocs) #save doc id to keywords mapping in a pickle
    titlesFile = "/beegfs/schubotz/ankit/data/zbMATH_titles.csv"
    getTitleFeat(titlesFile, uniqDocs) #save doc id to titles mappings
    abstrFile = "/beegfs/schubotz/noah/arxMLiv/zbmath_abstracts.csv"
    getAbstrFeat(abstrFile, uniqDocs)
    msccodeTotxt = "/beegfs/schubotz/ankit/data/msccode_to_text.csv"
    getMscsFeat(abstrFile, msccodeTotxt, uniqDocs)
    refidtoref = "/beegfs/schubotz/ankit/data/references_withIDs.csv"
    getRefrnFeat(refidtoref, uniqDocs)

main()