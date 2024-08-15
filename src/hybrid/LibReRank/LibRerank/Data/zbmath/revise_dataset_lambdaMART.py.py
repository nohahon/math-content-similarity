import csv
import sys
import pickle
import jsonlines

def getidealRecommendations():
    """
    get Ideal Recmnds in a list
    """
    listDocs = dict()
    #availablerec = read_and_print_csv_rows('/beegfs/schubotz/noah/arxMLiv/zbmath_abstracts.csv')
    with open("/beegfs/schubotz/ankit/data/recommendationPairs.csv", mode ='r') as csvfile:
        csvFile = csv.reader(csvfile)
        for lines in csvFile:
            IdsandRec = list(filter(None, lines))
            listDocs[IdsandRec[0]] = [idh for idh in IdsandRec[1:]]
    return listDocs

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

def cleaPoTRec():
    locati_ = "/beegfs/schubotz/ankit/code/evaluation/hybridApproach/scores/top1krec/onlyPotRec/"
    abstr = pickle.load(open(locati_+"title.pkl", 'rb'))
    revAbs = dict()
    for keyS in abstr.keys():
        listPotRec = list()
        for eachP in abstr[keyS]:
            if str(keyS) != eachP[0]:
                listPotRec.append(eachP)
        revAbs[keyS] = listPotRec
    with open("top1krec/onlyPotRec/title.pkl", 'wb') as wbf:
        pickle.dump(revAbs, wbf)

def resultFile():
    locati_ = "/beegfs/schubotz/ankit/code/evaluation/hybridApproach/scores/top1krec/"
    abstr = pickle.load(open(locati_+"abstract.pkl", 'rb'))
    seed_idlrecmnds = getidealRecommendations()
    resultsdict = {"seed": None, "idealRcmnds": None, "baselineRcmnds":None}
    with jsonlines.open('rslts_all_sort.jsonl', mode='w') as writer:
        for eachSeed in abstr.keys():
            baselinercmnds = dict()
            baselinercmnds["0"] = [int(eachSeed), 1.0]
            ab,cit,kw,mth,msc,tlts = getFeatuResRe(eachSeed)
            wholedict = ab+cit+kw+mth+msc+tlts
            whldict_print = sorted_list = sorted(wholedict[:11], key=lambda x: x[1], reverse=True)
            print("len here: ", len(whldict_print))
            for id_,pot_rcmnds in enumerate(whldict_print):
                baselinercmnds[str(id_)] = [int(pot_rcmnds[0]), pot_rcmnds[1]]
            resultsdict["seed"] = eachSeed
            resultsdict["idealRcmnds"] = seed_idlrecmnds[eachSeed]
            resultsdict["baselineRcmnds"] = baselinercmnds
            writer.write(resultsdict)

resultFile()


