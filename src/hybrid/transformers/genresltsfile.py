import sys
import pickle
import jsonlines
sys.path.append('/beegfs/schubotz/ankit/code/mabowdor')
import mabowdorScores

def getResultsFile():
    predctns = pickle.load(open("saved_outputf/pred_scores_tfrerank.pkl", 'rb'))
    print(len(predctns))
    resultsdict = {"seed": None, "idealRcmnds": None, "baselineRcmnds":None}
    with jsonlines.open('rslts_BERT_rerank.jsonl', mode='w') as writer:
        seed_idlrecmnds = mabowdorScores.getidealRecommendations()
        for eachK in predctns.keys():
            baselinercmnds = dict()
            baselinercmnds[str(0)] = [int(eachK), 1.0]
            dictOfposneg = {'POSITIVE': [], 'NEGATIVE': []}
            for eachInner in predctns[eachK]:
                dictOfposneg[eachInner[1]].append([eachInner[0], eachInner[2]])
            print(len(dictOfposneg['POSITIVE']))
            sort_lst = sorted(dictOfposneg['NEGATIVE'], key = lambda x: x[1])
            id_h = 0
            for id_,eachRcmnds in enumerate(sort_lst[:1500]):
                id_h += 1
                baselinercmnds[str(id_h)] = [int(eachRcmnds[0]),eachRcmnds[1]]
            resultsdict["seed"] = eachK
            resultsdict["idealRcmnds"] = seed_idlrecmnds[eachK]
            resultsdict["baselineRcmnds"] = baselinercmnds
            #print(resultsdict)
            writer.write(resultsdict)
            #sys.exit(0)

getResultsFile()
