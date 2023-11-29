import math
import numpy as np
import jsonlines

"""
"resultsFile" contains jsons with seed, idealRecmnds and generatedRcmnds
Example:
{"seed": "1371474",
"idealRcmnds": ["2067093", "6572608", "3545808", "6095689"],
"baselineRcmnds": {0: [1371474, 615.019], 1: [2066342, 331.019], 2: [2067093, 312.34818]}
"""


def P_R_F1_at_k(k, resultsFile):
    """
    Calculate precision, recall, and F1-score at k for a given results file.

    Parameters:
    k (int): The value of k for calculating precision and recall.
    resultsFile (str): The path to the results file.

    Returns:
    tuple: A tuple containing the mean precision, mean recall, and mean F1-score.
    """

    with jsonlines.open(resultsFile) as reader:
        recalls = []
        precisions = []
        f1s = []
        for line in reader:
            gold_recs = set(line["idealRcmnds"])
            if len(gold_recs) == 0:
                continue
            predicted = line["baselineRcmnds"]
            predicted = set(
                [str(rec[0]) for _, rec in predicted.items()][:k],
            )  # {0:[rec_id,similarity_score],1:[rec_id,similarity_score],..}

            hits = len(predicted.intersection(gold_recs))

            # not sure about this. If you hit 1 in k=3 and you have 10 gold recs you have recall .33
            # but if you hit 1 in k=10 with gold recs 10 you have recal .1
            # recall = hits / (len(gold_recs) if len(gold_recs) <= k else k)
            recall = hits / len(gold_recs)
            precision = hits / k
            f1 = (
                (2 * (precision * recall) / (precision + recall))
                if (precision + recall) > 0
                else 0
            )

            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)

        return np.mean(precisions), np.mean(recalls), np.mean(f1s)


def metric_R_at_n(n, resultsFile):
    """
    n : value of k in R@k
    resultsFile : see comment at the top
    """
    totalIdealrecomm = 0
    countofSeeds = 0
    with jsonlines.open(resultsFile) as reader:
        for obj in reader:
            toIdeal = len(obj["idealRcmnds"])
            if toIdeal > n:
                topn = obj["idealRcmnds"][:n]
            else:
                topn = obj["idealRcmnds"]
            if toIdeal < 1:
                # for safety
                print("No ideal recommendtions found: ", obj["seed"])
            else:
                countofSeeds += 1
                idealRecIn_n = 0
                for eachRank in obj["baselineRcmnds"].keys():
                    if eachRank == 0:
                        # ignore 0th rank as it must be the seed doc itself
                        continue
                    else:
                        if str(obj["baselineRcmnds"][eachRank][0]) in topn:
                            idealRecIn_n += 1
                totalIdealrecomm += idealRecIn_n / toIdeal
            # print(toIdeal, idealRecIn_n)
    return totalIdealrecomm / countofSeeds


def metric_P_at_n(n, resultsFile):
    """
    n : value of k in P@k
    resultsFile : see comment at the top
    """
    totalIdealrecomm = 0
    countofSeeds = 0
    with jsonlines.open(resultsFile) as reader:
        for obj in reader:
            toIdeal = len(obj["idealRcmnds"])
            if toIdeal > n:
                topn = obj["idealRcmnds"][:n]
            else:
                topn = obj["idealRcmnds"]
            if toIdeal < 1:
                # for safety
                print("No idea recommendtions found: ", obj["seed"])
            else:
                countofSeeds += 1
                idealRecIn_n = 0
                for eachRank in obj["baselineRcmnds"].keys():
                    if eachRank == 0:
                        continue
                    else:
                        if str(obj["baselineRcmnds"][eachRank][0]) in topn:
                            idealRecIn_n += 1
                totalIdealrecomm += idealRecIn_n / len(topn)
    return totalIdealrecomm / countofSeeds


def metric_nDCG(resultsFile):
    """
    # Test cases from wikipedia DCG article
    trueRel = [3,3,3,2,2,2]
    obtainedRel = [3,2,3,0,1,2]
    """
    totIdrec = 0
    finalnDCG = 0
    countofSeeds = 0
    with jsonlines.open(resultsFile) as reader:
        for obj in reader:
            dCGtrue = 0
            dCGbasel = 0
            toIdeal = len(obj["idealRcmnds"])
            totIdrec += toIdeal
            trueRel = list()
            for te, ele in enumerate(obj["idealRcmnds"]):
                trueRel.append(toIdeal - te)
            obtainedRel = list()
            for idealRec in obj["idealRcmnds"]:
                for key in obj["baselineRcmnds"].keys():
                    if int(idealRec) == obj["baselineRcmnds"][key][0]:
                        obtainedRel.append(toIdeal - int(key))
            revisedobtRel = list()
            for eachObt in obtainedRel:
                if eachObt < 0:
                    revisedobtRel.append(0)
                else:
                    revisedobtRel.append(eachObt)
            while len(revisedobtRel) < len(trueRel):
                revisedobtRel.append(0)
            for idh, rel in enumerate(trueRel):
                dCGtrue += rel / math.log(idh + 2, 2)
            for idhh, relh in enumerate(revisedobtRel):
                dCGbasel += relh / math.log(idhh + 2, 2)
            if dCGtrue == 0:
                print("Check size of your ideal recommendatiions")
            else:
                countofSeeds += 1
                finalnDCG += dCGbasel / dCGtrue
    print(finalnDCG / countofSeeds)


def metric_MRR(resultsFile):
    recipr_rank = 0
    countofSeeds = 0
    with jsonlines.open(resultsFile) as reader:
        for obj in reader:
            countofSeeds += 1
            for eachIdeal in obj["idealRcmnds"]:
                bslineIds = [
                    str(every[0]) for every in obj["baselineRcmnds"].values()
                ]
                if eachIdeal in bslineIds:
                    for eachRank in obj["baselineRcmnds"].keys():
                        if eachIdeal in obj["baselineRcmnds"][eachRank]:
                            recipr_rank += 1 / int(eachRank)
    return recipr_rank / countofSeeds


# print("metric_P_at_5: ", metric_P_at_n(5, resultsFilefilt))
