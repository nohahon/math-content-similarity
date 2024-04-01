import math
import jsonlines

"""
"resultsFile" contains jsons with seed, idealRecmnds and generatedRcmnds
Example:
{"seed": "1371474",
"idealRcmnds": ["2067093", "6572608", "3545808", "6095689"],
"baselineRcmnds": {0: [1371474, 615.019], 1: [2066342, 331.019], 2: [2067093, 312.34818]}
"""


def R_at_n(n, resultsFile):
    """
    n : value of k in P@k
    ToDos: replace baselineRcmnds wit generatedRec
    """
    total_seeds = 0  # to keep track of total seeds
    with jsonlines.open(resultsFile) as reader:
        recallfortotseeds = 0
        for obj in reader:
            total_seeds += 1
            get_K_generatedRec = list()  # get top k generated rec
            totalRel_items = len(obj["idealRcmnds"])  # total ideal rec
            for intr, eachRank in enumerate(obj["baselineRcmnds"].keys()):
                if intr == 0:
                    if obj["baselineRcmnds"][eachRank][0] != int(obj["seed"]):
                        print(
                            "Check: The seed is not the first recoommendation",
                        )
                    continue
                elif intr > n:
                    break
                else:
                    get_K_generatedRec.append(
                        obj["baselineRcmnds"][eachRank][0],
                    )
            count = 0  # count nr of gen rec in ideal rec
            for eachGenRec in get_K_generatedRec:
                if str(eachGenRec) in obj["idealRcmnds"]:
                    count += 1
            recallfortotseeds += count / totalRel_items
    return recallfortotseeds / total_seeds


def P_at_n(n, resultsFile):
    """
    n : value of k in P@k
    ToDos: replace baselineRcmnds wit generatedRec
    """
    total_seeds = 0  # to keep track of total seeds
    with jsonlines.open(resultsFile) as reader:
        recallfortotseeds = 0
        for obj in reader:
            total_seeds += 1
            get_K_generatedRec = list()  # get top k generated rec
            totalRel_items = len(obj["idealRcmnds"])
            for intr, eachRank in enumerate(obj["baselineRcmnds"].keys()):
                if intr == 0:
                    if obj["baselineRcmnds"][eachRank][0] != int(obj["seed"]):
                        print(
                            "Check: The seed is not the first recoommendation",
                        )
                    continue
                elif intr > n:
                    break
                else:
                    get_K_generatedRec.append(
                        obj["baselineRcmnds"][eachRank][0],
                    )
            count = 0  # count nr of gen rec in ideal rec
            for eachGenRec in get_K_generatedRec:
                if str(eachGenRec) in obj["idealRcmnds"]:
                    count += 1
            balenced_P = n
            if n > totalRel_items:
                balenced_P = totalRel_items  # adjust to get ideal P=1 if possible else we will never get P=1 for seeds that have < k ideal recs
            recallfortotseeds += count / n
    return recallfortotseeds / total_seeds


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
