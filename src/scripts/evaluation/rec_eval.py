import math
import jsonlines


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


def getF1(prec, rec):
    return 2 * (prec * rec) / (prec + rec)


def get_nDCG_real(resultsFile):
    score = 0
    countseeds = 0
    with jsonlines.open(resultsFile) as reader:
        for obj in reader:
            bslineIds = [
                str(every[0])
                for every in list(obj["baselineRcmnds"].values())[1:]
            ]
            idealRcmnds = obj["idealRcmnds"]
            score += ndcg(idealRcmnds, bslineIds, len(idealRcmnds))
            countseeds += 1
    return score / 80


def calculate_relevance_scores(ideal_list, retrieved_list):
    """Generate relevance scores for retrieved_list based on their position in ideal_list"""
    ideal_ranking_dict = {
        doc: len(ideal_list) - idx for idx, doc in enumerate(ideal_list)
    }
    return [ideal_ranking_dict.get(doc, 0) for doc in retrieved_list]


def dcg(relevances, k=None):
    """Compute the Discounted Cumulative Gain."""
    relevances = relevances[:k]
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def ndcg(ideal_list, retrieved_list, k=None):
    """Compute the Normalized Discounted Cumulative Gain."""
    ideal_relevances = calculate_relevance_scores(ideal_list, ideal_list)
    retrieved_relevances = calculate_relevance_scores(
        ideal_list,
        retrieved_list,
    )
    idcg = dcg(ideal_relevances, k)
    if idcg == 0:
        return 0
    return dcg(retrieved_relevances, k) / idcg


def metric_MRR(resultsFile):
    recipr_rank = 0
    countofSeeds = 0
    with jsonlines.open(resultsFile) as reader:
        for obj in reader:
            countofSeeds += 1
            bslineIds = [
                str(every[0])
                for every in list(obj["baselineRcmnds"].values())[1:]
            ]
            findRank = True
            for eachIdeal in obj["idealRcmnds"]:
                if findRank:
                    if eachIdeal in bslineIds:
                        rank = bslineIds.index(eachIdeal) + 1
                        findRank = False
                        break
            if not findRank:
                recipr_rank += 1 / rank
    return recipr_rank / countofSeeds
