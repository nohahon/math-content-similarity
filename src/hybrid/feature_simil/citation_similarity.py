import re
import os
import sys
import csv
import torch
import pickle
import jsonlines
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel

csv.field_size_limit(100000000)

INSTRUCTIONS = {
    "qa": {
        "query": "Represent this query for retrieving relevant documents: ",
        "key": "Represent this document for retrieval: ",
    },
}


def getidealrecommendations():
    listDocs = dict()
    with open(
        "/beegfs/schubotz/ankit/data/recommendationPairs.csv",
        mode="r",
    ) as csvfile:
        csvFile = csv.reader(csvfile)
        for lines in csvFile:
            IdsandRec = list(filter(None, lines))
            listDocs[IdsandRec[0]] = IdsandRec[1:]
    return listDocs


def getSEEDIds():
    """get seed IDS in a list"""
    listDocs = list()
    with open(
        "/beegfs/schubotz/ankit/data/recommendationPairs.csv",
        mode="r",
    ) as csvfile:
        csvFile = csv.reader(csvfile)
        for lines in csvFile:
            IdsandRec = list(filter(None, lines))
            listDocs.append(IdsandRec[0])
    return listDocs


def getDocandRefstyle():
    dictRef = dict()
    filename = "/beegfs/schubotz/ankit/data/references_withIDs.csv"
    with open(filename, "r", encoding="utf-8", errors="ignore") as csvfile:
        csvreader = csv.reader(csvfile)
        first_row = next(csvreader)  # Read the first row
        for eachro in csvreader:
            dictRef[eachro[2]] = eachro[1]
    return dictRef


def getDocID_to_zbl():
    dictRef = dict()
    with open(
        "/beegfs/schubotz/ankit/data/zbMATH_id_to_ZBL.csv",
        mode="r",
    ) as csvfile:
        csvFile = csv.reader(csvfile)
        next(csvFile)
        for lines in csvFile:
            dictRef[lines[0]] = lines[1]
    zbl_to_ids = {y: x for x, y in dictRef.items()}
    return dictRef, zbl_to_ids


def getAllReferences():
    """Combine refrences in ZBL and and normal format"""
    file_zblcit = "/beegfs/schubotz/ankit/data/math_citation.csv"
    file_ref = "/beegfs/schubotz/ankit/data/references_withIDs.csv"
    idToZBLcit_o = dict()
    with open(file_zblcit, "r", encoding="utf-8", errors="ignore") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for eachro in csvreader:
            if eachro[1] != "":
                idToZBLcit_o[eachro[0]] = eachro[1]
    idToZBLcit_re = dict()
    for eachD in idToZBLcit_o.keys():
        temp_c = idToZBLcit_o[eachD].split(";")
        temp_cn = list()
        for eachE in temp_c:
            ele_n = eachE.split(" ")
            if "Zbl" in ele_n:
                ele_n = "".join(ele_n)
                ele_n = ele_n.split("Zbl")[1]
                temp_cn.append(ele_n)
            elif "JFM" in ele_n:
                ele_n = "".join(ele_n)
                ele_n = ele_n.split("JFM")[1]
                temp_cn.append(ele_n)
            elif "ERAM" in ele_n:
                ele_n = "".join(ele_n)
                ele_n = ele_n.split("ERAM")[1]
                temp_cn.append(ele_n)
            else:
                # some IDs start woth "JM" or no identifier or just weird latex form.
                # Upon manually checking these documents had no to very little dataat zbMATH Open
                # Hence ignored for now
                continue
        idToZBLcit_re[eachD] = temp_cn
    print("The pribt shooooooooo: ", idToZBLcit_o["1262405"])
    idToRefrences_o = defaultdict(lambda: list())
    with open(file_ref, "r", encoding="utf-8", errors="ignore") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for eachro in csvreader:
            if eachro[2] != "":
                idToRefrences_o[eachro[0]].append([eachro[1], eachro[2]])
    getExistingDEtoRefStyle = getDocandRefstyle()
    for eachID in idToZBLcit_re.keys():
        if eachID in idToRefrences_o.keys():
            listOfpresentZBL = [ele[1] for ele in idToRefrences_o[eachID]]
            for eachZBL in idToZBLcit_re[eachID]:
                if eachZBL not in listOfpresentZBL:
                    if eachZBL in getExistingDEtoRefStyle.keys():
                        idToRefrences_o[eachID].append(
                            [getExistingDEtoRefStyle[eachZBL], eachZBL],
                        )
        else:
            for eachZBL in idToZBLcit_re[eachID]:
                if eachZBL in getExistingDEtoRefStyle.keys():
                    idToRefrences_o[eachID].append(
                        [getExistingDEtoRefStyle[eachZBL], eachZBL],
                    )
    return idToRefrences_o


def getRefSimilarity():
    id_toRefrences = getAllReferences()
    zbl_to_ids, ids_to_zbl = getDocID_to_zbl()
    seed_references = dict()
    references_toseeds = defaultdict(lambda: list())
    seed_ids = getSEEDIds()
    for eachId in id_toRefrences.keys():
        if eachId in seed_ids:
            seed_references[ids_to_zbl[eachId]] = id_toRefrences[eachId]
    getDocref = getDocandRefstyle()
    print("Initial seed ref len: ", len(seed_references))
    for eachId in id_toRefrences.keys():
        for inter in id_toRefrences[eachId]:
            try:
                if zbl_to_ids[inter[1]] in seed_ids:
                    references_toseeds[inter[1]].append(
                        [getDocref[ids_to_zbl[eachId]], eachId],
                    )
            except:
                continue
                # some ZBL Ids are DE Ids (mistakes)
    print("Initial re seed len: ", len(references_toseeds))
    return seed_references, references_toseeds


def getScores():
    getDocref = getDocandRefstyle()
    seed_references, references_toseeds = getRefSimilarity()
    instruction = INSTRUCTIONS["qa"]
    tokenizer = AutoTokenizer.from_pretrained("BAAI/llm-embedder")
    model = AutoModel.from_pretrained("BAAI/llm-embedder")
    for eachK in references_toseeds.keys():
        if eachK in getDocref.keys():
            print(eachK, getDocref[eachK])
            queries = [instruction["query"] + getDocref[eachK]]
            query_inputs = tokenizer(
                queries,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            keys = [
                instruction["key"] + key[0]
                for key in references_toseeds[eachK]
            ]
            key_inputs = tokenizer(
                keys,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                query_outputs = model(**query_inputs)
                key_outputs = model(**key_inputs)
                query_embeddings = query_outputs.last_hidden_state[:, 0]
                key_embeddings = key_outputs.last_hidden_state[:, 0]
                query_embeddings = torch.nn.functional.normalize(
                    query_embeddings,
                    p=2,
                    dim=1,
                )
                key_embeddings = torch.nn.functional.normalize(
                    key_embeddings,
                    p=2,
                    dim=1,
                )
            similarity = query_embeddings @ key_embeddings.T
            with open(
                "data_ne/referencesScores/citedIn/" + str(eachK) + ".pkl",
                "wb",
            ) as f:
                pickle.dump(similarity, f)


def getScoresFromPick():
    """scores are stored a dir so getting them in a dict to use"""
    dir_seedref = "data_ne/referencesScores/refrences"
    dir_reftoseed = "data_ne/referencesScores/citedIn"
    getAllseeds_r = os.listdir(dir_seedref)
    getAllseed_c = os.listdir(dir_reftoseed)
    seedref_scores = dict()
    reftoseed_scores = dict()
    pattern = r"(.*?)"
    for eachS in getAllseeds_r:
        with open(os.path.join(dir_seedref, eachS), "rb") as f:
            seedToref = pickle.load(f)
        seedref_scores[re.findall(pattern, eachS)[0]] = seedToref[0]
    # print(seedref_scores)
    for eachS_ in getAllseed_c:
        with open(os.path.join(dir_reftoseed, eachS_), "rb") as f:
            refToseed = pickle.load(f)
        reftoseed_scores[re.findall(pattern, eachS_)[0]] = refToseed[0]
    # print(reftoseed_scores)
    return seedref_scores, reftoseed_scores


def arrangeScoresAsRec():
    zbl_to_ids, ids_to_zbl = getDocID_to_zbl()
    seed_references, references_toseeds = getRefSimilarity()
    getDocref = getDocandRefstyle()
    seedToRef = dict()
    refToSeeds = dict()
    for eachK in references_toseeds.keys():
        if eachK in getDocref.keys():
            refToSeeds[zbl_to_ids[eachK]] = [
                key[1] for key in references_toseeds[eachK]
            ]
    for eachK in seed_references.keys():
        try:
            if eachK in getDocref.keys():
                seedToRef[zbl_to_ids[eachK]] = [
                    key[1] for key in seed_references[eachK]
                ]
        except:
            continue
    scrs_seedref, scrs_refseeds = getScoresFromPick()
    # now assigning key a score
    seedToref_final = dict()
    for eachSeed in seedToRef.keys():
        dict_loc = dict()
        scores_ = scrs_seedref[ids_to_zbl[eachSeed]]
        for id_, eachRec in enumerate(seedToRef[eachSeed]):
            dict_loc[eachRec] = scores_[id_]
        seedToref_final[eachSeed] = dict_loc
    print("Final seeedRef len: ", len(seedToref_final))
    refToseed_final = dict()
    for eachSeed in refToSeeds.keys():
        dict_loc = dict()
        try:
            scores_ = scrs_refseeds[ids_to_zbl[eachSeed]]
            for id_, eachRec in enumerate(refToSeeds[eachSeed]):
                dict_loc[eachRec] = scores_[id_]
            refToseed_final[eachSeed] = dict_loc
        except:
            print(eachSeed)
            continue
            # print(scrs_refseeds.keys())
            # sys.exit(0)
    print("Final ref seed len: ", len(refToseed_final))
    with open("seedToref_scores.pkl", "wb") as f:
        pickle.dump(seedToref_final, f)
    with open("refToseed_scores.pkl", "wb") as f:
        pickle.dump(refToseed_final, f)


def createResults():
    zbl_to_ids, ids_to_zbl = getDocID_to_zbl()
    with open("seedToref_scores.pkl", "rb") as f:
        seedToref_ = pickle.load(f)
    rev_seedToref_ = dict()
    for eachS in seedToref_.keys():
        localCorrect = dict()
        for eachre in seedToref_[eachS].keys():
            try:
                localCorrect[zbl_to_ids[eachre]] = seedToref_[eachS][eachre]
            except:
                print(eachre)
        rev_seedToref_[eachS] = localCorrect
    # print(rev_seedToref_[list(rev_seedToref_.keys())[0]])
    with open("refToseed_scores.pkl", "rb") as f:
        refToseed__ = pickle.load(f)
    # print(refToseed_[list(refToseed_.keys())[0]])
    # sys.exit(0)
    mergeddict = {**rev_seedToref_, **refToseed__}
    rankedS = dict()
    for eachSeed in mergeddict.keys():
        sorted_ = sorted(
            mergeddict[eachSeed].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        rankedS[eachSeed] = sorted_
    with open("rslts_directRef.pkl", "wb") as f:
        pickle.dump(rankedS, f)
    sys.exit(0)
    resultsdict = {"seed": None, "idealRcmnds": None, "baselineRcmnds": None}
    seed_idlrecmnds = getidealrecommendations()
    with jsonlines.open("rslts_directRefrences.jsonl", mode="w") as writer:
        for eachSeed in rankedS.keys():
            baselinercmnds = dict()
            for id_, pot_rcmnds in enumerate(rankedS[eachSeed][:11]):
                baselinercmnds[str(id_)] = [
                    int(pot_rcmnds[0]),
                    pot_rcmnds[1].item(),
                ]
            resultsdict["seed"] = eachSeed
            resultsdict["idealRcmnds"] = seed_idlrecmnds[eachSeed]
            resultsdict["baselineRcmnds"] = baselinercmnds
            writer.write(resultsdict)
    # print(len(set(seedToref_.keys()).union(set(refToseed__.keys()))))


def getTrueSeedsHavCit():
    file_zblcit = "/beegfs/schubotz/ankit/data/math_citation.csv"
    file_ref = "/beegfs/schubotz/ankit/data/references_withIDs.csv"

    idToRefrences_o = defaultdict(lambda: list())
    with open(file_ref, "r", encoding="utf-8", errors="ignore") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for eachro in csvreader:
            if eachro[0] == "1247615":
                print("Inside math_cit: ", eachro)
            if eachro[2] != "":
                idToRefrences_o[eachro[0]].append(eachro[2])

    idToZBLcit_o = dict()
    with open(file_zblcit, "r", encoding="utf-8", errors="ignore") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for eachro in csvreader:
            if eachro[0] == "1247615":
                print("Inside math_next h: ", eachro)
            if eachro[1] != "":
                idToZBLcit_o[eachro[0]] = eachro[1]

    idToZBLcit_re = dict()
    for eachD in idToZBLcit_o.keys():
        temp_c = idToZBLcit_o[eachD].split(";")
        temp_cn = list()
        for eachE in temp_c:
            ele_n = eachE.split(" ")
            if "Zbl" in ele_n:
                ele_n = "".join(ele_n)
                ele_n = ele_n.split("Zbl")[1]
                temp_cn.append(ele_n)
            elif "JFM" in ele_n:
                ele_n = "".join(ele_n)
                ele_n = ele_n.split("JFM")[1]
                temp_cn.append(ele_n)
            elif "ERAM" in ele_n:
                ele_n = "".join(ele_n)
                ele_n = ele_n.split("ERAM")[1]
                temp_cn.append(ele_n)
            else:
                # some IDs start woth "JM" or no identifier or just weird latex form.
                # Upon manually checking these documents had no to very little dataat zbMATH Open
                # Hence ignored for now
                continue
        idToZBLcit_re[eachD] = temp_cn

    print("in re: ", idToZBLcit_re["1247615"])
    print("in o: ", idToRefrences_o["1247615"])
    # print(len(idToZBLcit_re))
    # print(len(idToRefrences_o))
    # sys.exit(0)
    # print(len(idToZBLcit_o.keys()))
    seedIds = getSEEDIds()
    # print(len(set(list(idToZBLcit_re.keys())).union(set(list(idToRefrences_o.keys())))))
    print(
        "documents whose ref are available: ",
        len(
            set(list(idToZBLcit_re.keys()))
            .union(set(list(idToRefrences_o.keys())))
            .intersection(set(seedIds)),
        ),
    )
    # print("documents whose ref are available: ",set(list(idToZBLcit_re.keys())).union(set(list(idToRefrences_o.keys()))).intersection(set(seedIds)))
    # print("doc whos ref are uinav: ", set(seedIds) - set(list(idToZBLcit_re.keys())).union(set(list(idToRefrences_o.keys()))).intersection(set(seedIds)))

    final_links = dict()
    for each_k in set(list(idToRefrences_o.keys())).union(
        set(list(idToZBLcit_re.keys())),
    ):
        set_o = set()
        set_re = set()
        if each_k in idToRefrences_o.keys():
            set_o = set(idToRefrences_o[each_k])
        if each_k in idToZBLcit_re.keys():
            set_re = set(idToZBLcit_re[each_k])
        setAdd = set_o.union(set_re)
        final_links[each_k] = list(setAdd)

    # sys.exit(0)
    print(
        "Related in the final: ",
        set(list(final_links.keys())).intersection(set(seedIds)),
    )
    # with open("zbMATHOpen_citNw.pkl", 'wb') as f:
    #    pickle.dump(final_links,f)

    seedIds = getSEEDIds()
    # alldata = idToZBLcit_o
    # alldata = getAllReferences()
    print(
        len(
            set(list(idToZBLcit_o.keys()))
            .union(set(list(idToZBLcit_re.keys())))
            .intersection(set(seedIds)),
        ),
    )
    # print(len(set(list(alldata.keys())).intersection(set(seedIds))))
    sys.exit(0)
    # print(alldata[list(alldata.keys())[0]])
    for eachSeed in set(list(alldata.keys())).intersection(set(seedIds)):
        print(alldata[eachSeed])


getTrueSeedsHavCit()
# createResults()
# getScoresFromPick()
# arrangeScoresAsRec()
# getScores()
# getRefSimilarity()
# getAllReferences()
# getDocID_to_zbl()
