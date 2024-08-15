import pickle
from collections import defaultdict

csv.field_size_limit(100000000)


def getSEEDIds():
    """get seed IDS in a list"""
    listDocs = list()
    with open(
        "data/recommendationPairs.csv",
        mode="r",
    ) as csvfile:
        csvFile = csv.reader(csvfile)
        for lines in csvFile:
            IdsandRec = list(filter(None, lines))
            listDocs.append(IdsandRec[0])
    return listDocs


def getDocID_to_zbl():
    dictRef = dict()
    with open(
        "data/zbMATH_id_to_ZBL.csv",
        mode="r",
    ) as csvfile:
        csvFile = csv.reader(csvfile)
        next(csvFile)
        for lines in csvFile:
            dictRef[lines[0]] = lines[1]
            # print(dictRef)
            # sys.exit(0)
    zbl_to_ids = {y: x for x, y in dictRef.items()}
    return dictRef, zbl_to_ids


def list_to_dict(input_list):
    return {
        element: [
            other_element
            for other_element in input_list
            if other_element != element
        ]
        for element in input_list
    }


def readANdCOntructCo():
    ids_toZBL, zbl_to_Ids = getDocID_to_zbl()
    cit_r = defaultdict(lambda: list())
    with open("zbMATHOpen_citNw.pkl", "rb") as f:
        seedToref = pickle.load(f)
    for eachVal in seedToref.values():
        if len(eachVal) > 1:
            lrtrnd_dict = list_to_dict(eachVal)
            for key_ in lrtrnd_dict.keys():
                cit_r[key_] += lrtrnd_dict[key_]
            # print(cit_r)
            # sys.exit(0)
    print(len(cit_r))
    allSeeds = getSEEDIds()
    for eachS in allSeeds:
        print(eachS, len(cit_r[zbl_to_Ids[eachS]]))


readANdCOntructCo()
