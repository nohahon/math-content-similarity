import re
import csv
import pickle
from spacy.lang.en import English

csv.field_size_limit(100000000)
nlp = English()
tokenizer = nlp.tokenizer


def getSEEDIds():
    """
    get seed IDS in a list
    """
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


def getidealrecommendations():
    listDocs = list()
    with open(
        "data/recommendationPairs.csv",
        mode="r",
    ) as csvfile:
        csvFile = csv.reader(csvfile)
        for lines in csvFile:
            IdsandRec = list(filter(None, lines))
            listDocs += IdsandRec[1:]
    return listDocs


def read_and_print_csv_rows(filename):
    """
    filename = file where zbMATH data is present
    output= dict with ey as zbMATH ID and value as asbstract/review/summarry
    """
    dataWhole = dict()
    with open(filename, "r", encoding="utf-8", errors="ignore") as csvfile:
        csvreader = csv.reader(csvfile)
        first_row = next(csvreader)  # Read the first row
        for eachro in csvreader:
            dataWhole[eachro[0]] = eachro[2]
    return dataWhole


def getSimilarMCSs(seedMSCs, recMSCs):
    commMSCs = len(list(set(seedMSCs.split()) & set(recMSCs.split())))
    mscOverlap = (commMSCs / len(seedMSCs.split())) * 100
    # print(seedMSCs, recMSCs)
    # print("MSC overlap is: ", mscOverlap)
    return mscOverlap


def getTokens(textPass):
    listToks = set()
    rmSPace = " ".join(textPass.split())
    tokensReview = tokenizer(rmSPace)
    regex = re.compile("[1-9+.,-;@_!#$%^&*()<>?/\\\|}{~:]")
    for indTok in tokensReview:
        if regex.search(indTok.lower_) == None:
            listToks.add(indTok.lower_)
    return list(listToks)


def getSimilarKey(seedK, recK):
    seedKeyw = getTokens(seedK)
    recKeyw = getTokens(recK)
    if len(seedKeyw) == 0:
        overlap = 0
    else:
        lenInters = len(list(set(seedKeyw) & set(recKeyw)))
        overlap = (lenInters / len(seedKeyw)) * 100
    return overlap


def getSimilarities():
    allAbstrs = read_and_print_csv_rows(
        "arxMLiv/zbmath_abstracts.csv",
    )
    print(len(allAbstrs))
    alScores = dict()
    for seed in getSEEDIds():
        locDict = dict()
        for eachD in allAbstrs.keys():
            locDict[eachD] = getSimilarKey(allAbstrs[seed], allAbstrs[eachD])
        alScores[seed] = locDict
    sorted_dict = dict()
    for each_ in alScores.keys():
        sorted_dict[each_] = sorted(
            alScores[each_].items(),
            key=lambda x: x[1],
            reverse=True,
        )
    with open("KeywordsSimil_scores.pkl", "wb") as f:
        pickle.dump(sorted_dict, f)


getSimilarities()
