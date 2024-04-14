import sys
import pickle
import numpy as np
import jsonlines
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
sys.path.append('/beegfs/schubotz/ankit/code/mabowdor')
import mabowdorScores

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

def get_data_r(dataset):
    # getting re-ranker training data and testing data
    possamp, negsamp = pickle.load(open(dataset, 'rb'))
    records = []
    seedToidlrcmnds = mabowdorScores.getidealRecommendations()
    seedrecPrs = list()
    pos_seeds = set([eachEle[0] for eachEle in possamp])
    for eachSeed in pos_seeds:
        abs_, cits, kwr, mth, msc, ttle =  getFeatuResRe(eachSeed)
        #for id_,eachF in enumerate([abs_, kwr, mth, msc, ttle]):
        for id_,eachF in enumerate([abs_, cits, kwr, mth, msc, ttle]):
            for eachEle in eachF:
                if eachEle[0] in seedToidlrcmnds[eachSeed]:
                    records.append([eachEle[1],id_,1.0])
                    seedrecPrs.append([eachSeed,eachEle[0]])
                else:
                    records.append([eachEle[1],id_,0.0])
                    seedrecPrs.append([eachSeed,eachEle[0]])
    with open("mpnn_data.pkl", "wb") as wpf:
        pickle.dump(seedrecPrs, wpf)
    return records

def mpnn():
    dataset_train = "/beegfs/schubotz/ankit/code/evaluation/hybridApproach/re-ranker/LibRerank/Data/zbmath/train_posandnegPairs.pkl"
    list_of_list_samps = get_data_r(dataset_train)
    training_data = np.array(list_of_list_samps)
    X_train = training_data[:, :-1]  # features: similarity score and feature ID
    y_train = training_data[:, -1]   # labels
    
    dataset_test = "/beegfs/schubotz/ankit/code/evaluation/hybridApproach/re-ranker/LibRerank/Data/zbmath/test_posandnegPairs.pkl"
    list_of_testsampe = get_data_r(dataset_test)
    test_data = np.array(list_of_testsampe)
    test_data = test_data[:,:-1]
    # Standardizing the feature data for better performance with the neural network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Defining the neural network classifier
    nn_clf = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
    
    # Training the neural network classifier
    nn_clf.fit(X_train_scaled, y_train)
    
    # Preparing the test data by applying the same scaling
    X_test_scaled = scaler.transform(test_data)
    
    # Predicting probabilities for the test data
    test_probs = nn_clf.predict_proba(X_test_scaled)  # Getting the probability for label 1.0

    with open("mpnn_predictns.pkl", "wb") as wpf:
        pickle.dump(test_probs, wpf)


def getEvalScoresReRanker():
    """
    Given predictions from re-ranker, get values in resultsfiles
    """
    predictions_dir = "mpnn_predictns.pkl"
    seedrecpairs_dir = "mpnn_data.pkl"
    with open(predictions_dir, "rb") as prdf:
        predictions = pickle.load(prdf)
    with open(seedrecpairs_dir, "rb") as prda:
        seedrecpairs = pickle.load(prda)
    dictScores = defaultdict(lambda:list()) #for storing seed and their predictions
    print(len(predictions), len(seedrecpairs))
    for id_,eachPred in enumerate(predictions):
        dictScores[seedrecpairs[id_][0]].append([seedrecpairs[id_][1],eachPred[1]])
    print(len(dictScores))
    dictSortedScores = dict()
    for eachSeed in dictScores.keys():
        sorted_list = sorted(dictScores[eachSeed], key=lambda x: x[1], reverse=True)
        dictSortedScores[eachSeed] = sorted_list
    resultsdict = {"seed": None, "idealRcmnds": None, "baselineRcmnds":None}
    with jsonlines.open('rslts_mpnn.jsonl', mode='w') as writer:
        seed_idlrecmnds = mabowdorScores.getidealRecommendations()
        for each_ in dictSortedScores.keys():
            baselinercmnds = dict()
            baselinercmnds[str(0)] = [int(each_), 1.0]
            id_h = 0
            for id_,eachRcmnds in enumerate(dictSortedScores[each_][:1500]):
                id_h += 1
                baselinercmnds[str(id_h)] = [int(eachRcmnds[0]),eachRcmnds[1]]
            resultsdict["seed"] = each_
            resultsdict["idealRcmnds"] = seed_idlrecmnds[each_]
            resultsdict["baselineRcmnds"] = baselinercmnds
            writer.write(resultsdict)

getEvalScoresReRanker()
#mpnn()
