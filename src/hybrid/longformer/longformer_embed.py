import os
import sys
import pickle
import torch
import numpy as np
import jsonlines
from typing import List, Tuple, Dict
from torch.nn.functional import cosine_similarity
from transformers import LongformerModel, LongformerTokenizer
sys.path.append('/beegfs/schubotz/ankit/code/mabowdor')
import mabowdorScores

def getIdealRec():
    return mabowdorScores.getidealRecommendations()

def getFeatuResRe(seed_):
    """lod all top 1k recommendations and retrun seeds"""
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
    return abstr[seed_][:1000],citn_val,keywrd[seed_][:1000],math[seed_][:1000],msc[seed_][:1000],titls[seed_][:1000]

def getseedpotrec(dataset):
    # return dict with key as seed and val as set of pot recmnds
    possamp, negsamp = pickle.load(open(dataset, 'rb'))
    seedpotrec = dict()
    pos_seeds = set([eachEle[0] for eachEle in possamp])
    for eachSeed in pos_seeds:
        abs_, cits, kwr, mth, msc, ttle =  getFeatuResRe(eachSeed)
        candidateRec = set()
        for eachF in [abs_, cits, kwr, mth, msc, ttle]: #all features [abs_, cits, kwr, mth, msc, ttle]
            for eachEle in eachF:
                candidateRec.add(eachEle[0])
        seedpotrec[eachSeed] = candidateRec
    return seedpotrec

def load_features():
    """ Load all features"""
    dir_feat = "/beegfs/schubotz/ankit/code/SeqRecExp/zbDataReRanker/dataFeatures/"
    features_ = {
        "feat_abs" : pickle.load(open(dir_feat+"initRank_featAbstr.pkl", "rb")),
        "feat_kwrd" : pickle.load(open(dir_feat+"initRank_featKwrd.pkl", "rb")),
        "feat_ref" : pickle.load(open(dir_feat+"initRank_featRefrnc.pkl", "rb")),
        "feat_mscs" : pickle.load(open(dir_feat+"initRank_featMSCs.pkl", "rb")),
        "feat_title" : pickle.load(open(dir_feat+"initRank_featTitle.pkl", "rb"))
    }
    return features_

def getdocString(doc_id, features):
    """given Is and feature return One Hot String """
    oneHotstr = ""
    for key in features.keys():
        feature_dict = features[key]
        if doc_id in feature_dict:
            oneHotstr += feature_dict[doc_id]
            oneHotstr += " "
    return oneHotstr

def getModel():
    model_name='jpwahle/longformer-base-plagiarism-detection'
    tokenizer = LongformerTokenizer.from_pretrained(model_name)
    model = LongformerModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer

def get_longformer_embeddings(documents: List[str], batch_size=100) -> torch.Tensor:
    """Generate embeddings for a list of documents using Longformer."""
    all_embeddings = []
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        encoded = tokenizer(batch_docs, add_special_tokens=True, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**encoded)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]  # Extract the [CLS] token embeddings
        all_embeddings.append(batch_embeddings)
    # Concatenate all batch embeddings into a single tensor
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings

def rank_recommendations(seed_document, recommendations, batch_size=100):
    """Rank recommendations based on cosine similarity to the seed document."""
    documents = [seed_document[1]] + recommendations[1]
    embeddings = get_longformer_embeddings(documents, batch_size=batch_size)
    seed_emb = embeddings[0].unsqueeze(0)
    rec_embs = embeddings[1:]
    # Calculate cosine similarities between the seed and each recommendation
    similarities = cosine_similarity(seed_emb, rec_embs).squeeze()
    # Pair each recommendation with its similarity score and sort by score
    scored_recommendations = [(doc, float(sim)) for doc, sim in zip(recommendations[0], similarities)]
    scored_recommendations.sort(key=lambda x: x[1], reverse=True)

    return scored_recommendations

def lngFrmrEmbedCos():
    dataset_test = "/beegfs/schubotz/ankit/code/evaluation/hybridApproach/re-ranker/LibRerank/Data/zbmath/test_posandnegPairs.pkl"
    sedpotrec = getseedpotrec(dataset_test)
    alreadyDon = "/beegfs/schubotz/ankit/data/HyMathRecResults/reRanker/LongFormEmbedd/data_scores/"
    alreadyDon_list = os.listdir(alreadyDon)
    numbers_alreadydone = [filename.split('_')[1].split('.')[0] for filename in alreadyDon_list]
    features = load_features()
    for eachSeed in sedpotrec.keys():
        if eachSeed not in numbers_alreadydone:
            seed_str = getdocString(eachSeed, features)
            rec_str = [getdocString(potRec, features) for potRec in sedpotrec[eachSeed]]
            ranked_recs = rank_recommendations([eachSeed, seed_str],[list(sedpotrec[eachSeed]), rec_str])
            with open("data_scores/lnfrm_"+str(eachSeed)+".pkl", "wb") as wpf:
                pickle.dump(ranked_recs, wpf)

def getEvalScoresReRanker():
    """ Given ecosnien similartites get evalresult file for evaluation scores """
    getallDirFiles = os.listdir("data_scores/")
    seed_idlrecmnds = getIdealRec()
    resultsdict = {"seed": None, "idealRcmnds": None, "baselineRcmnds":None}
    with jsonlines.open('rslts_lnfrmer_base_combft_embe.jsonl', mode='w') as writer:
        for eachF in getallDirFiles:
            docName = eachF.split('_')[1].split('.')[0]
            scores_ = pickle.load(open("data_scores/"+eachF, 'rb'))
            baselinercmnds = dict()
            baselinercmnds[str(0)] = [int(docName), 1.0]
            id_h = 0
            for eachScores in scores_[:20]:
                id_h += 1
                baselinercmnds[str(id_h)] = [int(eachScores[0]), eachScores[1]]
            resultsdict["seed"] = docName
            resultsdict["idealRcmnds"] = seed_idlrecmnds[docName]
            resultsdict["baselineRcmnds"] = baselinercmnds
            writer.write(resultsdict)

def testEmbeddings():
    """ Testing if correct embeddings are being generated """
    features = load_features()
    seed_str = getdocString("1192411", features)
    pot_rec = getdocString("1408414", features)
    encoded = tokenizer([seed_str, pot_rec],add_special_tokens=True, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**encoded)
        batch_embeddings = outputs.last_hidden_state[:, 0, :]
        print(cosine_similarity(batch_embeddings[0].unsqueeze(0), batch_embeddings[1].unsqueeze(0)))

#lngFrmrEmbedCos()
#getEvalScoresReRanker()