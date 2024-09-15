import sys
import pickle
import jsonlines
from transformers import pipeline

def evalScores(seed_,k_, idealRec, generatedRec):
    get_K_generatedRec = list()
    for intr, eachRank in enumerate(generatedRec.keys()):
        if intr == 0:
            if generatedRec[eachRank][0] != int(seed_):
                get_K_generatedRec.append(str(generatedRec[eachRank][0]))
        elif intr > k_:
            break
        else:
            get_K_generatedRec.append(str(generatedRec[eachRank][0]))
    count = len(set(get_K_generatedRec).intersection(set(idealRec)))
    precision = count/k_
    recall = count/len(idealRec)
    return recall,precision

def getText(listOfdocs_, titles_):
    text_ = list()
    for eachDoc in listOfdocs_:
        if eachDoc in titles_:
            text_.append(" "+str(eachDoc)+" :"+titles_[eachDoc]+".")
    return text_

def llamaM(msg_):
    messages = [{"role": "user", "content": msg_},]
    model_= "/beegfs/schubotz/ankit/code/evaluation/hybridApproach/re-ranker/promptbased/rereranker/lfactoryFT/LLaMA-Factory/models/llama3_lora_sft_zb/"
    pipe = pipeline("text-generation", model=model_,device=0, max_length=10000)
    return pipe(messages)

def getPrompt(seed_, potrec_):
    loc_title = "/beegfs/schubotz/ankit/code/SeqRecExp/zbDataReRanker/dataFeatures/"
    titls = pickle.load(open(loc_title+"initRank_featKwrd.pkl", 'rb'))
    seedText = getText([seed_], titls)[0]
    potentialdocText = getText(potrec_, titls)
    prmpt = "You are working as a Recommender System in scientific library with research papers in pure and applied mathematics. The library you are working with is zbMATHOpen. I will give you a seed document and a list of documents. You have to rank the documents from the list in ranked order considering first document in the ranked list as most likely recommendation and last as least. You can consider conceptual similarity, interlinked concepts to rank, etc. Along with document I also give assigned ID followed by document title. Seed document "
    prmpt += seedText
    prmpt += " ".join(potentialdocText)
    prmpt += " Please only output ranked list of IDs wihtout any text in the response and do not explain the output."
    return prmpt

def rererank(seed, potenRec):
    potentRec = [str(potenRec[rec][0]) for rec in list(potenRec.keys())[1:90]]
    getprompt = getPrompt(seed, set(potentRec))
    llaResp = llamaM(getprompt)
    return llaResp[0]['generated_text'][1]['content']

def saveGenerateRankedlist():
    dir_jsonF = "/beegfs/schubotz/ankit/data/HyMathRecResults/reRanker/rslts_mpnn_15_10_hdl.jsonl"
    with jsonlines.open(dir_jsonF) as reader:
        with jsonlines.open('data/fineTuned/keywords_full.jsonl', mode='w') as writer:
            for obj in reader:
                resultsdict = {"seed": None, "response": None}
                re_ranked = rererank(obj["seed"], obj["baselineRcmnds"])
                writer.write({"seed": obj["seed"], "response": re_ranked})
                writer.close()
                writer = jsonlines.open('data/fineTuned/keywords_full.jsonl', mode='a')

saveGenerateRankedlist()