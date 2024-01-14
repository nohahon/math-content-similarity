import re
import os
import sys
import csv
import torch
import random
import itertools
import pickle
from title_similarity import getSEEDIds
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
csv.field_size_limit(100000000)

INSTRUCTIONS = {
    "qa": {
        "query": "Represent this query for retrieving relevant documents: ",
        "key": "Represent this document for retrieval: ",
    }
}

def getDocandRefstyle():
    dictRef = dict()
    filename = "/beegfs/schubotz/ankit/data/references_withIDs.csv"
    with open(filename, 'r', encoding="utf-8", errors='ignore') as csvfile:
        csvreader = csv.reader(csvfile)
        first_row = next(csvreader)  # Read the first row
        for eachro in csvreader:
            dictRef[eachro[2]] = eachro[1]
    return dictRef

def getDocID_to_zbl():
    dictRef = dict()
    with open("/beegfs/schubotz/ankit/data/zbMATH_id_to_ZBL.csv", mode ='r') as csvfile:
        csvFile = csv.reader(csvfile)
        first_row = next(csvFile)
        for lines in csvFile:
            dictRef[lines[0]] = lines[1]
    zbl_to_ids = {y: x for x, y in dictRef.items()}
    return dictRef,zbl_to_ids

    def getAllReferences():
    """Combine refrences in ZBL and and normal format"""
    file_zblcit = "/beegfs/schubotz/ankit/data/math_citation.csv"
    file_ref = "/beegfs/schubotz/ankit/data/references_withIDs.csv"
    idToZBLcit_o = dict()
    with open(file_zblcit, 'r', encoding="utf-8", errors='ignore') as csvfile:
        csvreader = csv.reader(csvfile)
        first_row = next(csvreader)
        for eachro in csvreader:
            if eachro[1] != "":
                idToZBLcit_o[eachro[0]]=eachro[1]
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
    print("The pribt shooooooooo: ",idToZBLcit_o["1262405"])
    idToRefrences_o = defaultdict(lambda: list())
    with open(file_ref, 'r', encoding="utf-8", errors='ignore') as csvfile:
        csvreader = csv.reader(csvfile)
        first_row = next(csvreader)
        for eachro in csvreader:
            if eachro[2] != "":
                idToRefrences_o[eachro[0]].append([eachro[1],eachro[2]])
    getExistingDEtoRefStyle = getDocandRefstyle()
    for eachID in idToZBLcit_re.keys():
        if eachID in idToRefrences_o.keys():
            listOfpresentZBL = [ele[1] for ele in idToRefrences_o[eachID]]
            for eachZBL in idToZBLcit_re[eachID]:
                if eachZBL not in listOfpresentZBL:
                    if eachZBL in getExistingDEtoRefStyle.keys():
                        idToRefrences_o[eachID].append([getExistingDEtoRefStyle[eachZBL], eachZBL])
        else:
            for eachZBL in idToZBLcit_re[eachID]:
                if eachZBL in getExistingDEtoRefStyle.keys():
                    idToRefrences_o[eachID].append([getExistingDEtoRefStyle[eachZBL], eachZBL])
    return idToRefrences_o

def getRefSimilarity():
    id_toRefrences = getAllReferences()
    zbl_to_ids,ids_to_zbl = getDocID_to_zbl()
    seed_references = dict()
    references_toseeds = defaultdict(lambda:list())
    seed_ids = getSEEDIds()
    for eachId in id_toRefrences.keys():
        if eachId in seed_ids:
            seed_references[ids_to_zbl[eachId]] = id_toRefrences[eachId]
    print(len(seed_references))
    print(seed_references.keys())
    getDocref= getDocandRefstyle()
    for eachId in id_toRefrences.keys():
        for inter in id_toRefrences[eachId]:
            try:
                if zbl_to_ids[inter[1]] in seed_ids:
                    references_toseeds[inter[1]].append([getDocref[ids_to_zbl[eachId]] ,eachId])
            except:
                continue
                # some ZBL Ids are DE Ids (mistakes)
    print(len(references_toseeds))
    print(references_toseeds.keys())
    return seed_references,references_toseeds

def getScores():
    getDocref= getDocandRefstyle()
    seed_references,references_toseeds = getRefSimilarity()
    instruction = INSTRUCTIONS["qa"]
    tokenizer = AutoTokenizer.from_pretrained('BAAI/llm-embedder')
    model = AutoModel.from_pretrained('BAAI/llm-embedder')
    for eachK in seed_references.keys():
        if eachK in getDocref.keys():
	        queries = [instruction["query"] +  getDocref[eachK]]
	        query_inputs = tokenizer(queries, padding=True,truncation=True, return_tensors='pt')
	        keys = [instruction["key"] + key[0] for key in seed_references[eachK]]
	        key_inputs = tokenizer(keys, padding=True,truncation=True, return_tensors='pt')
	        with torch.no_grad():
	           query_outputs = model(**query_inputs)
	           key_outputs = model(**key_inputs)
	           query_embeddings = query_outputs.last_hidden_state[:, 0]
	           key_embeddings = key_outputs.last_hidden_state[:, 0]
	           query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
	           key_embeddings = torch.nn.functional.normalize(key_embeddings, p=2, dim=1)
	        similarity = query_embeddings @ key_embeddings.T
	        with open(+str(eachK)+'_.pkl', 'wb') as f:
	           pickle.dump(similarity,f)

getScores()
#getRefSimilarity()
#getAllReferences()
