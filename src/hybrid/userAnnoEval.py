import sys
import math
import csv
import numpy as np
import os
import re
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score

def getSeedVal(filename):
	# given annotation file return seed to scores in dict
	dictPrec = defaultdict(lambda: list())
	with open(filename, 'r', encoding="utf-8", errors='ignore') as csvfile:
		csvreader = csv.reader(csvfile)
		first_row = next(csvreader)  # Read the first row
		for eachro in csvreader:
			dictPrec[eachro[0]] += eachro[2]
	return dictPrec

def calculatePrec(filename,n_):
	# out of top k recmnds how many are relevant
	seedscores = getSeedVal(filename)
	prec = 0
	for val in seedscores.values():
		revval = [int(st) for st in val[:n_]]
		sum_k = sum(revval)
		prec += (sum_k/n_)
	return prec/len(seedscores.keys())
	 

def calculateRecall(filename, n_):
	# out of all relevant rcmnds how many are present in top k
	seedscores = getSeedVal(filename)
	prec = 0
	for val in seedscores.values():
		revval = [int(st) for st in val[:n_]]
		sum_k = sum(revval)
		prec += (sum_k/10)  #as all 10 recommendations are relevant
	return prec/len(seedscores.keys())

def getF1(prec, rec):
    return 2 * (prec * rec) / (prec + rec)

def getMRR(filename):
	# get MRR
	seedscores = getSeedVal(filename)
	mrr = 0
	for val in seedscores.values():
		try:
			mrr += 1/(val.index('2')+1)
		except:
			continue
	return mrr/len(seedscores.keys())

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

def calc_ndcg(filename):
	seedscores = getSeedVal(filename)
	ndcg_ = 0
	for val in seedscores.values():
		idealRec = [2,2,2,2,2,2,2,2,2,2]
		relrec = [int(st) for st in val]
		ndcg_ += ndcg(idealRec, relrec)
	return ndcg_/len(seedscores.keys())

def calculateEValScores(filename):
	# for Prec and Recall we collapsed 2 to 1 so relevant: 1. irrelevant: 0 (manually in the annotation file)
	# for MRR & nDCG we kept the original annotations
	# filename = "originalAnno/annotation.csv"
	for i in [3,5,10]:
		print("Precision, recall, F1 at ",i, " is: ", calculatePrec(filename,i), 
			calculateRecall(filename,i), getF1(calculatePrec(filename,i), calculateRecall(filename,i)))
	print("MRR and nDCG is: ", getMRR(filename), calc_ndcg(filename))

def useragreement(binary_mode=False):
	annos = list()
	annoNames = list()
	foldername = 'originalAnno'
	for file in os.listdir(foldername):
		filepath = os.path.join(foldername, file)
		if filepath.endswith('.CSV') or filepath.endswith('.csv'):
			annos.append( getSeedVal(filepath) )
			annoNames.append( re.match(".*_(.*?)\\..*", file).group(1) )
			# print("Cohen's kappa loaded " + filepath) # helps debugging

	matrix = list()
	for annotator in annos:
		annomap = list()
		for seedID in annotator.values():
			for score in seedID:
				if binary_mode and 0 < int(score):
					score = 1
				annomap.append(int(score))
		matrix.append(annomap)

	for i, anno in enumerate(annoNames):
		for j in range(i):
			kappa = cohen_kappa_score(matrix[i], matrix[j])
			print( '{}: {:7.4f}'.format( "Cohen's Kappa {:s}-{:s}".format(anno, annoNames[j]).ljust(27), kappa ) )

def fleiss(binary_mode=False):
	annos = list()
	foldername = 'originalAnno'

	for file in os.listdir(foldername):
		filepath = os.path.join(foldername, file)
		if filepath.endswith('.CSV') or filepath.endswith('.csv'):
			annos.append( getSeedVal(filepath) )
			print("Fleiss' kappa loaded " + filepath)

	N = len(annos[0])*10 # number of seed/rec combinations, ie total number of annotated pairs
	n = len(annos) # number of annotators
	k = 3 - binary_mode # number of scores (here 0, 1, 2)

	matrix = list()
	pairid = 0
	for seedID in annos[0]:
		if not seedID: continue

		for recID in range(len(annos[0][seedID])):
			matrix.append(list())
			for score in range(k):
				counter = 0
				for annotator in annos:
					if binary_mode:
						if 0 < score and score <= int(annotator[seedID][recID]):
							counter = counter+1
						elif 0 == score and score == int(annotator[seedID][recID]):
							counter = counter+1
					else:
						if score == int(annotator[seedID][recID]):
							counter = counter+1
				matrix[pairid].append(counter)
			pairid = pairid + 1

	PA = sum([sum([nij * (nij-1) for nij in row]) / (n*(n-1)) for row in matrix]) / N
	print("PA =", PA)

	PE = sum( [p**2 for p in [sum([ row[col] for row in matrix ])/(N*n) for col in range(k)]] )
	print("PE =", PE)

	kappa = -float("inf")
	try:
		kappa = (PA - PE) / (1 - PE)
		kappa = float("{:.4f}".format(kappa))
	except ZeroDivisionError:
		print("Expected agreement = 1")

	print("Fleiss' Kappa =", kappa)

	return matrix


print("Calculate pair-wise cohen's kappa:")
useragreement()

print("\nCalculate pair-wise cohen's kappa with binary annotations:")
useragreement(True)

print("\nCalcuate fleiss' kappa")
fleiss()

print("\nCalcuate fleiss' kappa with binary annotations:")
fleiss(True)

print("Evaluation scores of Expert-1\n")
calculateEValScores("originalAnno/expert-1.csv")

print("Evaluation scores of Expert-2\n")
calculateEValScores("originalAnno/expert-2.csv")

print("Evaluation scores of Non-expert-1\n")
calculateEValScores("originalAnno/non-expert-1.csv")

print("Evaluation scores of Non-expert-2\n")
calculateEValScores("originalAnno/non-expert-2.csv")