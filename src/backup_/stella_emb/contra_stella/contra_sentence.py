import os
import pandas as pd
import numpy as np
import random
import pickle
from math import prod
import torch
from torch import nn
from torch.utils.data import DataLoader
import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModel, AutoTokenizer, AdamW
from datasets import load_dataset
from huggingface_hub import Repository
from llm2vec.loss import HardNegativeNLLLoss
import sys
sys.path.append('/beegfs/schubotz/ankit/data')
sys.path.append('/beegfs/schubotz/ankit/data/zbReviewCitData/')
sys.path.append('../../tf_idf_algrthmn')
import zbCitData_st
from time import time

# Configurations
model_name = "dunzhang/stella_en_400M_v5"
output_dir = "./trained_model_sent"
push_to_hub = True
new_model_name = "contrastive-stella-stent"
checkpoint_dir = "./checkpoints_sent"
os.makedirs(checkpoint_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load Model and Tokenizer with `trust_remote_code=True`
model = SentenceTransformer(model_name, device=device, trust_remote_code=True)

for param in model.parameters():
    param.requires_grad = True

# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, target):
        # Calculate cosine similarity between the two embeddings
        #cosine_similarity = nn.CosineSimilarity(dim=-1)
        sim = torch.matmul(embedding1,embedding2.T)
        # Contrastive loss based on cosine similarity
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(sim,target)
        return loss

def read_results_file(file_path):
    getallfls = os.listdir(file_path)
    print(getallfls)
    genRecmnds = {}
    seeddocids = list()
    countDocs = 0
    for i,eachF in enumerate(getallfls):
        print(eachF)
        with open(file_path+eachF, 'r') as json_file:
            # Load the content of the file into a Python dictionary
            data = json.load(json_file)
            try:
                list_true = isinstance(data[list(data.keys())[0]][0],list)
            except:
                list_true = isinstance(data[list(data.keys())[1]][0],list)
        for eackDoc in data.keys():
            countDocs += 1
            seeddocids.append(eackDoc)
            if list_true:
                gen_recmnds = [str(ea_[0]) for ea_ in data[eackDoc][:60]]
            else:
                gen_recmnds = [str(ea_) for ea_ in data[eackDoc][:60]]
            genRecmnds[eackDoc] = gen_recmnds
    return genRecmnds

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

# Load Dataset
data_ = "/beegfs/schubotz/ankit/data/zbReviewCitData/citation_dataset.csv"
dataFr = zbCitData_st.load_csv_to_dataframe(data_)  # Load main dataset
main_df = zbCitData_st.getMainData(usenans=True)[['document_id','title','text']].dropna()
train_df, test_, valid_ = zbCitData_st.split_dataframe(dataFr)  # Split data into train, test, valid
main_df["document_id"] = pd.to_numeric(main_df["document_id"], errors="coerce")
main_df["document_id"] = main_df["document_id"].fillna(0).astype("string")
train_citations = set(
    elem.strip()
    for citations in train_df['citation_de'].dropna()
    for elem in citations.split(';')
)
main_document_ids = set(main_df['document_id'])
pos_docs_ = main_document_ids.intersection(train_citations)
seed_docs_ = main_document_ids.intersection({str(doc_id) for doc_id in train_df['document_id'].tolist()})
pos_seed_docs = pos_docs_.union(seed_docs_)
pos_seed_docs = {str(doc) for doc in pos_seed_docs}
pos_pairs = []
for i,row in train_df[['document_id','citation_de']].iterrows():
    if str(row['document_id']) in seed_docs_:
        for citation in row['citation_de'].split('; '):
            if citation.strip() in pos_docs_:
                pos_pairs.append((i,row['document_id'],citation.strip()))
#get hard negatives:
pos_neg_trips = []
negs_file_path = "/beegfs/schubotz/ankit/code/zbReviewCit/stella_emb/stella_base/data/base_/extended_keywords_vs_extended_keywords/scores_train/"
hard_negs = read_results_file(negs_file_path)
for i,seed_id,doc_id in pos_pairs:
    try:
        hard_negs_cands = hard_negs[str(seed_id)]
    except:
        continue
    hard_negs_cands = [cand for cand in hard_negs_cands if not cand in pos_seed_docs]
    pos_neg_trips.append( (seed_id,doc_id,hard_negs_cands[0]) )
    hard_negs_cands.pop(0)
    hard_negs[seed_id] = hard_negs_cands

random.seed(42)
random.shuffle(pos_neg_trips)
print("Lenth of all citations is: ", len(pos_docs_))
train_docs = set(train_df['document_id'])
doc_text_dict = dict(zip(main_df['document_id'].astype(str), zip(main_df['title'], main_df['text'])))

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-6)

# Training Loop
contrastive_loss = ContrastiveLoss()
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100000
torch.cuda.memory._record_memory_history(
   max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
)

def train_loop(batch_size=6):
    for epoch in range(5):  # Number of epochs
        model.train()
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        loss_acc = []
        for batch_start in range(0,len(pos_neg_trips),batch_size):
            #print(batch_start)
            batch_ids = pos_neg_trips[batch_start:batch_start+batch_size]
            batch_ids = [doc_id  for trip in batch_ids for doc_id in trip]
            optimizer.zero_grad()
            doc_strs = ["\nDescription: ".join(doc_text_dict.get(str(doc_id), "")) for doc_id in batch_ids]
            embeddings = model.encode(doc_strs, convert_to_tensor=True, device=device, normalize_embeddings=True)
            # Split embeddings into pairs
            embeddings1 = embeddings[::3]
            embeddings2 = embeddings[1::3]
            embeddings3 = embeddings[2::3]
            embeddings_docs = torch.cat((embeddings2,embeddings3),0)
            target = torch.tensor(range(len(embeddings1))).to(device)
            sim_embs = torch.matmul(embeddings1,embeddings_docs.T)
            predictions = torch.tensor([argmax(row) for row in sim_embs]).to(device)
            correct_predictions_curr = (predictions == target).sum().item()
            correct_predictions += correct_predictions_curr
            accuracy_curr = correct_predictions_curr/prod(target.size())
            total_predictions += prod(target.size())
            # Compute contrastive loss
            loss = contrastive_loss(embeddings1, embeddings_docs, target)
            loss_acc.append((accuracy_curr, loss))
            if batch_start<20*batch_size:
                print("batch_ids: \n", batch_ids,flush=True)
                print("sim_embs: \n",sim_embs,flush=True)
                print("predictions: \n",predictions,flush=True)
                print("target: \n", target,flush=True)
                print("correct_predictions: \n", correct_predictions,flush=True)
                print("total_predictions: \n", total_predictions,flush=True)
            if batch_start%(50*batch_size)==0:
                print(batch_start)
                cum_loss = sum([la[1]  for la in loss_acc[-50:]])/50
                print("loss: \n"+str(cum_loss),flush=True)
                cum_acc = sum([la[0]  for la in loss_acc[-50:]])/50
                print("accuracy: \n"+str(cum_acc),flush=True)
                print("batch_ids: \n", batch_ids,flush=True)
                print("sim_embs: \n",sim_embs,flush=True)
                if cum_loss>2.4:
                    print("model fucked")
                    sys.exit(1)
            if batch_start%(500*batch_size)==0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_step_{batch_start//batch_size+1}.pt")
                os.makedirs(checkpoint_path, exist_ok=True)
                model.save_pretrained(checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if batch_start < 5*batch_size:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{batch_start + 1}.pt")
                os.makedirs(checkpoint_path, exist_ok=True)
                model.save_pretrained(checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")
        #save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch + 1}.pt")
        os.makedirs(checkpoint_path, exist_ok=True)
        model.save_pretrained(checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")    
        accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch + 1}: Loss = {epoch_loss / (len(pos_neg_trips)//batch_size)}, Accuracy = {accuracy * 100:.2f}%")
        with open(f'loss_acc_{epoch + 1}.pkl','wb') as fh:
            pickle.dump(loss_acc,fh)
        #torch.cuda.memory._record_memory_history(enabled=None)
    # Save and Push Model to Hugging Face Hub
    if push_to_hub:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        repo = Repository(local_dir=output_dir, clone_from=new_model_name)
        repo.push_to_hub(commit_message="Initial commit for contrastive trained model")

    print("Training complete and model pushed to Hugging Face Hub!")

if __name__ == "__main__":
    train_loop()
