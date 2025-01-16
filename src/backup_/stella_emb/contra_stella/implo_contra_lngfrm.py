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
output_dir = "./nala"
push_to_hub = True
new_model_name = "nqlq"
checkpoint_dir = "./nqlq"
os.makedirs(checkpoint_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load Model and Tokenizer with `trust_remote_code=True`
vector_dim = 1024
vector_linear_directory = f"2_Dense_{vector_dim}"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
vector_linear = torch.nn.Linear(in_features=model.config.hidden_size, out_features=vector_dim)
vector_linear_dict = {
    k.replace("linear.", ""): v for k, v in
    torch.load(os.path.join("/beegfs/schubotz/.cache/huggingface/hub/models--dunzhang--stella_en_400M_v5/snapshots/24e2e1ffe95e95d807989938a5f3b8c18ee651f5", f"{vector_linear_directory}/pytorch_model.bin"),map_location = device).items()
}
vector_linear.load_state_dict(vector_linear_dict)
vector_linear.to(device)

# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, target):
        # Calculate cosine similarity between the two embeddings
        sim = torch.matmul(embedding1,embedding2.T)
        print("cos simil: ", type(sim))
        print("cos simil: ", sim.size())
        sys.exit(0)
        # Contrastive loss based on cosine similarity
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(sim,target)
        return loss

def read_results_file(file_path):
    getallfls = os.listdir(file_path)
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
                #if countDocs > 20:
                #    sys.exit(0)
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
        #print(i)
        #raise
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

# Preprocessing Function
def preprocess_function(examples):
    # Tokenize text
    inputs = tokenizer(examples, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}

# DataLoader
def collate_fn(batch):
    input_ids = torch.cat([item["input_ids"] for item in batch])
    attention_mask = torch.cat([item["attention_mask"] for item in batch])
    #labels = torch.tensor([item["label"] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask}

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-6)

# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
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
            batch_ids = pos_neg_trips[batch_start:batch_start+batch_size]
            print("batch_ids: ", batch_ids)
            batch_ids = [doc_id  for trip in batch_ids for doc_id in trip]
            print("test tit: ", batch_ids[::3])
            doc_strs = ["\nDescription: ".join(doc_text_dict.get(str(doc_id), "")) for doc_id in batch_ids]
            records = [preprocess_function(doc_str) for doc_str in doc_strs]
            batch = collate_fn(records)
            optimizer.zero_grad()
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # Generate embeddings
            last_hidden_state = model(input_ids=input_ids, attention_mask=attention_mask)[0]
            last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            embeddings = nn.functional.normalize(vector_linear(embeddings), p=2, dim=-1)
            print("embeddings: ", embeddings.size())
            # Split embeddings into pairs
            embeddings1 = embeddings[::3]
            print("embeddings1: ",embeddings1.size())
            embeddings2 = embeddings[1::3]
            print("embeddings2: ",embeddings2.size())
            embeddings3 = embeddings[2::3]
            print("embeddings3: ",embeddings3.size())
            embeddings_docs = torch.cat((embeddings2,embeddings3),0)
            print("embeddings_docs :", embeddings_docs.size())
            target = torch.tensor(range(len(embeddings1))).to(device)
            #compute accuracy
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
                #with open("loss_acc/epoch_{epoch+1}_batch_{batch_start//6}.txt",'w') as fh:
                #    fh.write("loss: \n"+str(cum_loss))
                #    fh.write("accuracy: \n"+str(cum_acc))
            #if batch_start<10*batch_size:
            #    try:
            #       torch.cuda.memory._dump_snapshot(f"mem_before_backward_{epoch}_{batch_start}.pickle")
            #    except Exception as e:
            #    except Exception as e:
            #       logger.error(f"Failed to capture memory snapshot {e}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            #if batch_start<10*batch_size:
            #
            #    try:
            #       torch.cuda.memory._dump_snapshot(f"mem_before_step_{epoch}_{batch_start}.pickle")
            #    except Exception as e:
            #       logger.error(f"Failed to capture memory snapshot {e}")
            optimizer.step()
            #if batch_start<10*batch_size:

             #  try:
             #      torch.cuda.memory._dump_snapshot(f"mem_after_step_{epoch}_{batch_start}.pickle")
             #   except Exception as e:
             #      logger.error(f"Failed to capture memory snapshot {e}")
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
