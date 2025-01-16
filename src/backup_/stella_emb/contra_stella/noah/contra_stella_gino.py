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
from datasets import load_dataset, Dataset
from huggingface_hub import Repository
from llm2vec.loss import HardNegativeNLLLoss
import sys
sys.path.append('/beegfs/schubotz/ankit/data')
sys.path.append('/beegfs/schubotz/ankit/data/zbReviewCitData/')
sys.path.append('../../../tf_idf_algrthmn')
import zbCitData_st
from time import time
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
    SimilarityFunction
)
from sentence_transformers.losses import MultipleNegativesRankingLoss, CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator, InformationRetrievalEvaluator

# Configurations
model_name = "dunzhang/stella_en_400M_v5"
dataset_name = "AnkitSatpute/zbMath_contra_rand"
output_dir = "./trained_model_params"
push_to_hub = True
new_model_name = "contrastive-stella-embeddings"
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)


def load_auto_model():
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
    return model, tokenizer, vector_linear, vector_dim, device

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
                #if countDocs > 20:
                #    sys.exit(0)
    return genRecmnds

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

def construct_triplets_and_data(split='train'):
    #dataset = load_dataset(dataset_name)
    # Load Dataset
    data_ = "/beegfs/schubotz/ankit/data/zbReviewCitData/citation_dataset.csv"
    dataFr = zbCitData_st.load_csv_to_dataframe(data_)  # Load main dataset
    main_df = pd.read_csv('/beegfs/schubotz/ankit/data/zbmath_complete_cleaned.csv')[['document_id','title','text']].dropna()
    #main_df = zbCitData_st.getMainData(usenans=True)[['document_id','title','text']].dropna()
    train_df, test_df, valid_df = zbCitData_st.split_dataframe(dataFr)  # Split data into train, test, valid
    if split=='train':
        split_df = train_df
    elif split=='valid':
        split_df = valid_df
    else:
        split_df = test_df
    main_df["document_id"] = pd.to_numeric(main_df["document_id"], errors="coerce")
    main_df["document_id"] = main_df["document_id"].fillna(0).astype("string")
    train_citations = set(
        elem.strip()
        for citations in split_df['citation_de'].dropna()
        for elem in citations.split(';')
    )
    main_document_ids = set(main_df['document_id'])
    pos_docs_ = main_document_ids.intersection(train_citations)
    seed_docs_ = main_document_ids.intersection({str(doc_id) for doc_id in split_df['document_id'].tolist()})
    pos_seed_docs = pos_docs_.union(seed_docs_)
    pos_seed_docs = {str(doc) for doc in pos_seed_docs}
    pos_pairs = []
    for i,row in split_df[['document_id','citation_de']].iterrows():
        if str(row['document_id']) in seed_docs_:
            for citation in row['citation_de'].split('; '):
                if citation.strip() in pos_docs_:
                    pos_pairs.append((i,row['document_id'],citation.strip()))
    #get hard negatives:
    pos_neg_trips = []
    negs_file_path = f"/beegfs/schubotz/ankit/code/zbReviewCit/stella_emb/stella_base/data/base_/text_vs_text/scores/{split}/"
    hard_negs = read_results_file(negs_file_path)
    for i,seed_id,doc_id in pos_pairs:
        try:
            hard_negs_cands = hard_negs[str(seed_id)]
        except:
            #print(i)
            #raise
            continue
        hard_negs_cands = [cand for cand in hard_negs_cands if (cand in main_document_ids and not cand in pos_seed_docs)]
        pos_neg_trips.append( (seed_id,doc_id,hard_negs_cands[0]) )
        hard_negs_cands.pop(0)
        hard_negs[seed_id] = hard_negs_cands

    random.seed(42)
    random.shuffle(pos_neg_trips)
    print("Lenth of all citations is: ", len(pos_docs_))
    train_docs = set(split_df['document_id'])
    doc_text_dict = dict(zip(main_df['document_id'].astype(str), zip(main_df['title'], main_df['text'])))
    return pos_neg_trips, doc_text_dict

def construct_dataset(split='train',limit=0):
    pos_neg_trips, doc_text_dict = construct_triplets_and_data(split)
    if limit:
        pos_neg_trips = pos_neg_trips[:limit]
    seed_texts = ["\nDescription: ".join(doc_text_dict[str(seed_id)]) for seed_id, pos_id, neg_id in pos_neg_trips]
    pos_texts = ["\nDescription: ".join(doc_text_dict[str(pos_id)]) for seed_id, pos_id, neg_id in pos_neg_trips]
    neg_texts = ["\nDescription: ".join(doc_text_dict[str(neg_id)]) for seed_id, pos_id, neg_id in pos_neg_trips]
    dataset = Dataset.from_dict({"seed":seed_texts,"positive":pos_texts,"negative":neg_texts})
    return dataset

def construct_ir_evaluator(split='valid',msc='',used_data=['title','text']):
    from abstella_base import get_test_data, load_main_data
    main_data = load_main_data()
    #selects only seeds and docs that are in msc14.
    subset_df, main_data = get_test_data(main_data,split,msc)
    #create data according to desired format of InformationRetrievalEvaluator
    corpus_ids = main_data['document_id'].tolist()
    corpus_ids = [str(cid) for cid in corpus_ids]
    corpus_texts = list(main_data[used_data].itertuples(index=False,name=None))
    corpus_texts = ["\n".join([used_data[i]+": "+tup[i] for i in range(len(used_data))]) for tup in corpus_texts]
    corpus = dict(zip(corpus_ids,corpus_texts))
    query_ids = subset_df['document_id'].tolist()
    query_ids = [str(qid) for qid in query_ids if str(qid) in corpus_ids]
    query_texts = [corpus[qid] for qid in query_ids]
    queries = dict(zip(query_ids,query_texts))
    relevant_doc_strs = {str(qid):rel_str for qid,rel_str in subset_df.set_index('document_id')['citation_de'].to_dict().items() if str(qid) in query_ids}
    relevant_docs = {qid:[doc_id for doc_id in rel_str.split("; ") if doc_id in corpus_ids] for qid,rel_str in relevant_doc_strs.items()}
    relevant_docs = {qid:rids for qid,rids in relevant_docs.items() if rids}
    return InformationRetrievalEvaluator(queries=queries,corpus=corpus,relevant_docs=relevant_docs,name=f"zbCit{msc}-{split}")
    

def train_loop_sentence_transformers(batch_size=6):
    import wandb
    import time
    wandb.login(key='a073ecfea9d9468d0f28569e00fd9c582d288448')
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # 1. Load a model to finetune with 2. (Optional) model card data
    model = SentenceTransformer(
        "dunzhang/stella_en_400M_v5",trust_remote_code=True
    )
    # 3. Load a dataset to finetune on.
    train_dataset = construct_dataset() 
    #eval_dataset = construct_dataset('valid',limit=1000)

    # 4. Define a loss function.
    loss = CachedMultipleNegativesRankingLoss(model)

    # 5. Specify training arguments.
    args = SentenceTransformerTrainingArguments(
        output_dir=f'models/stella_400_5e6_{batch_size}_{timestr}',
        log_level='debug',
        num_train_epochs=4,
        #at the risk of much higher running times, we might want to check whether this causes degrading performance.
        bf16=True,
        #can maybe be deactivated to ensure consistent batch size, but risks memory errors.
        #auto_find_batch_size=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        max_grad_norm=0.5,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        evaluation_strategy="epoch",
        #because the evaluation takes some time (but then only 4 minutes, actually) 
        #eval_steps=0.125,
        save_strategy="epoch",
        #meaning 1/0.125=8 total saved checkpoints. 
        #save_steps=0.125,
        #currently irrelevant, because only 8 checkpoints are saved. There's also an option to make sure that the highest
        #performing model is saved, but the default for highest performing 
        save_total_limit=10,
        logging_steps=3,
        run_name=f'models/stella_400_5e6_{batch_size}_{timestr}'
    )
    # 6. Create an evaluator & evaluate on the base model
    #dev_evaluator = TripletEvaluator(
    #    anchors=eval_dataset["seed"],
    #    positives=eval_dataset["positive"],
    #    negatives=eval_dataset["negative"],
    #    main_distance_function=SimilarityFunction.COSINE,
    #    name="cit-data-dev",
    #)
    dev_evaluator = construct_ir_evaluator(split='valid',msc='14',used_data=['title','text'])
    dev_evaluator(model)

    # 7. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model = model,
        args = args,
        train_dataset = train_dataset,
        loss = loss,
        evaluator = dev_evaluator)
    trainer.train()

    # 8. Save the trained model
    model.save_pretrained(f"models/stella_400_5e6_{batch_size}_{timestr}/final") 

# Preprocessing Function
def preprocess_function(examples):
    # Tokenize text
    inputs = tokenizer(examples, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}

# Preprocess Dataset
#processed_dataset = dataset.map(preprocess_function, batched=True)

# DataLoader
def collate_fn(batch):
    input_ids = torch.cat([item["input_ids"] for item in batch])
    attention_mask = torch.cat([item["attention_mask"] for item in batch])
    #labels = torch.tensor([item["label"] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask}

#train_dataloader = DataLoader(processed_dataset["train"], batch_size=16, shuffle=True, collate_fn=collate_fn)

#ignore
def train_loop(batch_size=6):
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-6)

    # Training Loop
    model.to(device)
    contrastive_loss = ContrastiveLoss()
    MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100000
    torch.cuda.memory._record_memory_history(
       max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
    )

    for epoch in range(5):  # Number of epochs
        model.train()
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        loss_acc = []
        #try:
        #    torch.cuda.memory._dump_snapshot(f"mem_{epoch}.pickle")
        #except Exception as e:
        #    logger.error(f"Failed to capture memory snapshot {e}")
        #batch consists of 8 seeds, 8 positive docs and 8 hard negatives.
        #for each seed we will in the end have 1 positive and 15 negatives.
        for batch_start in range(0,len(pos_neg_trips),batch_size):
            #print(batch_start)
            batch_ids = pos_neg_trips[batch_start:batch_start+batch_size]
            batch_ids = [doc_id  for trip in batch_ids for doc_id in trip]
            doc_strs = ["\nDescription: ".join(doc_text_dict.get(str(doc_id), "")) for doc_id in batch_ids]
            records = [preprocess_function(doc_str) for doc_str in doc_strs]
            batch = collate_fn(records)
            optimizer.zero_grad()
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            #if batch_start<10*batch_size:
                #try:
                #   torch.cuda.memory._dump_snapshot(f"mem_before_embedding_{epoch}_{batch_start}.pickle")
                #except Exception as e:
                #   logger.error(f"Failed to capture memory snapshot {e}")
                
            # Generate embeddings
            last_hidden_state = model(input_ids=input_ids, attention_mask=attention_mask)[0]
            #if batch_start<10*batch_size:
                #try:
                #   torch.cuda.memory._dump_snapshot(f"mem_after_last_hidden_{epoch}_{batch_start}.pickle")
                #except Exception as e:
                #   logger.error(f"Failed to capture memory snapshot {e}")
            last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            embeddings = nn.functional.normalize(vector_linear(embeddings), p=2, dim=-1)
            #embeddings = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
            # Split embeddings into pairs
            embeddings1 = embeddings[::3]
            embeddings2 = embeddings[1::3]
            embeddings3 = embeddings[2::3]
            embeddings_docs = torch.cat((embeddings2,embeddings3),0)
            target = torch.tensor(range(len(embeddings1))).to(device)
            #labels = torch.tensor([[1 if j  == target[i] else 0 for j in range(len(embeddings_docs))]
            #                        for i in range(len(embeddings1))]).to(device)
            #compute accuracy
            sim_embs = torch.matmul(embeddings1,embeddings_docs.T)
            #predictions = (sim_embs > 0.85).long().to(device) #assumming most dissimilar samples have similarity scores
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
                #with open("loss_acc/epoch_{epoch+1}_batch_{batch_start//6}.txt",'w') as fh:
                #    fh.write("loss: \n"+str(cum_loss))
                #    fh.write("accuracy: \n"+str(cum_acc))
            #if batch_start<10*batch_size:
            #    try:
            #       torch.cuda.memory._dump_snapshot(f"mem_before_backward_{epoch}_{batch_start}.pickle")
            #    except Exception as e:
            #       logger.error(f"Failed to capture memory snapshot {e}")
            loss.backward()
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

def test_senttrans_vs_trans(model_path,batch_size=6):
    from sentence_transformers import SentenceTransformer
    import faiss
    model_s = SentenceTransformer(model_path, device=device, trust_remote_code=True)
    model_t = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    corr_predictions_t = 0
    corr_predictions_s = 0
    for batch_start in range(0,batch_size*300,batch_size):
        #print(batch_start)
        batch_ids = pos_neg_trips[batch_start:batch_start+batch_size]
        batch_ids = [doc_id  for trip in batch_ids for doc_id in trip]
        doc_strs = ["\nDescription: ".join(doc_text_dict.get(str(doc_id), "")) for doc_id in batch_ids]
        records = [preprocess_function(doc_str) for doc_str in doc_strs]
        batch = collate_fn(records)
        # Move data to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        #transf
        last_hidden_state = model_t(input_ids=input_ids, attention_mask=attention_mask)[0]
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        embeddings_t = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        embeddings_t = nn.functional.normalize(vector_linear(embeddings_t), p=2, dim=-1)
        # Split embeddings into pairs
        embeddings1_t = embeddings_t[::3]
        embeddings2_t = embeddings_t[1::3]
        embeddings3_t = embeddings_t[2::3]
        embeddings_docs_t = torch.cat((embeddings2_t,embeddings3_t),0)
        target = torch.tensor(range(len(embeddings1_t))).to(device)
        sim_embs_t = torch.matmul(embeddings1_t,embeddings_docs_t.T)
        predictions_t = torch.tensor([argmax(row) for row in sim_embs_t]).to(device)
        correct_predictions_curr_t = (predictions_t == target).sum().item()
        corr_predictions_t += correct_predictions_curr_t
        #senttrans
        embeddings_s = model_s.encode(doc_strs, convert_to_tensor=True, device=device, normalize_embeddings=True)
        #embeddings_s = faiss.normalize_L2(embeddings_s)
        embeddings1_s = embeddings_s[::3]
        embeddings2_s = embeddings_s[1::3]
        embeddings3_s = embeddings_s[2::3]
        embeddings_docs_s = torch.cat((embeddings2_s,embeddings3_s),0)
        sim_embs_s = torch.matmul(embeddings1_s,embeddings_docs_s.T)
        predictions_s = torch.tensor([argmax(row) for row in sim_embs_s]).to(device)
        correct_predictions_curr_s = (predictions_s == target).sum().item()
        corr_predictions_s += correct_predictions_curr_s
    return corr_predictions_t, corr_predictions_s

if __name__ == "__main__":
    train_loop_sentence_transformers(batch_size=512)
