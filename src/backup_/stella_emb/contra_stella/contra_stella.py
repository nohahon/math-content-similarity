import wandb
import time
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
sys.path.append('../../tf_idf_algrthmn')
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
    subset_df, main_data = get_test_data(main_data,split,msc)
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

sweep_config = {
    "method": "grid",
    "name": "second-grid-sweep",
    "metric": {
        "goal": "maximize",
        "name": "eval/recall@10"
    },
    "parameters": {
        "epochs": {
            "values": [1]
        },
        "batch_size": {
            "values": [2048]
        },
        "learning_rate": {
            "values": [5e-5]
        },
        "weight_decay": {
            "values": [0.01]
        }
    }
}

wandb.login(key='a073ecfea9d9468d0f28569e00fd9c582d288448')

def train_loop_sentence_transformers(config=None):
    with wandb.init(config=config):
        config = wandb.config
    # 1. Load a model to finetune with 2. (Optional) model card data
    model = SentenceTransformer(
        "dunzhang/stella_en_400M_v5",trust_remote_code=True
    )
    # 3. Load a dataset to finetune on.
    train_dataset = construct_dataset() 
    # 4. Define a loss function.
    loss = CachedMultipleNegativesRankingLoss(model)
    # 5. Specify training arguments.
    args = SentenceTransformerTrainingArguments(
        output_dir=f'models/stella_400_secondsweep',
        log_level='debug',
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        bf16=True,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        run_name=f'models/stella_400_secondsweep'
    )
    # 6. Evaluator
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
    model.save_pretrained(f"models/stella_400_secondsweep") 

if __name__ == "__main__":
    os.environ["WANDB_CONSOLE"] = "off"
    sweep_id = wandb.sweep(sweep_config, project='second-sweep')
    wandb.agent(sweep_id, train_loop_sentence_transformers, count=50)

