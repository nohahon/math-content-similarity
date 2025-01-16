import os
import sys
import json
import random
import wandb
wandb.login(key='a073ecfea9d9468d0f28569e00fd9c582d288448')
import torch
import torch.nn.functional as F
import pandas as pd
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments, BatchSamplers
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import TrainingArguments
sys.path.append('../../tf_idf_algrthmn')
import zbCitData_st
from sklearn.metrics import accuracy_score

# Function to compute cosine similarity and classify based on threshold
def compute_metrics(p):
    """Compute accuracy for contrastive examples using cosine similarity."""
    predictions, labels = p
    preds = []
    for i in range(len(predictions)):
        # Extract seed and recommended text from the dataset
        seed_text = predictions[i][0]
        recmnd_text = predictions[i][1]  
        seed_emb = model.encode([seed_text], convert_to_tensor=True, device=device, normalize_embeddings=True)
        recmnd_emb = model.encode([recmnd_text], convert_to_tensor=True, device=device, normalize_embeddings=True)
        # Compute cosine similarity between seed and recommended text embeddings
        cos_sim = F.cosine_similarity(seed_emb, recmnd_emb, dim=-1)
        # If cosine similarity is above the threshold, consider them similar (label = 1)
        if cos_sim >= 0.8:
            predicted_label = 1
        else:
            predicted_label = 0
        preds.append(predicted_label)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# Read the recommendation results file to extract hard negatives
def read_results_file(file_path):
    getallfls = os.listdir(file_path)
    genRecmnds = {}
    for eachF in getallfls:
        print(f"Processing file: {eachF}")
        with open(file_path + eachF, 'r') as json_file:
            data = json.load(json_file)
            try:
                list_true = isinstance(data[list(data.keys())[0]][0], list) if isinstance(data, dict) else False
            except:
                continue
        for doc_id, rec_list in data.items():
            genRecmnds[doc_id] = [str(e[0]) if list_true else str(e) for e in rec_list[:60]]
    return genRecmnds

def construct_triplets_and_data(split='train'):
    # Load Dataset
    data_ = "/beegfs/schubotz/ankit/data/zbReviewCitData/citation_dataset.csv"
    dataFr = zbCitData_st.load_csv_to_dataframe(data_)
    main_df = zbCitData_st.getMainData(usenans=True)[['document_id', 'title', 'text']].dropna()
    train_df, test_, valid_ = zbCitData_st.split_dataframe(dataFr)
    if split=='train':
        split_df = train_df
    elif split=='valid':
        split_df = valid_
        #print("I am in Valid split")
    else:
        split_df = test_
    #print("size of valid: ", split_df.shape)
    main_df["document_id"] = pd.to_numeric(main_df["document_id"], errors="coerce")
    main_df["document_id"] = main_df["document_id"].fillna(0).astype("string")
    train_citations = set(
        elem.strip()
        for citations in split_df['citation_de'].dropna()
        for elem in citations.split(';')
    )
    #print("Number of citations: ", len(train_citations))
    main_document_ids = set(main_df['document_id'])
    pos_docs_ = main_document_ids.intersection(train_citations)
    seed_docs_ = main_document_ids.intersection({str(doc_id) for doc_id in split_df['document_id'].tolist()})
    #print("seed docs string :", len(seed_docs_))
    pos_seed_docs = pos_docs_.union(seed_docs_)
    pos_seed_docs = {str(doc) for doc in pos_seed_docs}
    # Generate positive pairs
    pos_pairs = []
    for i, row in train_df[['document_id', 'citation_de']].iterrows():
        #print(str(row['document_id']), type(list(seed_docs_)[0])) #error here for Valid
        #sys.exit(0)
        if str(row['document_id']) in seed_docs_:
            for citation in row['citation_de'].split('; '):
                #print(type(citation), type(list(pos_docs_)[0]))
                #sys.exit(0)
                if citation.strip() in pos_docs_:
                    pos_pairs.append((i, row['document_id'], citation.strip()))
    #print("positive pairs: ", len(pos_pairs))
    # Read hard negatives from a results file
    hard_negs = read_results_file(f"/beegfs/schubotz/ankit/code/zbReviewCit/stella_emb/stella_base/data/base_/text_vs_text/scores/{split}/")
    pos_neg_trips = []
    #print("totla hard negatives got from valid: ", len(hard_negs))
    for i, seed_id, doc_id in pos_pairs:
        try:
            hard_negs_cands = hard_negs[str(seed_id)]
        except:
            continue
        hard_negs_cands = [cand for cand in hard_negs_cands if (cand in main_document_ids and not cand in pos_seed_docs)]
        pos_neg_trips.append((seed_id, doc_id, hard_negs_cands[0]))
        hard_negs_cands.pop(0)
        hard_negs[seed_id] = hard_negs_cands
    #print(len(hard_negs), "hard negatives: ")
    random.seed(42)
    random.shuffle(pos_neg_trips)
    #print("Lenth of all citations is: ", len(pos_docs_))
    train_docs = set(train_df['document_id'])
    doc_text_dict = dict(zip(main_df['document_id'].astype(str), zip(main_df['title'], main_df['text'])))
    #print("len of ps and neg samps and string: ", len(pos_neg_trips), len(doc_text_dict))
    #sys.exit(0)
    return pos_neg_trips, doc_text_dict

# Prepare the contrastive loss dataset
# Maybe we should add more hard negatives, easy negatives directly here
def construct_dataset(split='train', limit=0):
    pos_neg_trips, doc_text_dict = construct_triplets_and_data(split)
    contrastive_examples = []
    for seed_id, doc_id, hard_neg in pos_neg_trips:
        positive_text = doc_text_dict.get(str(doc_id), "") 
        negative_text = doc_text_dict.get(str(hard_neg), "")
        query_text = doc_text_dict.get(str(seed_id), "") 
        contrastive_examples.append({"seed": query_text, "recmnd": positive_text, "label": 1})
        contrastive_examples.append({"seed": query_text, "recmnd": negative_text, "label": 0})
    print("Len of samps: ", len(contrastive_examples))
    cleaned_contrastive_examples = [
        {"seed": ex["seed"], "recmnd": ex["recmnd"], "label": ex["label"]}
        for ex in contrastive_examples
        if ex["seed"] and ex["recmnd"] and ex["label"] is not None
    ]
    print("Len of samps after cleaning: ", len(cleaned_contrastive_examples))
    #cleaned_contrastive_examples = cleaned_contrastive_examples[:100]
    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_dict({"seed": [ex["seed"] for ex in cleaned_contrastive_examples],
                                   "recmnd": [ex["recmnd"] for ex in cleaned_contrastive_examples],
                                   "label": [ex["label"] for ex in cleaned_contrastive_examples]})
    return train_dataset

model_name = "dunzhang/stella_en_400M_v5"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(model_name, device=device, trust_remote_code=True)

# Define the loss function (Contrastive Loss)
loss = losses.ContrastiveLoss(model)

#training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="contra-stella-pos1negAb",
    # Optional training parameters:
    log_level='debug',
    num_train_epochs=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_ratio=0.1,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    fp16=True,  # Set to False if your GPU can't handle FP16
    bf16=False,  # Set to True if your GPU supports BF16
    # Optional tracking/debugging parameters:
    evaluation_strategy="steps",
    eval_steps=1000,
    save_strategy="epoch",
    logging_steps=1000,
    run_name="stella-contra-pos1negAb",
)

train_dataset = construct_dataset()
eval_dataset = construct_dataset('test', limit=10000)

#print(type(eval_dataset["seed"]))
#print(len(eval_dataset["seed"]))
#sys.exit(0)

# Initialize the evaluator
binary_acc_evaluator = BinaryClassificationEvaluator(
    sentences1=eval_dataset["seed"],
    sentences2=eval_dataset["recmnd"],
    labels=eval_dataset["label"],
    name="zbreview-cit",
)

eval_res = binary_acc_evaluator(model)
print("Before training eval: ", eval_res)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    compute_metrics=compute_metrics,
)

eval_res = binary_acc_evaluator(model)
print("After training eval: ", eval_res)

trainer.train()
model.save_pretrained("stella-400-Contra-Pos1HardNegAb")
model.push_to_hub("stella-400-Contra-Pos1HardNegAb")

