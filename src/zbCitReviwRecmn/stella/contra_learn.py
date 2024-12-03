import os
import sys
import torch
import random
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
sys.path.append('../../tf_idf_algrthmn/')
import zbCitData_st
from collections import defaultdict
from itertools import combinations
from transformers import AutoModel, AutoTokenizer, AdamW
from datasets import load_dataset
from huggingface_hub import Repository

# Configurations
model_name = "dunzhang/stella_en_400M_v5"
output_dir = "./trained_model_params"
push_to_hub = True
new_model_name = "contrastive-stella-embeddings"
checkpoint_dir = "./chckpnts_stle_rand"
os.makedirs(checkpoint_dir, exist_ok=True)

# Load Model and Tokenizer with `trust_remote_code=True`
vector_dim = 1024
vector_linear_directory = f"2_Dense_{vector_dim}"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
vector_linear = torch.nn.Linear(in_features=model.config.hidden_size, out_features=vector_dim)
vector_linear_dict = {
    k.replace("linear.", ""): v for k, v in
    torch.load(os.path.join("/beegfs/schubotz/.cache/huggingface/hub/models--dunzhang--stella_en_400M_v5/snapshots/24e2e1ffe95e95d807989938a5f3b8c18ee651f5", f"{vector_linear_directory}/pytorch_model.bin")).items()
}
vector_linear.load_state_dict(vector_linear_dict)
vector_linear.cuda()

# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, label):
        embedding1_norm = nn.functional.normalize(embedding1, p=2, dim=-1)
        embedding2_norm = nn.functional.normalize(embedding2, p=2, dim=-1)
        # Calculate cosine similarity between the two embeddings
        cosine_similarity = torch.sum(embedding1_norm * embedding2_norm, dim=-1)
        # Contrastive loss based on cosine similarity
        loss = torch.mean((1 - label) * torch.pow(torch.clamp(cosine_similarity, min=-1.0), 2) +
                          label * torch.pow(torch.clamp(1 - cosine_similarity, min=0.0), 2))
        return loss

# Load Dataset
data_ = "/beegfs/schubotz/ankit/data/zbReviewCitData/citation_dataset.csv"
dataFr = zbCitData_st.load_csv_to_dataframe(data_)  # Load main dataset
main_df = zbCitData_st.getMainData()
train_df, test_, valid_ = zbCitData_st.split_dataframe(dataFr)  # Split data into train, test, valid
main_df["document_id"] = pd.to_numeric(main_df["document_id"], errors="coerce")
main_df["document_id"] = main_df["document_id"].fillna(0).astype("string")
train_citations = set(
    elem.strip()
    for citations in train_df['citation_de'].dropna()
    for elem in citations.split(';')
)
main_document_ids = set(main_df['document_id'])
dictSamps = defaultdict(lambda: 0)
pos_docs_ = main_document_ids.intersection(train_citations)
print("Lenth of all citations is: ", len(pos_docs_))
train_docs = set(train_df['document_id'])
setForNegDocs = train_docs.union(train_citations)
setForNegDocs = main_document_ids - setForNegDocs # ensuring there are no doc ids from training set
setForNegDocs = list(setForNegDocs)
doc_text_dict = dict(zip(main_df['document_id'].astype(str), main_df['text']))

# Preprocessing Function
def preprocess_function(examples, labels):
    # Tokenize text
    inputs = tokenizer(examples, padding="longest", truncation=True, max_length=512, return_tensors="pt")
    return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "label": labels}

# Preprocess Dataset
#processed_dataset = dataset.map(preprocess_function, batched=True)

# DataLoader
def collate_fn(batch):
    input_ids = torch.cat([item["input_ids"] for item in batch], dim=0)
    attention_mask = torch.cat([item["attention_mask"] for item in batch], dim=0)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

#train_dataloader = DataLoader(processed_dataset["train"], batch_size=16, shuffle=True, collate_fn=collate_fn)
# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
contrastive_loss = ContrastiveLoss()

for epoch in range(5):  # Number of epochs
    model.train()
    epoch_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for index, row in train_df.iterrows():
        records, labels = [], []
        docs_pos = list()
        docs_pos.append(row['document_id'])
        docs_pos += [eles_ for eles_ in row['citation_de'].split(';') if eles_ in pos_docs_]
        pos_combs = list(combinations(docs_pos, 2))
        if len(pos_combs) > 0:
            random_elements = random.sample(setForNegDocs, len(pos_combs))
            random_elements.append(row['document_id'])
            neg_combs = list(combinations(random_elements ,2))
            for eachComb in pos_combs + neg_combs:
                doc1_str = doc_text_dict.get(str(eachComb[0]), "")
                doc2_str = doc_text_dict.get(str(eachComb[1]), "")
                label = [1 if eachComb in pos_combs else 0]
                records.append(preprocess_function(doc1_str + doc2_str, label))
            batch = collate_fn(records)
            optimizer.zero_grad()
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            # Generate embeddings
            last_hidden_state = model(input_ids=input_ids, attention_mask=attention_mask)[0]
            last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            embeddings = nn.functional.normalize(vector_linear(embeddings), p=2, dim=-1)
            #embeddings = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
            # Split embeddings into pairs
            embeddings1 = embeddings[::2]
            embeddings2 = embeddings[1::2]
            labels = labels[::2]
            #compute accurracy
            cosine_similarity = torch.sum(embeddings1 * embeddings2, dim=-1)
            predictions = (cosine_similarity > 0.8).long() #assumming most dissimilar samples have similarity scores
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            # Compute contrastive loss

            loss = contrastive_loss(embeddings1, embeddings2, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            #if batch_idx < 5:
            #    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{batch_idx + 1}.pt")
            #    torch.save({
            #        'epoch': batch_idx,
            #        'model_state_dict': model.state_dict(),
            #        'optimizer_state_dict': optimizer.state_dict(),
            #    }, checkpoint_path)
            #    print(f"Checkpoint saved at {checkpoint_path}")
    #save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch + 1}_.pt")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
    accuracy = correct_predictions / total_predictions
    print(f"Epoch {epoch + 1}: Loss = {epoch_loss / len(train_dataloader)}, Accuracy = {accuracy * 100:.2f}%")

# Save and Push Model to Hugging Face Hub
if push_to_hub:
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    repo = Repository(local_dir=output_dir, clone_from=new_model_name)
    repo.push_to_hub(commit_message="Initial commit for contrastive trained model")

print("Training complete and model pushed to Hugging Face Hub!")
