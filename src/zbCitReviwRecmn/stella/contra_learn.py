import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AdamW
from datasets import load_dataset
from huggingface_hub import Repository

# Configurations
model_name = "dunzhang/stella_en_400M_v5"
dataset_name = "AnkitSatpute/zbMath_contra_rand"
output_dir = "./trained_model_params"
push_to_hub = True
new_model_name = "contrastive-stella-embeddings"

# Load Model and Tokenizer with `trust_remote_code=True`
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, label):
        # Calculate pairwise distance
        euclidean_distance = torch.norm(embedding1 - embedding2, dim=-1)
        # Contrastive loss
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# Load Dataset
dataset = load_dataset(dataset_name)
# Preprocessing Function
def preprocess_function(examples):
    # Tokenize text
    inputs = tokenizer(examples["text"])
    return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "label": examples["label"]}

# Preprocess Dataset
processed_dataset = dataset.map(preprocess_function, batched=True)

# DataLoader
def collate_fn(batch):
    input_ids = torch.tensor([item["input_ids"] for item in batch])
    attention_mask = torch.tensor([item["attention_mask"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

train_dataloader = DataLoader(processed_dataset["train"], batch_size=32, shuffle=True, collate_fn=collate_fn)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
contrastive_loss = ContrastiveLoss()

for epoch in range(10):  # Number of epochs
    model.train()
    epoch_loss = 0.0
    for batch in train_dataloader:
        optimizer.zero_grad()
        # Move data to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        # Generate embeddings
        embeddings = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        # Split embeddings into pairs
        embeddings1 = embeddings[::2]
        embeddings2 = embeddings[1::2]
        labels = labels[::2]
        # Compute contrastive loss
        loss = contrastive_loss(embeddings1, embeddings2, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}: Loss = {epoch_loss / len(train_dataloader)}")

# Save and Push Model to Hugging Face Hub
if push_to_hub:
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    repo = Repository(local_dir=output_dir, clone_from=new_model_name)
    repo.push_to_hub(commit_message="Initial commit for contrastive trained model")

print("Training complete and model pushed to Hugging Face Hub!")
