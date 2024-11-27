import os
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
checkpoint_dir = "./checkpoints"
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
dataset = load_dataset(dataset_name)

# Preprocessing Function
def preprocess_function(examples):
    # Tokenize text
    inputs = tokenizer(examples["text"], padding="longest", truncation=True, max_length=512, return_tensors="pt")
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
    correct_predictions = 0
    total_predictions = 0
    for batch_idx,batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        # Move data to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        # Generate embeddings
        last_hidden_state = model(**input_ids)[0]
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
        predictions = (cosine_similarity > 0).long()
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)
        # Compute contrastive loss
        loss = contrastive_loss(embeddings1, embeddings2, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        #save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch + 1}_batch{batch_idx + 1}.pt")
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
