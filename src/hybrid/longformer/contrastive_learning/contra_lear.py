import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import LongformerTokenizer
from datasets import load_dataset, DatasetDict
from transformers import Trainer, TrainingArguments
from transformers import LongformerModel, LongformerConfig

def loaddataset(dataset_dir):
    """Load dataset from Huggingface website """
    return load_dataset(dataset_dir, download_mode="force_redownload")

def tokenize_function(examples):
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=4096)

class ContrastiveLongformer(nn.Module):
    def __init__(self):
        super(ContrastiveLongformer, self).__init__()
        config = LongformerConfig.from_pretrained('allenai/longformer-base-4096')
        self.longformer = LongformerModel(config)

    def forward(self, input_ids, attention_mask):
        outputs = self.longformer(input_ids, attention_mask=attention_mask)
        return outputs.pooler_output  # Or outputs.last_hidden_state[:, 0]


def train():
    """Train model with contrastive learning and push to hub """
    datasetonHF = "AnkitSatpute/zbMathCombinedSamps"
    dataset = loaddataset(datasetonHF)
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    model = ContrastiveLongformer()
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )
    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.push_to_hub("AnkitSatpute/LngfrmrContraMod")

train()