from src.models.transformer_model import Transformer
from src.models.contrastive_loss import ContrastiveLoss

from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import pandas as pd
from src.wandb_.wandb_client import WandbClient

wandbc = WandbClient("test")
train_dataset = pd.read_csv(wandbc.load_dataset("final_train_dataset"))
dev_dataset = pd.read_csv(wandbc.load_dataset("final_dev_dataset"))

model_args = {
    "max_length": 512,
    "peft": False,
    "pooling_mode": "mean",
}
embedding_model = Transformer("BAAI/llm-embedder")
tokenized_train = embedding_model.tokenize(train_dataset)
tokenized_dev = embedding_model.tokenize(dev_dataset)

loss_model = ContrastiveLoss(embedding_model)

training_args = TrainingArguments(
    report_to="wandb",
    output_dir="./",
    overwrite_output_dir=True,
    per_device_eval_batch_size=4,
    per_device_train_batch_size=4,
    num_train_epochs=2,
    do_eval=True,
    evaluation_strategy="steps",
    logging_steps=20,
    eval_steps=20,
    disable_tqdm=False,
    weight_decay=0.1,
    warmup_steps=1,
    learning_rate=4e-5,
    run_name="test",
)
trainer = Trainer(
    loss_model,
    training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    tokenizer=embedding_model.tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=embedding_model.tokenizer),
)
trainer.train()
wandbc.finish()
