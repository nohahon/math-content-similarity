from src.models.transformer_model import Transformer
from src.models.contrastive_loss import ContrastiveLoss
from src.wandb_.wandb_client import WandbClient


from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
import pandas as pd

wandbc = WandbClient("test_small")
train_dataset = pd.read_csv(wandbc.load_dataset("final_train_dataset"))
dev_dataset = pd.read_csv(wandbc.load_dataset("final_dev_dataset"))

model_args = {
    "max_length": 512,
    "peft": False,
    "pooling_mode": "mean",
}
embedding_model = Transformer("BAAI/llm-embedder", model_args=model_args)
tokenized_train = embedding_model.tokenize(train_dataset)
tokenized_dev = embedding_model.tokenize(dev_dataset)

loss_model = ContrastiveLoss(embedding_model)

training_args = TrainingArguments(
    report_to="wandb",
    output_dir="./",
    overwrite_output_dir=True,
    per_device_eval_batch_size=8,
    per_device_train_batch_size=8,
    save_total_limit=3,
    num_train_epochs=5,
    do_eval=True,
    evaluation_strategy="steps",
    logging_steps=20,
    eval_steps=20,
    disable_tqdm=False,
    weight_decay=0.1,
    warmup_steps=1,
    learning_rate=4e-5,
    run_name="test_small",
    load_best_model_at_end=True,
)
trainer = Trainer(
    loss_model,
    training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    tokenizer=embedding_model.tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=embedding_model.tokenizer),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
trainer.train()
wandbc.finish()
