from src.models.transformer_model import Transformer
from src.models.contrastive_loss import ContrastiveLoss

from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import pandas as pd
from src.wandb_.wandb_client import WandbClient
import wandb

wandbc = WandbClient("test") 
dataset = pd.read_csv(wandbc.load_dataset('final_train_dataset'))

embedding_model = Transformer("distilbert-base-uncased")
tokenized = embedding_model.tokenize(dataset)
loss_model = ContrastiveLoss(embedding_model)

training_args = TrainingArguments(
    report_to="wandb",
    output_dir="./",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=3,
    logging_steps=1,
    disable_tqdm=False,
    weight_decay=0.1,
    warmup_steps=1,
    learning_rate=4e-5,
    run_name="test",
)
trainer = Trainer(loss_model, training_args, train_dataset=tokenized,tokenizer=embedding_model.tokenizer,data_collator=DataCollatorWithPadding(tokenizer=embedding_model.tokenizer))
trainer.train()
# loss_model.model.save('./first.pth')

# best_model = wandb.Artifact(f"model_{run.id}", type="model")
# best_model.add_file("my_model.h5")
# run.log_artifact(best_model)

wandbc.finish()
