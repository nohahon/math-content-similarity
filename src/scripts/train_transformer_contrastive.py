from src.models.transformer_model import Transformer
from src.models.contrastive_loss import ContrastiveLoss

from transformers import Trainer, TrainingArguments
import pandas as pd
from src.wandb_.wandb_client import WandbClient
import wandb

d = pd.read_csv(
    "/home/tomas/Documents/MBG/projects/AndreAnkitMath/MathContentSimilarity/data/final_train_dataset.csv",
)[:5]
embedding_model = Transformer("distilbert-base-uncased")
tokenized = embedding_model.tokenize(d)
loss_model = ContrastiveLoss(embedding_model)

wbclient = WandbClient("test")
training_args = TrainingArguments(
    report_to="wandb",
    output_dir="./",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=3,
    logging_steps=1,
    disable_tqdm=False,
    weight_decay=0.1,
    warmup_steps=10,
    learning_rate=4e-5,
    run_name="test",
)
trainer = Trainer(loss_model, training_args, train_dataset=tokenized)
trainer.train()
wbclient.upload_dataset(
    "/home/tomas/Documents/MBG/projects/AndreAnkitMath/MathContentSimilarity/data/final_train_dataset.csv",
)
# loss_model.model.save('./first.pth')

# best_model = wandb.Artifact(f"model_{run.id}", type="model")
# best_model.add_file("my_model.h5")
# run.log_artifact(best_model)

wbclient.finish()
