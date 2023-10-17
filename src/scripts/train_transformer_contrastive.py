from src.models.transformer_model import Transformer
from src.models.contrastive_loss import ContrastiveLoss

from transformers import Trainer, TrainingArguments

model = Transformer("distilbert-base-uncased")
input = model.tokenizer("This is it.", return_tensors="pt")
model(input)
loss = ContrastiveLoss(model)
print(loss)

# trainer = Trainer  # can be peft trainer
