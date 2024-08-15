import sys
import pickle
import evaluate
import numpy as np
from datasets import load_dataset
from collections import defaultdict
from transformers import AutoTokenizer,DataCollatorWithPadding, pipeline
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

#default params
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
accuracy = evaluate.load("accuracy")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def getdataset():
    accuracy = evaluate.load("accuracy")
    zbm_d = load_dataset("zbm_top1000_ttv_textformat")
    tokenized_zbm_d = zbm_d.map(preprocess_function, batched=True)
    print(len(tokenized_zbm_d["train"]))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id)
    training_args = TrainingArguments(
        output_dir="trainedmodels/BERT_rerank_withtextip",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    #print(type(tokenized_zbm_d["train"]))
    #print(type(tokenized_zbm_d["train"][1]))
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_zbm_d["train"],
        eval_dataset=tokenized_zbm_d["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()    
 
def testTrainedModel():
    dictOfrslts = defaultdict(lambda:list())
    testdata_ids = pickle.load(open("BERT_rerankdata_textformat.pkl", 'rb'))
    dataset_test = load_dataset("/zbm_top1000_ttv_textformat")["test"]
    print(len(testdata_ids), len(dataset_test))
    classifier = pipeline(task="text-classification", model="trainedmodels/BERT_rerank_withtextip/checkpoint-8534")
    for id_,eachSamp in enumerate(dataset_test):
        clsf_ = classifier(eachSamp['text'])
        dictOfrslts[testdata_ids[id_][0]].append([testdata_ids[id_][1], clsf_[0]["label"], clsf_[0]["score"]])
        #[{'label': 'NEGATIVE', 'score': 0.9531010985374451}]
    with open("pred_scores_tfrerank_withtext.pkl", "wb") as pfg:
        pickle.dump(dict(dictOfrslts), pfg)

#getdataset()
testTrainedModel()
