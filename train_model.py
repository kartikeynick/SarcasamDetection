from datasets import load_dataset

from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    TFRobertaModel,
    EarlyStoppingCallback
)
import pandas as pd
import shutil
import math

import torch
import numpy as np
import evaluate
from torch.utils.data import DataLoader

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
torch.backends.cudnn.benchmark = True

SKIP_TRAIN = True #WARNING: setting to false will destroy contents of "my_model"
SKIP_TEST = False

TRAIN_CAP = False
TEST_CAP = False

BS=32
GA_STEPS=1

#share this variable for loading in val and test datasets depending on the task
VAL_DATASET_PATH="dataset/parsed_sets/uc1_test.csv"
TRAIN_DATASET_PATH="dataset/parsed_sets/uc1_train.csv"

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def read_train_data(path, cap=False):
    if type(cap) == int:
        data = pd.read_csv(path).dropna().head(cap)
    else:
        data = pd.read_csv(path).dropna()
    texts = data["comment"].tolist()
    labels = data["label"].tolist()

    return texts, labels


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



val_texts, val_labels = read_train_data(VAL_DATASET_PATH, TEST_CAP)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
val_dataset = MyDataset(val_encodings, val_labels)

print("done tokenizing test")


if not SKIP_TRAIN:
    train_texts, train_labels = read_train_data(TRAIN_DATASET_PATH, TRAIN_CAP)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    train_dataset = MyDataset(train_encodings, train_labels)
    print("done tokenizing train")
    training_args = TrainingArguments(
        output_dir="./results",  # output directory
        num_train_epochs=2,  # total number of training epochs
        per_device_train_batch_size=BS,  # batch size per device during training
        fp16=True,
        per_device_eval_batch_size=64,  # batch size for evaluation
        # warmup_steps=100,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=10,
        gradient_accumulation_steps=GA_STEPS,
        gradient_checkpointing=True,
        # optim="adafactor",
        load_best_model_at_end = True,
        evaluation_strategy='steps',
        save_steps=100
    )

    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base"
    )

    print(training_args.learning_rate)

    # training_args.learning_rate *= math.sqrt(GA_STEPS*BS/8)
    training_args.learning_rate /= 5
    print(training_args.learning_rate)


    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        callbacks = [EarlyStoppingCallback(early_stopping_patience = 5)]
    )

    torch.cuda.empty_cache()
    trainer.train()
    try:
        shutil.rmtree("my_model")
    except: 
        pass
    trainer.save_model("my_model")
if not SKIP_TEST:
    model = RobertaForSequenceClassification.from_pretrained("my_model") #my model
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        eval_dataset=val_dataset,  # evaluation dataset
    )

    test_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    preds_logits, labels, metrics = trainer.prediction_loop(test_loader, description='prediction')
    preds = np.argmax(preds_logits, axis=-1)
    print(preds)

    metrics = ["accuracy", "precision", "recall", "f1"]

    for m in metrics:
        metric = evaluate.load(m)
        res = metric.compute(predictions=preds, references=labels)
        print(res)

