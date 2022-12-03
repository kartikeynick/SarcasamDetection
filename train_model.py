from datasets import load_dataset

from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import pandas as pd
import shutil

import torch
import numpy as np
import evaluate
from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

SKIP_TRAIN = False  # WARNING: setting to false will destroy contents of "my_model"
SKIP_TEST = False

TRAIN_CAP = 500
TEST_CAP = 100


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


train_texts, train_labels = read_train_data("dataset/train.csv", 400)
val_texts, val_labels = read_train_data("dataset/validation.csv", 100)


train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
train_dataset = MyDataset(train_encodings, train_labels)
val_dataset = MyDataset(val_encodings, val_labels)

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments


if not SKIP_TRAIN:
    training_args = TrainingArguments(
        output_dir="./results",  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=10,
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased"
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
    )

    trainer.train()
    shutil.rmtree("my_model")
    trainer.save_model("my_model")
if not SKIP_TEST:
    model = DistilBertForSequenceClassification.from_pretrained("my_model")
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
    )

    preds_logits, labels, metrics = trainer.predict(val_dataset)
    preds = np.argmax(preds_logits, axis=-1)
    print(preds)

    metrics = ["accuracy", "precision", "recall", "f1"]

    for m in metrics:
        metric = evaluate.load(m)
        res = metric.compute(predictions=preds, references=labels)
        print(res)


# dataset = load_dataset('csv', data_files='dataset/training_small.csv')
# print(dataset['train'])


# model = RobertaForSequenceClassification.from_pretrained('roberta-base')
# tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_length = 512)

# def tokenization(batched_text):
#     return tokenizer(batched_text['comment'], padding = True, truncation=True)


# train_data = dataset.map(tokenization, batched = True, batch_size = len(dataset))

# # train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# trainer = Trainer(
#     model=model,
#     train_dataset=train_data,
#     # args=training_
#     #args,
#     # compute_metrics=compute_metrics,
#     # eval_dataset=test_data
# )
# trainer.train()
