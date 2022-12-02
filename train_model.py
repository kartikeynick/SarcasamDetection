from datasets import load_dataset

from transformers import RobertaTokenizerFast, RobertaForSequenceClassification,Trainer, TrainingArguments
import pandas as pd

import torch

from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


def read_train_data(path):
    data = pd.read_csv(path)
    texts = data['comment'].tolist()
    labels = data['label'].tolist()

    return texts, labels

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_texts, train_labels = read_train_data('dataset/training_small.csv')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
train_dataset = MyDataset(train_encodings, train_labels)

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    # eval_dataset=val_dataset             # evaluation dataset
)
trainer.train()

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