from datasets import load_dataset

from transformers import RobertaTokenizerFast, RobertaForSequenceClassification,Trainer, TrainingArguments
import pandas as pd

csv = pd.read_csv("dataset/train.csv")
csv = csv.head(100)
csv.to_csv('dataset/training_small.csv')
print(csv)

# dataset = load_dataset('csv', data_files='dataset/training.csv')
# print(dataset)


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