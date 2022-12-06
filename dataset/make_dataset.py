import pandas as pd
import re
from sklearn.model_selection import train_test_split


df = pd.read_csv("sarcasm.csv")
data_count = df.shape[0]


# print("read")
# print(df)

new_df = df.iloc[:0].copy()

for i, (index, row) in enumerate(df.iterrows()):
    context = row['parent_comment']
    context = context.replace("!", ".")
    context = context.replace("?", ".")
    context_split = context.split(".")
    context_no_blanks =  list(filter(lambda x: x != '',context_split))
    if len(context_no_blanks) >= 3:
        # print(row)
        new_df.loc[len(new_df.index)] = row


    if i % 1000 == 0:
        print(f"{i/data_count * 100}% Done")

print(new_df.shape[0])
train, test_full = train_test_split(new_df, test_size=0.2)
validation, test = train_test_split(test_full , test_size=0.5)

train.to_csv("train.csv")
test.to_csv("test.csv")
validation.to_csv("validation.csv")