import pandas as pd
import json


#REDDIT

# train = pd.read_csv("train.csv")
# test = pd.read_csv("test.csv")
# validation = pd.read_csv("validation.csv")

# frames = [train, test, validation]
# result = pd.concat(frames)
# print(result.shape[0])
# s = result['label'].value_counts()
# t = result['subreddit'].value_counts()
# print(t)

# #pos
# print(s[1])
# print(s[1]/result.shape[0])


# #neg
# print(s[0])
# print(s[0]/result.shape[0])

# # 228833
# # 0.4945353161475836
# # 0.5054646838524164

#TWITTER 
# f = open("sarcasm.json")
# l = []
# for line in f.readlines():
#     j = json.loads(line)
#     print(j)
#     l.append(j)

# formatted = json.dumps(l)
# f = open("sarcasm_fixed.json", "w+")
# f.write(formatted)

# exit(0)

result = pd.read_json("sarcasm_fixed.json")
print(result)
s = result['is_sarcastic'].value_counts()

# #pos
print(s[1])
print(s[1]/result.shape[0])


# #neg
print(s[0])
print(s[0]/result.shape[0])

#positive label
guess = 1

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
for i, (index, row) in enumerate(result.iterrows()):
    if guess == 0:
        if row['is_sarcastic'] == guess:
            true_neg += 1
        else:
            false_neg += 1
    elif guess == 1:
        if row['is_sarcastic'] == guess:
            true_pos += 1
        else:
            false_pos += 1

accuracy = (true_pos + true_neg) / (true_neg + true_pos + false_neg + false_pos)
precision = true_pos / (true_pos + false_pos)
recall = true_pos / (true_pos + false_neg)
f1 = 2 * (precision * recall) / (precision + recall)

print("accuracy")
print(accuracy)
print('precision')
print(precision)
print('recall')
print(recall)
print("f1")
print(f1)