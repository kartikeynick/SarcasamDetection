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
