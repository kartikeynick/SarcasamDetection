import pandas as pd
import json, random


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
random.seed("COOL")

# result = pd.read_csv("test.csv")
# label_row = 'label'
# plaintext= 'comment'

result = pd.read_json("sarcasm_fixed.json")
label_row = 'is_sarcastic'
plaintext = 'headline'

print(result)
s = result[label_row].value_counts()

# #pos
print(s[1])
print(s[1]/result.shape[0])


# #neg
print(s[0])
print(s[0]/result.shape[0])

words=['will','almost','sure','oh','sorry','almost','want','said','know','talk',"yeah","right","like", "little", "still", "man", "friend", "time", "awesome", 'one', "so", "smart", 'what', 'nice', 'come', "nan", 'will', 'obviously', 'need', 'people', 'one', 'time', 'maybe', 'mean', 'no', 'game', 'really', 'totally' , 're',"well", "way"]

words2=['report','man','woman','nation','year','old','year','still','time','friend','american','area','little','make','one','guy','trump','new','back','people','will','way', 'will']

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
for i, (index, row) in enumerate(result.iterrows()):
    guess = 0
    for word in words:
        if str(word).lower() in str(row[plaintext]).lower():
            guess = 1
            break

    if guess == 0:
        if row[label_row] == guess:
            true_neg += 1
        else:
            false_neg += 1
    elif guess == 1:
        if row[label_row] == guess:
            true_pos += 1
        else:
            false_pos += 1

print("STATS")
print(true_pos)
print(false_pos)
# print(false_/pos)
print((true_neg + true_pos + false_neg + false_pos)) #SHOULD MATCH DS SIZE

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