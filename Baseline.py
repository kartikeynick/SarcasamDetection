import json
import random
import nltk
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
with open('Dataset/Test.json') as f:
  f = json.load(f)

data=f['test']

# Checking for random as well
rand=0 # for True (is Sarcasm)

# baseline for checking if everything is a sarcasm and also not
c=0 #counter
sarcasmT=0
sarcasmF=0

# Constructing a List based on the Word Cloud

words=['good','said','thing','will','almost','sure','oh','sorry','almost','want','science','said','know','talk','m',"s",]
words2=0
WCToken=[]
for i in data:
    c+=1
    x=data[i]
    y=x['sarcasm']# it is a Boolean
    if (y):
        sarcasmT+=1
    else:
        sarcasmF+=1
    # checking for Random
    t=random.choice([True,False])
    if(t):
        rand+=1

    y2 = x['utterance']
    y2 = nltk.word_tokenize(y2) # tokenize the uttrance
    # Converts each token into lowercase
    for i in range(len(y2)):
        y2[i] = y2[i].lower()

    print(y2)
    for j in words:
        print(j)
        if (j in y2):
            print("\n\nMatches one", j, "\t\t", y2)
            words2 += 1
    # Checking for the other one where some common elements are present in the sarcastic sentense
'''
    # if sarcastic then make the word cloud
    if (y):
        # converting the sentenses into a list
        y2=x['utterance']
        y2=nltk.word_tokenize(y2)
        # Converts each token into lowercase
        for i in range(len(y2)):
            y2[i] = y2[i].lower()
        WCToken+=y2 # adding all the sarcastic sentences into a single list


#### Constricting a Word Cloud to inserstand the most used words in Sarcastic sentence ####

print(WCToken)
comment_words = ''
comment_words += " ".join(WCToken) + " "
wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          min_font_size=10).generate(comment_words)

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show() # making a Word Cloid

'''
'''
    print(y2)
    for j in words:
        print(j)
        if (j in y2):
            print("\n\nMatches one",j,"\t\t",y2)
            words2+=1
    #Checking for the other one where some common elements are present in the sarcastic sentense.

'''

## calculating the percentage for the benchmark for either all sarcasm or all non sarcasm accuricy
sarcasmT=sarcasmT/c*100.0
sarcasmF=sarcasmF/c*100.0

# Is sarcasm Random
rand= rand/c*100.0

# for the common word one
words2=words2/c*100.0

print("Count = \t",c,"\nTrue  = \t",sarcasmT,"\nFalse = \t",sarcasmF,"\nBy Random:\t",rand,"\nWords Present for Sarcasm =\t",words2)
'''

The Resuly
Baseline -- Training ds
Count = 	 552 
True  = 	 50.54347826086957 
False = 	 49.45652173913043
By Random:	 47.82608695652174
or
By Random:	 44.927536231884055 

for Testing Dataset 

Count = 	 69 
True  = 	 49.275362318840585 
False = 	 50.72463768115942
'''

# Reading the file from the Twitter Dataset
tweets = []
for line in open('Dataset/Sarcasm_Headlines_Dataset.json', 'r'):
    tweets.append(json.loads(line))

sarcWords=['report','man','woman','nation','year','old','year','still','time','friend','american','area','little','make','one','guy','trump','new','back','people','will','way']
words2=0
c2=0
isSarcasam=0
notSarcasam=0
rand=0
for i in tweets:
    c2+=1
    if(i['is_sarcastic']):
        isSarcasam+=1
    else:
        notSarcasam+=1
    # for Random one
    t=random.choice([True,False])
    if(t):
        rand+=1

# if sarcastic then make the word cloud
    if (i['is_sarcastic']):
        # converting the sentenses into a list
        y2=i['headline']
        y2=nltk.word_tokenize(y2)
        # Converts each token into lowercase
        for i in range(len(y2)):
            y2[i] = y2[i].lower()
        WCToken+=y2 # adding all the sarcastic sentences into a single list
    for j in sarcWords:
        print(j)
        if (j in y2):
            print("\n\nMatches one", j, "\t\t", y2)
            words2 += 1
'''
#### Constricting a Word Cloud to inserstand the most used words in Sarcastic sentence ####

print(WCToken)
comment_words = ''
comment_words += " ".join(WCToken) + " "
wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          min_font_size=10).generate(comment_words)

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show() # making a Word Cloid

'''

isSarcasam=isSarcasam/c2*100.0
notSarcasam=notSarcasam/c2*100.0
rand=rand/c2*100.0
words2=words2/c2*100.0
print("\nIs Sarcasam = \t",isSarcasam,"\nNot Sarcasam = \t",notSarcasam,'\nRandom = ',rand,'\nWord Sarc',words2)


'''

The Result for the Twitter Dataset
 Total = 26709 

 Is Sarcasam  = 43.89531618555543 
 Not Sarcasam = 56.10468381444457
 Random =  50.106705604852294
'''

# Random
# checking for these below



# so many explaination signs
# repeated alases
# oh, sure, ha ha ha,
# longer questions
# more negative sentenses
# repeated words, no no no, ha ha ha

#
