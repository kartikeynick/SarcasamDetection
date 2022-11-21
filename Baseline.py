import json
'''
with open('Train.json') as f:
  f = json.load(f)

data=f['Train']
'''
'''
#baseline for checking if everything is a sarcasam and also not
c=0 #counter
sarcasamT=0
sarcasamF=0
for i in data:
    c+=1
    x=data[i]
    y=x['sarcasm']# it is a Boolean
    if (y):
        sarcasamT+=1
    else:
        sarcasamF+=1

xx=sarcasamT+sarcasamF

sarcasamT=sarcasamT/c*100.0
sarcasamF=sarcasamF/c*100.0
print("Count = \t",c,"\nTrue = \t",sarcasamT,"\nFalse = \t",sarcasamF)
'''
'''

Baseline
Count = 	 552 
True = 	 50.54347826086957 
False = 	 49.45652173913043
'''

#Reading the file from the Twitter Dataset
tweets = []
for line in open('Sarcasm_Headlines_Dataset.json', 'r'):
    tweets.append(json.loads(line))

c2=0
isSarcasam=0
notSarcasam=0
for i in tweets:
  c2+=1
  if(i['is_sarcastic']):
    isSarcasam+=1
  else:
    notSarcasam+=1


isSarcasam=isSarcasam/c2*100.0
notSarcasam=notSarcasam/c2*100.0

print("\n",isSarcasam,"\n",notSarcasam)


'''
 Total = 26709 

 Is Sarcasam = 43.89531618555543 
 Not Sarcasam = 56.10468381444457
'''

# so many explaination signs
# repeated alases
# oh, sure, ha ha ha,
# more negative sentenses
# repeated words, no no no, ha ha ha
