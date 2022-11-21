import json
import random
with open('sarcasam.json') as f:
  data = json.load(f)

keys=[]
for i in data.keys():
  keys.append(i)

# dividing the dataset into 80-10-10 randomly

# the first 80%
training=[]
test=[]
other=[]
for i in range(0,552):
  t=random.choice(keys)
  training.append(t)
  keys.remove(t)

for i in range(0,69):
  t = random.choice(keys)
  test.append(t)
  keys.remove(t)

other=keys #copying the rest in othe other


'''print(len(test),"\n",len(training),"\n",len(other))

print(test,"\n",training,"\n",other)


'''

# creating the Test, Compute Json file
testjson = {"test": {}}  # creating a disctionary name intents
otherjson={"Compute":{}}
counter=0
for i in range(0,69):
  counter+=1
  z1={test[i]:data[test[i]]}
  z2= z1={other[i]:data[test[i]]}
  testjson["test"].update(z1)  # appending the data into the dictionary as patterns and responses
  otherjson["Compute"].update(z2)

#Writing it in a json file
with open("Test.json", "w") as f:  # creating the file and dumpinf all the data in the dictionary into json file
    json.dump(testjson, f, indent=2)

with open("Compute.json", "w") as f:  # creating the file and dumpinf all the data in the dictionary into json file
    json.dump(otherjson, f, indent=2)


# For the training Data
trainingJson={"Train":{}}
for i in range(0,552):
  z3={training[i]:data[training[i]]}
  trainingJson["Train"].update(z3)  # appending the data into the dictionary as patterns and responses

with open("Train.json", "w") as f:  # creating the file and dumpinf all the data in the dictionary into json file
    json.dump(trainingJson, f, indent=2)

