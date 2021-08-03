# Creating-chatbot-model

  In this project I created a chatbot using python and [Google colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true), Create a new notebook to start working ! 
  
  The process of creating this chatbot cotains three stages as follows:
  
  ![image](https://user-images.githubusercontent.com/85634099/128045977-c49f3bba-50f5-4e3d-9074-027d72e52da0.png)
  
  
  ## Data
  Start by creating a Json file which will holds the chatbot intents, you can create it by opening any text editor paste your code and save that file with the .json extension. 
  
  This file contains the intents for the chatbot to recognize and respond correctly to the user as it has samples of what the user would say (patterns) and what's the correct respond (responses)
  *Note: there could be more then one accurate response )*
  
  ![image](https://user-images.githubusercontent.com/85634099/128067371-1c539a58-9db9-4c85-830e-625d6181badd.png)
  
  I imported json and then uploaded my Json file to my google colab notebook and load it into intents variable to be used later on:
  ```
  #import chat-bot intents file
  import json
with open('intents.json') as json_data:
    intents = json.load(json_data)
  ```
  
  ## Processing
  For this step you will need to import Natural Language Toolkit (NLTK) which has text processing libraries to be applied on the Json file's data from the previous step to loop through each sentence in the intent's patterns and tokenize each and every word in the sentence.
  
  ```
  #loop through each sentence in the intent's patters
for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each and every word in the sentence
        w=nltk.word_tokenize(pattern)
        
        #add word to the word list
        words.extend(w)
        
        #add word(s) to documents
        documents.append((w, intent['tag']))
        
        #add tags to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
  ```
  
  Then i continued working on the data by using the following code to find all unique words, remove the duplicated classes if there's any:
  
  ```
  #perform stemming and lower each word as well as remove duplicates

words=[stemmer.stem(w.lower()) for w in words if w not in ignore]
words=sorted(list(set(words)))

#remove duplicate classes
classes=sorted(list(set(classes)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)
  ```
  
  ## Training 
  In this step tensorflow is used to train the model in tflearn using all the data that we've processed during the past steps. We need few packages to be imported:
  
  ```
  #Libararies needed for TensorFlow processing
from tensorflow.python.framework import ops
import tensorflow as tf
import numpy as np
import tflearn
import random
  ```
  
  Then we trained the model using the words we extraced previously:
  
  ```#create training data
training=[]
output=[]

#create an empty array for output
output_empty=[0]*len(classes)

#create training set bag of words for each sentence
for doc in documents:
  #initialize bag of words
  bag=[]
  #list of tokenized words for the pattern
  pattern_words=doc[0]
  #stemming each word
  pattern_words=[stemmer.stem(word.lower()) for word in pattern_words]
  #create bag of words array
  for w in words:
    bag.append(1) if w in pattern_words else bag.append(0)
    
  #output is 1 for current tag and o for the rest of other tags
  output_row=list(output_empty)
  output_row[classes.index(doc[1])]=1
    
  training.append([bag,output_row])
    
#shuffling features and turning it into np.array
random.shuffle(training)
training=np.array(training)
  
#creating training lists
train_x =list(training[:,0])
train_y =list(training[:,1])
```

After the training process is done it's time to check if the model is well trained and that is by checking the classification values as follows:

```
ERROR_THRESHOLD = 0.0
def classify(sentence):
  #generate probabilities from the model
  results = model.predict([bow(sentence, words)])[0]
  
  #filter out prediction below a threshold
  results = [[i,r] for i, r in enumerate(results) if r> ERROR_THRESHOLD]
  
  #sort by strength of probability
  results.sort(key=lambda x: x[1], reverse=True)
  return_list=[]
  
  for r in results:
    return_list.append((classes[r[0]], r[1]))
  
  #return tuple of intent and probability
  return return_list

def response(sentence, userID='123', show_details=False):
  results=classify(sentence)
  
  if results:
    while results:
      for i in intents['intents']:
        if i['tag'] == results[0][0]:
          return print(random.choice(i['responses']))
      results.pop(0)
  ```
  
  ## The Results
  First let's check the classification values by giving the model a sentense and see if ir would be classified correctly under which intent:
  (ex: asking how old are you? shows highest possibility of 0.95 under the intent Age which is correct !)
  ![image](https://user-images.githubusercontent.com/85634099/128053636-24a616b4-c754-47a5-bcba-0cffd65363de.png)
  
  Finally, i made a loop taking the user input until the user says bye and terminate the program using following code lines:
  ```
  msg=''        
### MAIN PROGRAM ###
print("\n Hey ! I'm a Chatbot !\n Start a Conversation with me ! :D\n\n\n")

while classify(msg)[0][0] != 'goodbye':    
    # get the user input
    msg = input("You: ")
    print("Bot:",end=' ')
    response(msg)
  ```
 
  And this video shows how the chatbot correctly answered all the user's messages:
  
https://user-images.githubusercontent.com/85634099/128067949-276277fc-942f-444a-b932-9f75f0646ac8.mp4


  

