#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Group Assignment 2 - CS 985/CS 988
#Group AH - Peter Janik (201979128), Inno Mvula (201973944), Thom Reynard (201977555)
#April 8th 2020

#The approach used to complete the sentiment analysis was the combination of a Convolutional Neural Network and Long-Short Term Memory (LSTM) with word embeddings, encoding and decoding. The idea for this model is not new and it is “through the use of a multi-layered CNN and character level word embeddings that the Facebook team was able to classify the polarity of longer form text - Yelp and Amazon reviews - with 95.72 percent accuracy” (11). It is also through the strong previous performance of Tweet2Vec, a method developed at MIT for tweet representation, that relied on “learning tweet embeddings using (a) character-level CNN-LSTM encoder decoder” (12) that formed the base of our model.


# ## Packages and Databases:

# *Importing packages used.*

# In[ ]:


import numpy as np
import pandas as pd


# In[2]:


get_ipython().run_line_magic('tensorflow_version', '1.x')
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
print(keras.__version__)


# In[3]:


from keras.preprocessing import sequence


# In[ ]:


from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout


# *Getting datasets.*

# In[6]:


from google.colab import files
file = files.upload()
train = pd.read_csv("training.csv")
test = pd.read_csv("test.csv")


# *Taking a look at the datasets.*

# In[7]:


train.shape


# In[8]:


test.shape


# In[9]:


train.head(6)


# In[10]:


test.head(6)


# In[11]:


count = train["target"].value_counts() 
print(count) 


# *It appears that the database only includes tweets with targets of 0 or 4.*

# In[12]:


train.at[1,"text"]


# ## Parameters

# *The important parameters have been collected here for fine-tuning.*

# In[ ]:


#Parameters

vocab_size = 50000
#Set as None for no limit


max_words = 20
#Longest tweet after coding depends on the vocab size but is roughly 45.


batch_size = 5000
#train set has 1,000,000 entries

num_epochs = 10

embedding_size = 32


# ## Eliminating unnecessaary columns:

# *For training our model we only need the target and the tweet text.*

# In[133]:


train_raw = train[["target", "text"]]
train_raw.head(6)


# *To make our prediction we only need the tweet text.*

# In[134]:


test_raw = test[["text"]]
test_raw.head(6)


# ## Text Encoding and Padding:

# *Each tweet must be converted to a sequence of integers. The vocabulary size has been limited to avoid overftting with words only used in one tweet. Only the most common words are used, the rest are ommitted when converting to integer list form.*

# *Constructing the tokeniser. We are considering words as our basic unit, not characters, so char_level has been set to False.*

# In[135]:


tokenizer = keras.preprocessing.text.Tokenizer(num_words = vocab_size, lower  = True, 
                                               char_level = False)
tokenizer.fit_on_texts(train_raw["text"])
tokenizer.fit_on_texts(test_raw["text"])
print("Done.")


# *Applying the tokeniser to the text column to create a coded column.*

# In[136]:


train_raw["coded"] = tokenizer.texts_to_sequences(train_raw["text"])
test_raw["coded"] = tokenizer.texts_to_sequences(test_raw["text"])
print("Done.")


# *Checking the minimum and maximum tweet length after coding.*

# In[137]:


print('Maximum tweet length: {}'.format(max(len(max(train_raw["coded"], key=len)),
                                             len(max(test_raw["coded"], key=len)))))

print('Minimum tweet length: {}'.format(min(len(min(train_raw["coded"], key=len)),
                                             len(min(test_raw["coded"], key=len)))))


# In[138]:


train_raw["padded"] = train_raw["coded"]
test_raw["padded"] = test_raw["coded"]


# *All input tweets must have the same length. Longer tweets are shortened to the maximum number of words, shorter tweets are padded with zeroes which do not count as words.*
# 
# 

# In[139]:


train_raw["padded"] = sequence.pad_sequences(train_raw["coded"], maxlen=max_words,
                                             padding = "post", value=0).tolist()
test_raw["padded"] = sequence.pad_sequences(test_raw["coded"], maxlen=max_words,
                                             padding = "post", value=0).tolist()
train_raw.head(3)


# *Now the maximum tweet length and the minimum tweet length should both be equal to the maximum number of words specified earlier.*

# In[140]:


print('Maximum tweet length: {}'.format(max(len(max(train_raw["padded"], key=len)),
                                             len(max(test_raw["padded"], key=len)))))

print('Minimum tweet length: {}'.format(min(len(min(train_raw["padded"], key=len)),
                                             len(min(test_raw["padded"], key=len)))))


# ## Building Network:

# *Building model. The final layer uses softmax as the activation since our targets are 5 mutually exclusive classes (the numbers 0-4).*

# In[141]:


model=Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length=max_words))
model.add(LSTM(100))
model.add(Dense(5, activation="softmax"))

print(model.summary())


# In[142]:


model.compile(loss="sparse_categorical_crossentropy", 
             optimizer='adam', 
             metrics=['accuracy'])
print("Done.")


# ## Training Network:

# *Separating training set into validation set and true training set.*

# In[143]:


validation = train_raw[:batch_size]
new_train = train_raw[batch_size:]
print("Done.")


# *Checking these sets:*

# In[144]:


validation.head(3)


# In[145]:


new_train.head(3)


# *Setting up sets for model training.*

# In[146]:


X_valid = validation["padded"].values.tolist()
X_valid = np.asarray(X_valid)

y_valid = validation["target"].values.tolist()
y_valid = np.asarray(y_valid)

X_train = new_train["padded"].values.tolist()
X_train = np.asarray(X_train)

y_train = new_train["target"].values.tolist()
y_train = np.asarray(y_train)
print("Done.")


# *Training:*

# In[147]:


model.fit(X_train, y_train, validation_data=(X_valid, y_valid), 
          batch_size=batch_size, epochs=num_epochs)


# ## Testing

# *Evaluating test file and exporting.*

# In[148]:


to_test = test_raw["padded"].values.tolist()
to_test = np.asarray(to_test)
print("Done.")


# In[ ]:


predictions = model.predict_classes(to_test)


# In[150]:


predictions


# In[ ]:


test["target"] = list(predictions)


# In[152]:


test.head()


# In[ ]:


submission = test[["id","target"]]


# In[ ]:


submission.to_csv("prediction.csv",index=False)
files.download("prediction.csv")

