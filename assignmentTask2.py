#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# In[2]:


data = pd.read_csv("BankNote_Authentication.csv")


# In[3]:


data.head()


# In[4]:


X = data.drop("class", axis=1)
y = data["class"]


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[6]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[7]:


model = Sequential()
model.add(Dense(4, input_dim=4, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(1, activation="sigmoid"))


# In[8]:


model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"])


# In[9]:


model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2)


# In[10]:


score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss: %.2f%%" % (score[0] * 100))
print("Test accuracy: %.2f%%" % (score[1] * 100))

