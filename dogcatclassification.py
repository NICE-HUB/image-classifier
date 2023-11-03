#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense,Conv2D,MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IMG
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import  OneHotEncoder
from sklearn.preprocessing import  LabelEncoder

from sklearn.preprocessing import ColumnTransformer





# In[11]:


#checking data present in the training data,, displaying random data from the traing data
idx = random.randint(109,len(X_train))
plt.imshow(X_train[idx,:])
plt.show()


# In[12]:


# use imshow to display the image
plt.imshow(X_train[idx])
plt.show()


# In[53]:


pip install --upgrade scikit-learn


# In[ ]:





# In[16]:


# Read the dataset
df = pd.read_csv('C:/Users/HP/datasets/input.csv')
df = pd.read_csv('C:/Users/HP/datasets/input_test.csv')
df = pd.read_csv('C:/Users/HP/datasets/labels.csv')
df = pd.read_csv('C:/Users/HP/datasets/labels_test.csv')



# In[10]:


# Analyze the dataset (e.g., check for missing values, calculate statistics, etc.)
print(df.isnull().sum())


# In[11]:


# Visualize the dataset (e.g., using histograms, scatter plots, etc.)
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[3]:


X_train = np.loadtxt('C:/Users/HP/datasets/input.csv', delimiter=',')
y_train = np.loadtxt('C:/Users/HP/datasets/labels.csv', delimiter=',')

X_test = np.loadtxt('C:/Users/HP/datasets/input_test.csv', delimiter=',')
y_test = np.loadtxt('C:/Users/HP/datasets/labels_test.csv', delimiter=',')


# In[18]:


#check for the shape
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[4]:


X_train = X_train.reshape(len(X_train),100,100,3)
y_train = y_train.reshape(len(y_train),1)
X_test = X_test.reshape(len(X_test),100,100,3)
y_test = y_test.reshape(len(y_test),1)

X_train=X_train/255
X_test=X_test/255


# In[5]:


#trainig data scaled to range 0 to 1
X_train[1,:]


# In[22]:


model = Sequential()
model.add(Conv2D(32,(3,3), activation='relu',input_shape=(100,100,3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(32,(3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())


model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1,activation='sigmoid'))


# In[6]:


#checking data present in the training data,, displaying random data from the traing data
idx = random.randint(0,len(X_train))
plt.imshow(X_train[idx,:])
plt.show()


# In[23]:


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[20]:


model.evaluate(X_test,y_test)


# In[24]:


# normalize data
scaler = MinMaxScaler(feature_range=(0, 1))


# In[13]:


# convert data to numpy array before applying the scaler
X_train = np.array(X_train)
X_test = np.array(X_test)



# In[14]:


scaler = MinMaxScaler()


# In[16]:


from PIL import Image


# In[24]:


import os


# In[27]:


img = Image.open(r'C:\Users\HP\Downloads\meoww')


# In[ ]:




