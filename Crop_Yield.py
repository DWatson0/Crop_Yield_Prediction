#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import random


# In[2]:


seed = 1234
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)


# In[3]:


path = "Downloads/Crop Yiled with Soil and Weather.csv"
df = pd.read_csv(path)


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


#sanity check


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


#find missing value
df.isnull().sum()


# In[10]:


#missing value percentage
df.isnull().sum()/df.shape[0]*100


# In[11]:


#find duplicates
df.duplicated().sum()


# In[12]:


#Exploratory Data Analysis


# In[13]:


df.describe()


# In[14]:


#histogram for understanding distribution
for i in df.select_dtypes(include="number").columns:
    sns.histplot(data=df,x=i)
    plt.show()


# In[15]:


#boxplot-to-identify Outliers
for i in df.select_dtypes(include="number").columns:
    sns.boxplot(data=df,x=i)
    plt.show()


# In[16]:


#scatter plot to understand the relationship


# In[17]:


df.select_dtypes(include="number").columns


# In[18]:


for i in ['Fertilizer', 'temp', 'N', 'P', 'K']:
    sns.scatterplot(data=df,x=i,y='yeild')
    plt.show()


# In[19]:


#correlation 


# In[20]:


df.corr()


# In[21]:


plt.figure(figsize=(15,15))
sns.heatmap(df.corr(),annot=True)


# In[22]:


#no missing value
#no outliers
#drop duplicates
df.drop_duplicates()


# In[23]:


X = df[['Fertilizer','temp','N','P','K']]
y = df['yeild']


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


# In[25]:


model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(64,name = "L1",activation = "relu"),
        tf.keras.layers.Dense(32,name = "L2",activation = "relu"),
        tf.keras.layers.Dense(1,name = "L3"),
    ], name="Complex"
)
model.compile(
    loss=tf.keras.losses.MeanSquaredError,
    optimizer=tf.keras.optimizers.Adam(0.001)
)


# In[26]:


model.fit(X_train_scaled,y_train,epochs=500,validation_split=0.2)


# In[27]:


y_predict = model.predict(X_test_scaled)
mse = mean_squared_error(y_predict, y_test)
r2_test = r2_score(y_test, y_predict)

print(f"MSE score:{mse:.4f},r2 score:{r2_test:.3f}")


# In[ ]:




