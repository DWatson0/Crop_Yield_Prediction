#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# In[2]:


path = "Downloads/Crop Yiled with Soil and Weather.csv"
df = pd.read_csv(path)


# In[3]:


df.drop_duplicates()


# In[4]:


X = df[['Fertilizer','temp','N','P','K']]
y = df['yeild']
y = y.values.reshape(-1, 1)


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state=3)
X_cv, X_test, y_cv, y_test = train_test_split(X_test,y_test,test_size=0.5, random_state=3)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_cv_scaled = scaler.fit_transform(X_cv)
X_test_scaled = scaler.fit_transform(X_test)


# In[6]:


def relu(z):
    return np.maximum(0, z)


# In[7]:


def relu_derivative(z):
    return (z > 0).astype(float)


# In[8]:


def linear(z):
    return z


# In[9]:


def weight_bias_initialize():
    np.random.seed(42)
    W1 = np.random.randn(X_train_scaled.shape[1],64)*0.1
    b1 = np.zeros((1, 64))

    W2 = np.random.randn(64,32)*0.1
    b2 = np.zeros((1, 32))

    W3 = np.random.randn(32, 1)*0.1
    b3 = np.zeros((1, 1))

    return (W1,b1,W2,b2,W3,b3)


# In[10]:


def compute_final_errors(X, y, X_cv, y_cv, W1, b1, W2, b2, W3, b3):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = relu(z2)
    z3 = a2 @ W3 + b3
    a3 = linear(z3)
    trainError = np.mean((a3 - y) ** 2)

    z1 = X_cv @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = relu(z2)
    z3 = a2 @ W3 + b3
    a3 = linear(z3)
    cvError = np.mean((a3 - y_cv) ** 2)

    return trainError, cvError


# In[11]:


def training_model_lambda(_lambda_arr,alpha,X,y,X_cv,y_cv,epochs,patience=3000, min_delta=1e-4):
    for _lambda in _lambda_arr:
        W1,b1,W2,b2,W3,b3 = weight_bias_initialize()
        best_cv_error = float("inf")
        best_weights = None
        wait = 0

        for epoch in range(epochs):

            ##########Forward propagation##########
            z1 = X @ W1 + b1 #(2076,5)@(5,64)
            a1 = relu(z1)

            z2 = a1 @ W2 + b2 # (2076,64)@(64,32)
            a2 = relu(z2)

            z3 = a2 @ W3 + b3 #(2076,32)@(32,1)
            a3 = linear(z3)

            error = np.mean((a3 - y)**2 ) + _lambda * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
            ##########Backward propagation#############

            dz3 = a3 - y
            dW3 = a2.T @ dz3 + _lambda * W3
            db3 = np.sum(dz3,axis=0,keepdims = True)

            da2 = dz3 @ W3.T         
            dz2 = da2 * relu_derivative(z2)  
            dW2 = a1.T @ dz2 + _lambda * W2        
            db2 = np.sum(dz2, axis=0, keepdims=True)

            da1 = dz2 @ W2.T         
            dz1 = da1 * relu_derivative(z1)  
            dW1 = X.T @ dz1 + _lambda * W1        
            db1 = np.sum(dz1, axis=0, keepdims=True)


            ##########Update Weights####################
            W1 = W1 - alpha*dW1
            b1 = b1 - alpha*db1

            W2 = W2 - alpha*dW2
            b2 = b2 - alpha*db2

            W3 = W3 - alpha*dW3
            b3 = b3 - alpha*db3

            ##########Print error every 1000 Epochs#############
            if epoch % 1000 == 0:
                trainError,cvError = compute_final_errors(X, y, X_cv, y_cv, W1, b1, W2, b2, W3, b3)
                print(f"Epoch {epoch}, Train Error: {trainError:.4f}, Validation Error: {cvError:.4f}")

                 ############# Early stopping logic###############
                if cvError + min_delta < best_cv_error:
                    best_cv_error = cvError
                    best_weights = (W1.copy(), b1.copy(), W2.copy(), b2.copy(), W3.copy(), b3.copy())
                    wait = 0
                else:
                    wait += 1000

                if wait >= patience:
                    print(f"Early stopping at epoch {epoch} for lambda {_lambda:.5f}. Best validation error: {best_cv_error:.4f}")
                    break
        if best_weights:
            W1, b1, W2, b2, W3, b3 = best_weights

        trainError,cvError = compute_final_errors(X, y, X_cv, y_cv, W1, b1, W2, b2, W3, b3)
        print(f"\nlamdba:{_lambda:.5f}")
        print(f"Train Error: {trainError:.4f}")
        print(f"Cross validation Error: {cvError:.4f} \n")
        print("***"*15,"\n")


# In[12]:


_lambda_arr = np.array([0.00001,0.0001,0.001,0.01,0.1])
alpha = 0.0001
training_model_lambda(_lambda_arr,alpha,X_train_scaled,y_train,X_cv_scaled,y_cv,10000)
#lambda just right = 0.0001


# In[13]:


def training_model_alpha(alpha_arr,_lambda,X,y,X_cv,y_cv,epochs,patience=3000, min_delta=1e-4):
    for alpha in alpha_arr:
        W1,b1,W2,b2,W3,b3 = weight_bias_initialize()
        best_cv_error = float("inf")
        best_weights = None
        wait = 0
        for epoch in range(epochs):

            ##########Forward propagation##########
            z1 = X @ W1 + b1 #(2076,5)@(5,64)
            a1 = relu(z1)

            z2 = a1 @ W2 + b2 # (2076,64)@(64,32)
            a2 = relu(z2)

            z3 = a2 @ W3 + b3 #(2076,32)@(32,1)
            a3 = linear(z3)

            error = np.mean((a3 - y)**2 ) + _lambda * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
            ##########Backward propagation#############

            dz3 = a3 - y
            dW3 = a2.T @ dz3 + _lambda * W3
            db3 = np.sum(dz3,axis=0,keepdims = True)

            da2 = dz3 @ W3.T         
            dz2 = da2 * relu_derivative(z2)  
            dW2 = a1.T @ dz2 + _lambda * W2        
            db2 = np.sum(dz2, axis=0, keepdims=True)

            da1 = dz2 @ W2.T         
            dz1 = da1 * relu_derivative(z1)  
            dW1 = X.T @ dz1 + _lambda * W1        
            db1 = np.sum(dz1, axis=0, keepdims=True)


            ##########Update Weights####################
            W1 = W1 - alpha*dW1
            b1 = b1 - alpha*db1

            W2 = W2 - alpha*dW2
            b2 = b2 - alpha*db2

            W3 = W3 - alpha*dW3
            b3 = b3 - alpha*db3

            ##########Print error every 1000 Epochs#############
            if epoch % 1000 == 0:
                trainError,cvError = compute_final_errors(X, y, X_cv, y_cv, W1, b1, W2, b2, W3, b3)
                print(f"Epoch {epoch}, Train Error: {trainError:.4f}, Validation Error: {cvError:.4f}")
            ############# Early stopping logic###############
                if cvError + min_delta < best_cv_error:
                    best_cv_error = cvError
                    best_weights = (W1.copy(), b1.copy(), W2.copy(), b2.copy(), W3.copy(), b3.copy())
                    wait = 0
                else:
                    wait += 1000

                if wait >= patience:
                    print(f"Early stopping at epoch {epoch} for lambda {_lambda:.5f}. Best validation error: {best_cv_error:.4f}")
                    break

        if best_weights:
            W1, b1, W2, b2, W3, b3 = best_weights

        trainError,cvError = compute_final_errors(X, y, X_cv, y_cv, W1, b1, W2, b2, W3, b3)
        print(f"\nalpha:{alpha:.5f}")
        print(f"Train Error: {trainError:.4f}")
        print(f"Cross validation Error: {cvError:.4f} \n")
        print("***"*15,"\n")
    return (W1,b1,W2,b2,W3,b3)


# In[14]:


alpha_arr = np.array([0.0001,0.00003,0.00001])
_lambda = 0.0001
training_model_alpha(alpha_arr,_lambda,X_train_scaled,y_train,X_cv_scaled,y_cv,10000)


# In[15]:


alpha_arr = np.array([0.0001])
_lambda = 0.0001
W1,b1,W2,b2,W3,b3 = training_model_alpha(alpha_arr,_lambda,X_train_scaled,y_train,X_cv_scaled,y_cv,10000)


# In[18]:


def my_neuron_network(X,W1,b1,W2,b2,W3,b3):
    z1 = X @ W1 + b1 #(2076,5)@(5,64)
    a1 = relu(z1)

    z2 = a1 @ W2 + b2 # (2076,64)@(64,32)
    a2 = relu(z2)

    z3 = a2 @ W3 + b3 #(2076,32)@(32,1)
    a3 = linear(z3)
    return a3


# In[19]:


y_predict = my_neuron_network(X_test_scaled,W1,b1,W2,b2,W3,b3)

MSE_test =  np.mean((y_predict - y_test)**2)
r2_test = r2_score(y_test, y_predict)

print(f"MSE score:{MSE_test:.4f},r2 score:{r2_test:.3f}")

