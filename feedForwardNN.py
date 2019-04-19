#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 22:04:56 2019

@author: sachitnagpal
"""
#%%

'''
Steps: 
    1. Initialize Weights Matrix
    2. Forward propagation
    3. Compute weight gradients using backpropagation
        i) Begin from output layer and calculate derivative wrt z for each output node
        ii) Compute the same for each node in previous layer recursively
        iii) Compute weight and bias gradients
    4. Repeat 2 and 3 for all observations
    5. Compute weight gradients by taking average and update weight matrices
    6. Repeat 2-5 till no. of epochs
'''

#%%
#Algorithm : http://deeplearning.stanford.edu/wiki/index.php/Backpropagation_Algorithm

#%%

import numpy as np
from math import e
from scipy.misc import derivative
from collections import deque
import operator

#%%

#function defined to add the values of dictionaries with same keys. We will use it to add the weights matrix and alpha times the weight gradient matrix
def combine_dicts(a, b, op=operator.sub):
    return {**a, **b, **{k: op(a[k], b[k]) for k in a.keys() & b}}

#A couple of activation functions
def sigmoid(x):
    return 1/(1+e**(-x))

def tanh(x):
    return (e**x-e**(-x))/e**x+e**(-x)


#%%
class network:
    
    #The parameter, layerN is a list containing the number of nodes in each layer, including the input and output layers. Eg. layerN = [10000, 5,5,10] means the input layer has 10000 nodes, there are two hidden layers with 5 nodes each and the ouput layer has 10 nodes.   
    def __init__(self, layerN, activation=sigmoid, alpha=0.1):
        self.activation = activation
        self.alpha = alpha
        self.layerN = layerN
        self.W = {}                                 #Will contain weight matrices. Each key represents a layer and corresponding value is weight matrix for that layer
        self.b = {}                                 #Will contain bias vectors. Each key represents a layer and corresponding value is bias vector for that layer
        self.Wg = {}                                #Will contain weight gradients. Each key represents a layer and corresponding value is weight gradients matrix for that layer
        self.bg = {}                                #Will contain bias gradients. Each key represents a layer and corresponding value is bias gradients vector for that layer
        self.Z = {}                                 #Will contain Z vectors (i.e. XW + b). Each key represents a layer i and corresponding value is the vector Zi for that layer. Vector Zi has the Zij value for node j in layer i
        self.A = {}                                 #Will contain the activation vectors (i.e. f(Z)). Each key represents a layer i and corresponding value is the vector Ai for that layer. Vector Ai has the activation j for each node j in layer i
    
    #defining function which initializes weights and generates the weight matrix for each layer
    def initialize_weights(self):
        for l in range(len(self.layerN)-1):
            self.W[l] =  np.array([np.random.normal(loc=0, scale=0.01, size=self.layerN[l]) for i in range(self.layerN[l+1])]) #generate n random nos from N(0,0.1) and store in layerN[l+1]xlayerN[l] dimension matrix
            self.Wg[l] = np.zeros(shape=[self.layerN[l+1], self.layerN[l]]) #generate n random nos from N(0,0.1) and store in layerN[l+1]xlayerN[l] dimension matrix
            self.b[l] = np.random.normal(loc=0, scale=0.01, size=self.layerN[l+1]).reshape(-1,1)
            self.bg[l] = np.zeros(shape=[1, self.layerN[l+1]]).reshape(-1,1) 
            
    #Perform forward propagation
    def forward_prop(self,X):                                                               #Compute Z vector for the first layer. Store it in key 1 of dictionary Z
        self.A[0] = np.array(X).reshape(-1,1)                                                             #Compute activation vector for the first layer. Store it in key 1 of dictionary A
        for i in range(1, len(self.layerN)):
            self.Z[i] = self.W[i-1]@self.A[i-1] + self.b[i-1]                                           #Compute Z vector for the layer i
            self.A[i] = np.array([self.activation(i) for i in self.Z[i]])                   #Compute activation vector for the layer i. The last activation vector is the predicted Y.
    
    #compute weight gradients using backpropagation        
    def compute_gradient(self,y):
        self.deltas = deque([])
        self.deltas.appendleft(np.multiply(-(y-self.A[len(self.A)-1]), derivative(self.activation, self.Z[len(self.Z)], dx=1e-6)))
        for i in range(len(self.layerN)-2,0, -1):
            self.deltas.appendleft(np.multiply(derivative(self.activation, self.Z[i]), self.W[i].T@self.deltas[0])) 
        
        #weights and bias gradient
        for i in range(len(self.W)-1,-1,-1):
            self.Wg[i] += self.deltas[i].reshape(-1,1)@self.A[i].reshape(1,-1)
            self.bg[i] += self.deltas[i]
        
    def fit(self, X, y, epoch=100):
        y=y.reshape(-1,1)
        self.initialize_weights()
        for j in range(epoch):
            for i in range(len(X)):
                self.forward_prop(X[i])
                self.compute_gradient(y[i])
            #Computing average gradient for all the observations combined
            self.Wg = {k:(v/len(X))*self.alpha for k,v in self.Wg.items()}
            self.bg = {k:(v/len(X))*self.alpha for k,v in self.bg.items()}
            #updating weights and biases
            self.W = combine_dicts(self.W,self.Wg)
            self.b = combine_dicts(self.b,self.bg)
        
    def predict(self, X):
        self.forward_prop(X)
        return self.A[len(self.A)-1]
    
#%%
from sklearn import datasets
import pandas as pd
df = datasets.load_iris()
r = np.random.choice(150, size=150, replace=False)
X = df.data[r]
y = np.array(pd.get_dummies(df.target))[r]

model = network(layerN=[X.shape[1], 3, 3, 3])
model.fit(X,y, epoch=1)

#%%
#testing the network on the iris dataset
from sklearn import datasets
import pandas as pd
df = datasets.load_iris()
r = np.random.choice(150, size=150, replace=False)
X = df.data[r]
y = np.array(pd.get_dummies(df.target))[r]

model = network(layerN=[X.shape[1], 3, 3, 3], alpha=0.1)
model.fit(X,y, epoch=100)
#%%
preds = []
for i in range(len(X)):
    a = model.predict(X[i])
    preds.append(df.target_names[np.argmax(a)])
#%%
#comparision of speeds in appending and calling for lists and dictionaries
from time import time

t = time()
a = {}
for i in range(10000000):
    a[i] = sigmoid(i)
print(time()-t)
    
t = time()
b = []
for i in range(10000000):
    b.append(sigmoid(i))
print(time()-t)

t = time()
c = np.array([np.empty((1,10000000)), np.empty((1,10000000))])
for i in range(10000000):
    c[0][0,i] = sigmoid(i)
print(time()-t)

c = np.empty((1,10000000))
t = time()
for i in range(10000000):
    c[0,i] = sigmoid(i)
print(time()-t)


t = time()
for i in range(10000000):
    m = a[i]
print(time()-t)
    
t = time()
for i in range(10000000):
    m = b[i]
print(time()-t)


t = time()
for i in range(10000000):
    m = c[0,i]
print(time()-t)


            
    
            