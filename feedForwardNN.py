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
    4. Update weights and biases
    5. Repeat till no. of epochs
'''

#%%

import numpy as np
from math import e
from scipy.misc import derivative


#%%
def sigmoid(x):
    return 1/(1+e**(-x))

def tanh(x):
    return (e**x-e**(-x))/e**x+e**(-x)


#%%
class network:
    
    def __init__(self, layerN, activation=sigmoid, alpha=0.1):
        self.activation = activation
        self.layerN = layerN
        self.W = {}                                 #Will contain weight matrices. Each key represents a layer and corresponding value is weight matrix for that layer
        self.Z = {}                                 #Will contain Z vectors (i.e. XW + b). Each key represents a layer i and corresponding value is the vector Zi for that layer. Vector Zi has the Zij value for node j in layer i
        self.A = {}                                 #Will contain the activation vectors (i.e. f(Z)). Each key represents a layer i and corresponding value is the vector Ai for that layer. Vector Ai has the activation j for each node j in layer i
    
    #defining function which initializes weights and outputs the weight matrix for each layer
    #The parameter, layerN is a list containing the number of nodes in each layer, including the input and output layers. Eg. layerN = [10000, 5,5,10] means the input layer has 10000 nodes, there are two hidden layers with 5 nodes each and the ouput layer has 10 nodes.   
    def initialize_weights(self):
        for l in range(1, len(self.layerN)):
            self.W[l] =  np.asmatrix([np.random.normal(loc=0, scale=0.1, size=self.layerN[l]) for i in range(self.layerN[l-1])]) #generate n random nos from N(0,0.1) and store in layerN[l]xlayerN[l+1] dimension matrix
    
    #Perform forward propagation
    def forward_prop(self,X):                                                     #Compute Z vector for the first layer. Store it in key 1 of dictionary Z
        self.A[0] = X                                                             #Compute activation vector for the first layer. Store it in key 1 of dictionary A
        for i in range(1, len(self.W)+1):
            self.Z[i] = np.matmul(self.A[i-1], self.W[i])                                       #Compute Z vector for the layer i
            self.A[i] = np.asmatrix([self.activation(i) for i in self.Z[i].A1])                 #Compute activation vector for the layer i. The last activation vector is the predicted Y.
    
    #compute weight gradients using backpropagation        
    def compute_gradient(self, X, y):
        y-self.A[len(self.A)]
        return "weight_gradients"
    
    def fit(self, X, y, epoch=20):
        self.W = 'final_weights'
        
    def predict(self, X):
        return self.A[len(self.A)]
        
            
    
            
    
            