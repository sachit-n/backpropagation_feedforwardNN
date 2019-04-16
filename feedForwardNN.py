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
        i) Begin from output layer and find delta for each output node
        ii) Compute delta for each node in previous layer recursively
        iii) Compute weight and bias gradients
    4. Update weights and biases
    5. Repeat till no. of epochs
'''

#%%

import numpy as np
from math import e


#%%
def sigmoid(x):
    return 1/(1+e**x)


#%%
class network:

    def __init__(self):
       self.W = {}              #Will contain weight matrices. Each key represents a layer and corresponding value is weight matrix for that layer
       self.Z = {}              #Will contain Z vectors (i.e. XW + b). Each key represents a layer i and corresponding value is the vector Zi for that layer. Vector Zi has the Zij value for node j in layer i
       self.A = {}              #Will contain the activation vectors (i.e. f(Z)). Each key represents a layer i and corresponding value is the vector Ai for that layer. Vector Ai has the activation j for each node j in layer i
    
    #defining function which initializes weights and outputs the weight matrix for each layer
    #The parameter, layerN is a list containing the number of nodes in each layer, including the input and output layers. Eg. layerN = [10000, 5,5,10] means the input layer has 10000 nodes, there are two hidden layers with 5 nodes each and the ouput layer has 10 nodes.
    def initialize_weights(self, layerN):
        for l in range(len(layerN)-1):
            self.W[l+1] =  np.asmatrix([np.random.normal(loc=0, scale=0.1, size=layerN[l+1]) for i in range(layerN[l])]) #generate n random nos from N(0,0.1) and store in layerN[l]xlayerN[l+1] dimension matrix


      
    def forward_prop(self,X,W, activation=sigmoid):
        self.Z[1] = np.matmul(X, W[1])
        self.A[1] = [sigmoid(-i) for i in self.Z[1].A1]
        for i in range(2, len(W)+1):
            self.Z[i] = np.matmul(self.A[i-1], self.W[i])
            self.A[i] = [1/(1+e**i) for i in self.Z[i].A1]
            