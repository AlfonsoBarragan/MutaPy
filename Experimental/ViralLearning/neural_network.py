#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 17:23:47 2020

@author: alf
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

class neural_layer():
    
    def __init__(self, n_conn, n_neur, act_f):
        
        self.act_f  = act_f
        self.b      = np.random.rand(1, n_neur) * 2 - 1
        self.W      = np.random.rand(n_conn, n_neur) * 2 - 1
        
def create_nn(topology, act_f):
    
    nn = []
    
    for l, layer in enumerate(topology[:-1]):
        nn.append(neural_layer(layer, topology[l+1], act_f))
        
    return nn



p = 2

topology = [p, 4, 8, 16, 8, 4, 1]


def train(neural_net, X, Y, l2_cost, lr=0.5, train=True):
    
    out = [(None, X)]
    
    # Forward pass
    for l, layer in enumerate(neural_net):
        
        z = out[-1][1] @ neural_net[l].W + neural_net[l].b
        a = neural_net[l].act_f[0](z)
        
        out.append((z, a))

    # Backward pass
    deltas = []
    _W = 0
    if train:
        for l in reversed(range(0, len(neural_net))):
            
            z = out[l+1][0]
            a = out[l+1][1]
                
            if (l == len(neural_net)-1):
                # Calculate delta last layer
                deltas.insert(0, l2_cost[1](a, Y) * neural_net[l].act_f[1](a))
                
            else:
                # Calculate delta based on previous layer
                deltas.insert(0, deltas[0] @ _W.T * neural_net[l].act_f[1](a))
                
            _W = neural_net[l].W
            
            # Gradient descent
            neural_net[l].b = neural_net[l].b - np.mean(deltas[0], axis = 0, keepdims=True) * lr
            neural_net[l].W = neural_net[l].W - out[l][1].T @ deltas[0] * lr
        
    return out[-1][1]


red = create_nn(topology, sigm)

## DATASET ##


#%%

for i in range(2500):
    pY = train(red, X, Y, l2_cost, lr=0.1)
    
    if i % 25 == 0:
        loss.append(l2_cost[0](pY, Y))
        
        res = 50
        
        _x0 = np.linspace(-1.5, 1.5, res)
        _x1 = np.linspace(-1.5, 1.5, res)
        
        _Y = np.zeros((res, res))
        
        for i0, x0 in enumerate(_x0):
            for i1, x1 in enumerate(_x1):
                _Y[i0, i1] = train(red, np.array([[x0, x1]]), Y, l2_cost, train = False)[0][0]
        
        plt.pcolormesh(_x0, _x1, _Y, cmap="coolwarm")
        plt.axis("equal")

        plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c="skyblue")
        plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c="salmon")
        
        plt.show()
        plt.plot(range(len(loss)), loss)
        plt.show()
                
    
    