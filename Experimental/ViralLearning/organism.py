#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 18:55:40 2020

@author: alf
"""

import numpy as np
import matplotlib.pyplot as plt

class celular_layer():
    
    def __init__(self, n_conn, n_cells, act_f):
        
        self.act_f  = act_f
        self.b      = np.random.rand(1, n_cells) * 2 - 1
        self.W      = np.random.rand(n_conn, n_cells) * 2 - 1
        
class Organism():
    
    def __init__(self, topology, act_f):
        self.celular_net = self.create_celular_net(topology, act_f)
        
    def create_celular_net(self, topology, act_f):
    
        nn = []
        
        for l, layer in enumerate(topology[:-1]):
            nn.append(celular_layer(layer, topology[l+1], act_f))
            
        return nn
    
    def forward_pass(self, X):
        out = [(None, X)]
        
        # Forward pass
        for l, layer in enumerate(self.celular_net):
            
            z = out[-1][1] @ self.celular_net[l].W + self.celular_net[l].b
            a = self.celular_net[l].act_f[0](z)
            
            out.append((z, a))
    
        return out
    
    def train(self, X, Y, l2_cost, lr=0.5):
    
        # Forward pass
        out = self.forward_pass(X)
        
        # Backward pass
        deltas = []
        _W = 0
        
        for l in reversed(range(0, len(self.celular_net))):
            
            z = out[l+1][0]
            a = out[l+1][1]
                
            if (l == len(self.celular_net)-1):
                # Calculate delta last layer
                deltas.insert(0, l2_cost[1](a, Y) * self.celular_net[l].act_f[1](a))
                
            else:
                # Calculate delta based on previous layer
                deltas.insert(0, deltas[0] @ _W.T * self.celular_net[l].act_f[1](a))
                
            _W = self.celular_net[l].W
            
            # Gradient descent
            self.celular_net[l].b = self.celular_net[l].b - np.mean(deltas[0], axis = 0, keepdims=True) * lr
            self.celular_net[l].W = self.celular_net[l].W - out[l][1].T @ deltas[0] * lr
            
        return out[-1][1]
    
    def plot_learning(self, epochs, X, Y, l2_cost, lr, loss = []):
        
        for i in range(epochs):
            pY = self.train(X, Y, l2_cost, lr)
            
            if i % 25 == 0:
                loss.append(l2_cost[0](pY, Y))
                
                res = 50
                
                _x0 = np.linspace(-1.5, 1.5, res)
                _x1 = np.linspace(-1.5, 1.5, res)
                
                _Y = np.zeros((res, res))
                
                for i0, x0 in enumerate(_x0):
                    for i1, x1 in enumerate(_x1):
                        _Y[i0, i1] = self.forward_pass(np.array([[x0, x1]]))[0][0]
                
                plt.pcolormesh(_x0, _x1, _Y, cmap="coolwarm")
                plt.axis("equal")
        
                plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c="skyblue")
                plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c="salmon")
                
                plt.show()
                plt.plot(range(len(loss)), loss)
                plt.show()
        
        return loss
                        
