#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 19:06:02 2020

@author: alf
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

# Funciones activación        
sigm = (lambda x: 1 / (1 + np.e ** (-x)),
        lambda x: x * (1 - x))


# Funciones de coste

l2_cost = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
           lambda Yp, Yr: (Yp - Yr))

# Generación de datasets

n = 500
p = 2

X, Y = make_circles(n_samples=n, factor=0.5, noise=0.05)
Y = Y[:, np.newaxis]

plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c="skyblue")
plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c="salmon")
plt.axis("equal")
plt.show()

loss = []