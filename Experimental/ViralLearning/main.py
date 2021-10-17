
from cell import cell
from cell_system import cell_system

from organism import Organism
import resources as r

import numpy as np


X = np.array([[1,1],[1,0],[0,1],[0,0]])
Y = np.array([[1],[0],[0],[0]])

organismo = Organism([2, 8, 16, 32, 16, 8, 1], r.sigm)

loss_aux = []


#%%
loss = organismo.plot_learning(1000, X, Y, r.l2_cost, 0.5, loss_aux)
loss_aux = loss