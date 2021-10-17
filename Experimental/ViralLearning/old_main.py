#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 19:29:58 2020

@author: alf
"""


import copy
import math
import random

def add(input_data):
    total = 0
    for element in input_data:
        total += element

    return total/len(input_data)

def activate(input):
    return 1 / (1 + math.exp(-input))

def cuadratic_error(out_pred, out_real):
    return 1/2 * (out_real - out_pred)**2

def clone_cells(individual_cell, n_clones):
    cell_net = {}

    for i in range(n_clones):
        aux_cell = copy.deepcopy(individual_cell)
        aux_cell.identifier = 'c{}'.format(i)

        cell_net['c{}'.format(i)] = copy.deepcopy(aux_cell)


    return cell_net

def train_network(network, train, l_rate, n_epoch, expected):
    for epoch in range(n_epoch):
        outputs = []
        sum_error = 0
        for row in train:
            network.learn(row[:2], row[2:])
            outputs.append(network.decide(row[:2], 1))
        sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


train_data = [[1,1,1], [1,0,0], [0,1,0], [0,0,0], 
              [2,2,1], [2,1,0], [7,1,0], [9,0,0],
              [3,3,1], [3,2,0], [5,1,0], [0,9,0],
              [4,4,1], [4,5,0], [6,1,0], [0,5,0],
              [5,5,1], [2,5,0], [3,1,0], [0,3,0],]

expected = [1,0,0,0]

cell_base = cell('c{}', 1, activate, add)

cell_net = {'l0':['c0', 'c1'],'l1':['c2','c3'], 'l2':['c4']}

cells = clone_cells(cell_base, 5)

input_links = {'c0':{'input_0': 1}, 
               'c1':{'input_1': 1}, 
               'c2':{'c0': random.random(), 
                     'c1':random.random()}, 
               'c3':{'c0': random.random(), 
                     'c1':random.random()}, 
               'c4':{'c2': random.random(), 
                     'c3': random.random()}}

output_links = {'input_0':{'c0': 1}, 
                'input_1':{'c1': 1}, 
                'c0':{'c2': random.random(), 
                      'c3': random.random()}, 
                'c1':{'c2': random.random(), 
                      'c3': random.random()}, 
                'c2':{'c4': random.random()}, 
                'c3':{'c4': random.random()}}


cell_system_alpha = cell_system('cs_alpha', cell_net, cells, input_links, output_links, 0.1)