#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:04:50 2020

@author: alf
"""
import cell
import math
import numpy as np

class cell_system:
    
    identifier          = ''
    cell_net = {}
    cells               = {}
    input_links         = {}
    output_links        = {}
    f_act               = (lambda x: 1 / (1 + np.e ** (-x)),
                           lambda x: x * (1 - x))
    f_cost              = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
                           lambda Yp, Yr: (Yp - Yr))
    learning_rate       = 0.5
    
    def __init__(self, identifier, cell_net, cells, input_links, output_links, learning_rate = 0.2):
        
        self.identifier             = identifier
        self.cell_net               = cell_net
        self.cells                  = cells
        self.input_links            = input_links
        self.output_links           = output_links
        self.learning_rate          = learning_rate

    def communicate_decisition(self, cell_decided, output_data):
        
        for cell_in_net in self.output_links[cell_decided.identifier]:
            
            if self.cells[cell_in_net].combined_data.get(cell_decided.identifier) == None:
                self.cells[cell_in_net].combined_data[cell_decided.identifier] = output_data * self.output_links[cell_decided.identifier][cell_in_net]

    def decide(self, input_data, remove_garbage = 0):
        
        for i in range(len(input_data)):
            
            elected_cell = self.cells[(list(self.output_links['input_{}'.format(i)].keys())[0])]
            decisition = elected_cell.decide([input_data[i]])
            elected_cell.combined_data['input_{}'.format(i)] = input_data[i]
            self.communicate_decisition(elected_cell, decisition)
        
        for individual_cell in self.cells:
            try:
                if list(self.cells[individual_cell].combined_data.keys()).sort() == list(self.input_links[individual_cell].keys()).sort():
                    elected_individual_cell = self.cells[individual_cell]
                    decisition = elected_individual_cell.decide(list(elected_individual_cell.combined_data.values()))
                    self.communicate_decisition(elected_individual_cell, decisition)
            except KeyError as ke:
                last_cell = individual_cell
        
        result = self.cells[last_cell].decide(list(self.cells[last_cell].combined_data.values()))
        
        if (remove_garbage):
            for aux_cell in list(self.cells.values()):
                aux_cell.to_another_thing_butterfly()
        
        return result

    def get_input_links_from_cell(self, identifier):

        links = []

        for link in self.input_links[identifier]:
            links.append(self.input_links[identifier][link])

        return np.array(links) 


    def learn(self, input, output):

        # Typical forward pass
        output_try  = self.decide(input)
        deltas_list = []
        counter     = 0

        # Correct the weight of the links by backpropagation
        for i in reversed(range(len(self.cell_net))):
            
            layer = self.cell_net['l{}'.format(i)]
            errors = []
            
            if counter != 0:
                for j in layer:
                    error = 0.0
                    for cell_aux in self.cell_net['l{}'.format(i + 1)]:
                        error += (self.input_links[cell_aux][j] * self.cells[cell_aux].delta)
                        errors.append(error)
                
            else:
                for j in layer:
                    cell_aux = self.cells[j]                    
                    errors.append(output[0] - cell_aux.decide(list(cell_aux.combined_data.values())))
            
            for j in layer:
                cell_aux = self.cells[j]
                cell_aux.delta = errors[layer.index(j)] * self.f_act[1](cell_aux.decide(list(cell_aux.combined_data.values())))
            
            counter += 1
            
        self.update_connections()
        
        for cell_aux in self.cells.values():
            cell_aux.to_another_thing_butterfly()
        
    def update_connections(self):        
        counter = 0
        
        for i in list(self.cell_net.values()):
            if counter != 0:
                for cell_aux in i:
                    for j in list(self.input_links[cell_aux].keys()):
                        
                        aux = self.input_links[cell_aux][j]
                        self.input_links[cell_aux][j] += self.learning_rate * self.cells[cell_aux].delta * aux
            
            counter += 1
            
        for i in list(self.cell_net.values()):
            for cell_aux in i:
                for input_link in self.input_links[cell_aux]:
                    self.output_links[input_link][cell_aux] = self.input_links[cell_aux][input_link]
  
    def replicate():
        pass
        
            
    
