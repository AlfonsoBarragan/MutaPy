#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:33:11 2020

@author: alf
"""

import math

class cell:
    
    identifier          = ''
    internal_operation  = ''
    bias                = 0
    delta               = 0
    activation          = ''
    combined_data       = {}
    energy_required     = 0
    energy_obtained     = 0
    
    def __init__(self, identifier, energy_required, activation, internal_operation, combined_data = {}, energy_obtained = 0, bias = 1):
        
        self.identifier         = identifier
        self.energy_required    = energy_required
        self.internal_operation = internal_operation
        self.bias               = bias
        self.combined_data      = combined_data
        self.energy_obtained    = energy_obtained
        self.activation         = activation
        
    def decide(self, input_data):
        def int_operation() : pass
        def neuron_activation(): pass

        int_operation.__code__ = self.internal_operation.__code__
        neuron_activation.__code__ = self.activation.__code__
        
        return neuron_activation(int_operation(input_data))
    
    def continue_living(self):
        
        return self.energy_obtained >= self.energy_required
    
    def to_another_thing_butterfly(self):
        self.combined_data = {}
                      
                  