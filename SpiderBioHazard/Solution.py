from dataclasses import dataclass
import numpy as np

@dataclass(init=True, repr=True)
class Solution:
    """Solution class is a generic class to model the form and proccess that
        affects to the solutions of the optimization problem. You should overwrite
        the fitness_function and parasite_mutation methods to use this implementation
        of the sos algorithm.

        functions:
            __init__: Initialization method for the class.
                -> attribute_list (np.array):       This argument should be the list of 
                                                    characteristics that the solution had
                                                    and should use to calculate the fitness.
                -> interval_by_attr (np.array):     This argument should be a list of tuples
                                                    which each tuple should be the maximum 
                                                    and minimum value for a determinated 
                                                    attribute in attribute_list.
                -> _fitness_function (function):    This argument should be an executable
                                                    function to compute the fitness value
                                                    of the solution.
                -> _random_function (function):     This argument should be an executable
                                                    function to compute a complete 
                                                    randomized solution.
                -> _show_function (function):       This argument should be an executable
                                                    function to show the solution by 
                                                    screen. 
                -> _mutation_function (function):   This argument should be an executable
                                                    function to compute a mutation over 
                                                    solution.
                                                 
            __add__: Method to overload the operator +. It's need to overwrite it in order
                     to keep this algorithm generic to perform more than numerical 
                     optimizations. 

            ___sub_: Method to overload the operator -. It's need to overwrite it in order
                     to keep this algorithm generic to perform more than numerical 
                     optimizations. 


            fitness_function: Method that should return the computed value of fitness from
                              the solution. 
            
            mutation_function: Method that should process the mutation of the current 
                               solution's attributes. 

            randomize_function: Method that will randomize every attribute of the current
                                solution.

            show_solution: Method that will show the solution by screen.

        attributes:
            attribute_list (np.array):          This attribute are all the characteristics that had
                                                the solution to work with.

            interval_by_attr (np.array):        This attribute are a list of tuples that contains
                                                the maximum and minimum value for the attribute_list
                                                at the same index.

            _fitness_function (function):       This attribute is a function,  which its stored 
                                                the function to compute the fitness value of 
                                                the solution.

            _random_function (function):        This attribute is a function,  which its stored 
                                                the function to compute a completely random solution.
                                                
            _show_function (function):          This attribute is a function,  which its stored
                                                the function to show the solution by screen.
                                                
            _mutation_function (function):      This attribute is a function,  which its stored 
                                                the function to perform the mutation of the 
                                                current solution.
            
    """
    attribute_list: np.array
    interval_by_attr: np.array

    _fitness_function: callable
    _random_function: callable
    _show_function: callable
    _mutation_function: callable

    _add_funct: callable = None
    _sub_funct: callable = None
    _mul_funct: callable = None
    _div_funct: callable = None

    def __add__(self, other):
        return self._add_funct(self, other) 

    def __sub__(self, other):
        return self._sub_funct(self, other) 

    def __mul__(self, other):
        return self._mult_funct(self, other) 

    def fitness_function(self):
        return self._fitness_function(self)

    def mutation_function(self):
        return self._mutation_function(self) 
    
    def randomize_function(self):
        return self._random_function(self)

    def show_solution(self):
        return self._show_function(self)

    def correct_solution_limits(self, type_data):
        for attr_index, attr in enumerate(self.attribute_list):
            if (attr > self.interval_by_attr[attr_index][1]):
                self.attribute_list[attr_index] = self.interval_by_attr[attr_index][1]
            
            elif (attr < self.interval_by_attr[attr_index][0]):
                self.attribute_list[attr_index] = self.interval_by_attr[attr_index][0]
        
        try:
            exec(f"self.attribute_list = self.attribute_list.astype({type_data})")
        except Exception as e:
            print(e)
