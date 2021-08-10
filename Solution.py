class Solution:
    """Solution class is a generic class to model the form and proccess that
        affects to the solutions of the optimization problem. You should overwrite
        the fitness_function and parasite_mutation methods to use this implementation
        of the sos algorithm.

        functions:
            __init__: Initialization method for the class.
                -> attribute_list (list):           This argument should be the list of 
                                                    characteristics that the solution had
                                                    and should use to calculate the fitness.
                -> interval_by_attr (list):         This argument should be a list of tuples
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
            attribute_list (list):              This attribute are all the characteristics that had
                                                the solution to work with.

            interval_by_attr (list):            This attribute are a list of tuples that contains
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
    _fitness_function   = lambda x:1
    _random_function    = lambda x:1
    _show_function      = lambda x:1
    _mutation_function  = lambda x:1

    _add_funct = lambda x:1 
    _sub_funct = lambda x:1
    _mul_funct = lambda x:1
    _div_funct = lambda x:1

    attribute_list      = []
    interval_by_attr    = []

    def __init__(self, attribute_list:list, interval_by_attr:list,
                    _fitness_function, _random_function, _show_function, 
                    _mutation_function):

        self.attribute_list         = attribute_list
        self.interval_by_attr       = interval_by_attr
        self._fitness_function      = _fitness_function
        self._random_function       = _random_function
        self._show_function         = _show_function
        self._mutation_function     = _mutation_function

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

