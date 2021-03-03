# import section
import numpy as np


class Cellular_Automaton:
""" Generic class to create custom celullar automatons
"""
    def __init__(self, population:np.array, transition_rule:callable, neighbourhood:list):
        """ Initialization method of the class. 

        Parameters:
        population (np.array):          A numpy array initializate in the way that the user desires.

        transition_rule (callable):     A function kind object that contains the rules of transition of the cells from the automaton.
                                        This function should take two parameters:
                                            -> A population element / a cell. 
                                            -> The neighbourhood to work with.

                                        and finally should return:
                                            -> The final state of the giving cell with the correspondant neighbourhood.

        neighbourhood (list):           An int list object that contains the neighbourhood to look at in order to take the automaton
                                        to the next state.

        Returns:
        A Cellular_Automaton object kind

        """
        self.population         = population
        self.transition_rule    = transition_rule
        self.neighbourhood      = neighbourhood
        self._iterate           = compute_iterations()

    def compute_iterations(self):
        """ This function generates automatically the loop over the population and compute the transition rule giving on __init__ method.

        Parameters:
        This function doesn't need any parameter.

        Returns:
        This functions doesn't return anything. It just assigns iterate function 
        as an attribute of the class in order to use in the iterate function.
        
        """

        dimensions = len(self.population.shape)
        
        funct_iterate   = "def _iterate(self):\n"
        funct_iterate   += "\t pop_cpy = copy.deepcopy(self.population)\n"

        index_str       = ""

        for dim in range(dimensions):
            funct_iterate += "{0}for index_{1} in range(self.population.shape[{1}]):\n".format('\t'*i+1, i)
            index_str += "index_{},".format(i)

        funct_iterate += "{0}pop_cpy[{1}] = self.transition_rule(self.population[{1}], self.neighbourhood)\n".format('\t'*dimensions, index_str)
        funct_iterate += "\treturn pop_cpy\n"


    def iterate(self, n_iter:int):
        """ This functions execute the transition rule over all the population (throught _iterate) a giving number of iterations.

        Parameters:
        n_iter (int): The integer number of iterations to execute the celullar automaton.

        Returns:
        This function doesn't return anything. It just updates the population and 
        shows it.

        """
        for i in range(n_iter):
            self.population = self._iterate()
            print(self.to_string)
            
    def to_string(self):
        """ This functions shows the state of the population.

        Parameters:
        This function doesn't need any parameter at all.

        Returns:
        The string representation of the population that contains the Cellular_Automaton
        object in a certain instant.

        """
        return np.array2string(self.population, max_line_width=10, separator='|')

