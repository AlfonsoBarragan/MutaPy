import copy
import random
import functools

# Symbiotic Organisms Search Algorithm
## Paper's title: Symbiotic Organisms Search: A new metaheuristic optimization algorithm
## DOI: https://doi.org/10.1016/j.compstruc.2014.03.007
## Authors: Min-Yuan Cheng, Doddy Prayogo
## Year: 2014

#TODO Ending documentation

class Ecosystem:
    """ Ecosystem its a generic class that can manage all the Symbiotic Organisms Search
    Algorithm (SOS). 
    
        functions:
            __init__: Initialization method for the class.
            
                -> termination_criteria(function)   : This argument should be an executable
                                                      function to compute the fitness value
                                                      of the organism (solution).

                -> organisms (list)                 : This argument should be the list of 
                                                      characteristics that the solution had
                                                      and should use to calculate the fitness.
                                                   
                -> best_organism_funct (function)   : This argument should be an executable
                                                      function to compute the fitness value
                                                      of the organism (solution).

                -> number_of_organisms (int)        : This argument should be the list of 
                                                      characteristics that the solution had
                                                      and should use to calculate the fitness.

                -> n_iteration (int)                : This argument should be the list of 
                                                      characteristics that the solution had
                                                      and should use to calculate the fitness.
    """

    # Ecosystem characteristics
    n_iteration             = 0
    termination_criteria    = lambda x:1

    # Organisms characteristics
    number_of_organisms     = 0
    organisms               = [] 
    best_organism_funct     = lambda x:1

    def __init__(self, organisms:list, best_organism_funct:callable, 
                       termination_criteria:callable):
        self.termination_criteria    = termination_criteria
           
        self.number_of_organisms     = len(organism_list) 
        self.organisms               = organisms
        self.best_organism_funct     = best_organism_funct

         
    def identify_best_organism(self):
        return functools.reduce(self.best_organism_funct, self.organisms)

    def mutualism_phase(self, Xi, X_best):
        # Organisms selection
        Xj          = random.choice(copy.deepcopy(self.organisms).remove(Xi))

        Xi_final    = Xi
        Xj_final    = Xj

        # Beneficial factor choose
        BF1 = random.choice([1,2])
        BF2 = random.choice([1,2].remove(BF1))

        # Compute new mutualist organisms
        mutual_vector = (Xi + Xj)/2
        Xi_new = Xi + (random.uniform(0,1) * (X_best * mutual_vector))
        Xj_new = Xj + (random.uniform(0,1) * (X_best * mutual_vector))   

        if (Xi.fitness_function() < Xi_new.fitness_function()):
            Xi_final = copy.deepcopy(Xi_new)

        if (Xj.fitness_function() < Xj_new.fitness_function()):
            Xj_final = copy.deepcopy(Xj_new)
        

        self.organisms[i]                           = copy.deepcopy(Xi_final)         
        self.organisms[self.organisms.index(Xj)]    = copy.deepcopy(Xj_final)         

    def commensalism_phase(self, Xi, X_best):
        Xj          = random.choice(copy.deepcopy(self.organisms).remove(Xi))

        Xi_final    = Xi

        Xi_new      = Xi + (random.uniform(-1,1) * (X_best - Xj))
        
        if (Xi.fitness_function() < Xi_new.fitness_function()):
            Xi_final = Xi_new

        self.organisms[i] = copy.deepcopy(Xi_final) 

    def parasitism_phase(self, Xi):
        Xj          = random.choice(copy.deepcopy(self.organisms).remove(Xi))
        
        Xj_final    = Xj

        Xi.parasite_mutation()

        if (Xi.fitness_function() > Xj.fitness_function()):
            Xj_final = Xi

        self.organisms[self.organisms.index(Xj)] = copy.deepcopy(Xj_final)         

    def execute(self, iterations:int):
        while(self.n_iteration < iterations):
        
            for i in range(len(self.organisms)):
                Xi = self.organisms[i % len(self.organisms)]
                X_best      = self.identify_best_organism()
        
                self.mutualism_phase(Xi, X_best)

                self.commensalism_phase(Xi, X_best)

                self.parasitism_phase(Xi)

            self.n_iteration += 1


class Organism:
    """Organism class is a generic class to model the form and proccess that
        affects to the solutions of the optimization problem. You should overwrite
        the fitness_function and parasite_mutation methods to use this implementation
        of the sos algorithm.

        functions:
            __init__: Initialization method for the class.
                -> _fitness_function (function): This argument should be an executable
                                                 function to compute the fitness value
                                                 of the organism (solution).
                -> attribute_list (list)       : This argument should be the list of 
                                                 characteristics that the solution had
                                                 and should use to calculate the fitness.

            __add__: Method to overload the operator +. It's need to overwrite it in order
                     to keep this algorithm generic to perform more than numerical 
                     optimizations. 

            ___sub_: Method to overload the operator -. It's need to overwrite it in order
                     to keep this algorithm generic to perform more than numerical 
                     optimizations. 


            fitness_function: Method that should return the computed value of fitness from
                              the organism (solution). OVERWRITE IT!
            
            parasite_mutation: Method that should process the mutation of the attributes
                               that had the organism in order to create a parasitism
                               vector and make the parasitism phase. OVERWRITE IT!

        attributes:
            attribute_list (list): This attribute are all the characteristics that had
                                   the solution to work with.

    """
    _fitness_function   = lambda x:1
    attribute_list      = []

    def __init__(self, attribute_list:list):
       self.attribute_list      = attribute_list

    def __add__(self, other):
        return 1

    def __sub__(self, other):
        return 1
    def fitness_function(self):
        # OVERWRITE IT!
        return 1

    def parasite_mutation(self):
        # OVERWRITE IT!
        return 1


