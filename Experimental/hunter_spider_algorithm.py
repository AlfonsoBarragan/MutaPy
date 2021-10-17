# import section
import copy
import random
import math
import itertools
import functools

# explanation section

"""
A ver la vaina esta se me ocurrio una noche loca.

Cual es la mejor tela que puede componer una spider para capturar a su
presa? 

Teniendo una cantidad limitada de tela gastable X, determinar cual es la
estructura mas optima para la captura de una determinada presa.

-> Algoritmo de composicion de telas
-> Fitness de esa tela con respecto a las "presas"
-> Optimizacion de la tela con limitantes


"""
gen_dict_from_int   = lambda x: {item:0 for item in range(x)}
flatten_list        = lambda t: [item for sublist in t for item in sublist]

coordinates_oper        = lambda x1, x2: math.pow((x2 - x1), 2)
sum_funct               = lambda x, y: x+y   
distance_between_points = lambda p1, p2: math.sqrt(functools.reduce(sum_funct, map(coordinates_oper, p1, p2)))   
    

# implementation section
class Hunting_Spider:

    maximum_cost = 0
    

    def __init__(self, maximum_cost):
        self.maximum_cost = maximum_cost


    def create_web(self, prey):
        # To create a web the spider will move over all the 
        # different dimensions of the solution, droping 
        # spider silk (accumulating cost)
        
        # Parameter extraction from solution
        n_dimensions = prey.n_pars
        self.web = {}

        for dim_ind, dim_cont in enumerate(prey.regis_par):
            dim_cont.sort()
            maximum = random.uniform(dim_cont[0], dim_cont[len(dim_cont)-1])

            self.web[dim_ind] = [maximum * i for i in range(n_dimensions)] 
            print(self.web[dim_ind])


        cost = 0
        
        web_points = self._generate_web_points(n_dimensions)
        self.web = self._connect_web_points(web_points)
        for points in self.web:
            cost += list(map(distance_between_points, self.web[0], self.web[1]))[0]

        self.actual_cost = cost

    def _generate_web_points(self, n_dim):
        web_points_coord = {}
        dimension_list = []
        
        for i in self.web:
            dimension_list.append(self.web[i])

        dimension_list = flatten_list(dimension_list)
        web_points = list(set(itertools.combinations(dimension_list, n_dim)))

        for i in range(n_dim):
            filter_func_i = lambda x: x[i] in self.web[i] 
            web_points = list(filter(filter_func_i, web_points))

        return web_points
    
    def _connect_web_points(self, web_points):
        last_point = random.choice(web_points)
        _web_points = copy.deepcopy(web_points)

        _web_points.remove(last_point)
        path_traveled = []

        while _web_points != []:
            try:
                filter_last_point_connected = lambda x: (x[0] == last_point[0] or x[1] == last_point[1] or x[2] == last_point[2])
                candidate_points = list(filter(filter_last_point_connected, copy.deepcopy(_web_points)))
            
                new_point = random.choice(candidate_points)

                path_traveled.append((last_point, new_point))
                last_point = new_point

                _web_points.remove(last_point)
            except Exception:
                path_traveled   = []
                last_point      = random.choice(web_points)
                _web_points     = copy.deepcopy(web_points)
 
        return path_traveled
        

    def optimize_web(self, prey_array, prey_model):
        # Here we will train and upgrade the web to catch more efficiently
        # the preys. In order to make it, we will simulate to catch a 
        # group of the objective preys.
        maximum_lists = []
        for register in prey_model.regis_par:
            maximum_lists.append(register[len(register)-1])



        for prey_aux in prey_array:
            

            pass




class Prey:

    """ prey its a generic class to encapsulate the format of the registers
        that will be "hunted" by the spider.
    """
    
    def __init__(self, regis_par:list):
        """ Method to initialize the prey class
        
            parameters:
                regis_par (list): regis_par from register parameters. This
                                  is a list of sublists which will contain
                                  the different values from every
                                  parameter of the solution.

        """
        self.regis_par      = regis_par
        self.n_pars         = len(regis_par)





