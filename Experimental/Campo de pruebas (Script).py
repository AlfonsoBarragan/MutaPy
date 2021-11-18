#!/usr/bin/env python
# coding: utf-8

# # Pruebas de algoritmos inspirados en naturaleza
# Este cuaderno es para implementar los algoritmos inspirados en naturaleza para hacer experimentos y demas. Tambien se implementaran las metricas para medir sus capacidades.

# In[1]:


import math
import copy
import random
import functools
import numpy as np

from SpiderBioHazardLink import Solution
from Solution import Solution
from SpiderBioHazardLink import NatureInspiredAlgorithms

from numpy.random import choice
from sos_algorithm import Organism
from Galfgets import printProgressBar, mean, standard_desviation
from datetime import datetime


# In[42]:


int_by_attr = [number for number in range(0, 244)]

sum_lamb = lambda x: x[0]+x[1]

#aux = list(map(sum_lamb, int_by_attr))

print(int_by_attr[:1] + int_by_attr[1+1 :])


# In[73]:


def param_sol(size):
    # attrs = list(range(0, size))
    attrs = [n for n in range(1, size+1)]
    int_by_attr = [(1, size-1) for number in range(1, size+1)]
    
    return attrs, int_by_attr


def fitness_funct(self):
    sum_lamb    = lambda x: abs(x[0]-x[1])
    attack = 0

    for queen_col, queen_row in enumerate(self.attribute_list):
        queen_pos = [queen_col+1, queen_row]

        
        conf_attacks_dia  = lambda x: abs(x[0] - queen_pos[0]) - abs(x[1] - queen_pos[1])
        conf_attacks_line = lambda x: (x == queen_pos[1])     

        aux = [[index+1, element] for index, element in enumerate(self.attribute_list)]

        
        attacks_1 = map(conf_attacks_dia, aux[:queen_col] + aux[queen_col+1 :])
        attacks_2 = map(conf_attacks_line, self.attribute_list[:queen_col] + self.attribute_list[queen_col+1 :])

        attacks_1 = list(attacks_1)
        attacks_2 = list(attacks_2)
        
        final_attacks = [True for index,_ in enumerate(attacks_1) if (attacks_1[index] == 0) or (attacks_2[index] == True)]

        attack += (final_attacks.count(True))
        
        
    return attack

def fitness_funct2(self):
    sum_lamb    = lambda x: abs(x[0]-x[1])
    attack = 0

    for queen_col, queen_row in enumerate(self.attribute_list):
        queen_pos = [queen_col+1, queen_row]

        
        conf_attacks_dia  = lambda x: abs(x[0] - queen_pos[0]) - abs(x[1] - queen_pos[1])
        conf_attacks_line = lambda x: (x == queen_pos[1])     

        aux = [[index+1, element] for index, element in enumerate(self.attribute_list)]

        
        attacks_1 = map(conf_attacks_dia, aux[:queen_col] + aux[queen_col+1 :])
        attacks_2 = map(conf_attacks_line, self.attribute_list[:queen_col] + self.attribute_list[queen_col+1 :])

        attacks_1 = list(attacks_1)
        attacks_2 = list(attacks_2)
        
        final_attacks = [True for index,_ in enumerate(attacks_1) if (attacks_1[index] == 0) or (attacks_2[index] == True)]

        attack += (final_attacks.count(True))
        
        
    return attack

def random_funct(self):
    
    for index, attr in enumerate(self.attribute_list):
        self.attribute_list[index] = random.randint(self.interval_by_attr[index][0], self.interval_by_attr[index][1])
        

def show_function(self):
    
    str_to_show = "|"
    
    size = len(self.attribute_list)
    #TODO
        
def mutation_function(self):
    n_dim = random.randint(0, len(self.attribute_list))
    
    dim = list(range(len(self.attribute_list)))
    
    for i in range(n_dim):
        random_dim = random.choice(dim)
        dim.remove(random_dim)
        
        self.attribute_list[random_dim] = random.randint(self.interval_by_attr[random_dim][0],
                                                         self.interval_by_attr[random_dim][1])
    #TODO

def add_sols(self, other):
    add = lambda x: x[0]+x[1]
    pair_sols = list(zip(self.attribute_list, other.attribute_list))
    
    return list(map(add, pair_sols))

def subs_sols(self, other):
    subs = lambda x: x[0]-x[1]
    pair_sols = list(zip(self.attribute_list, other.attribute_list))
    
    return list(map(subs, pair_sols))

def mult_sols(self, other):
    mult = lambda x: x[0]*x[1]
    pair_sols = list(zip(self.attribute_list, other))
    
    return list(map(mult, pair_sols))

def div_sols(self, other):
    div = lambda x: x[0]/x[1]
    pair_sols = list(zip(self, other))
    
    return list(map(div, pair_sols))

def generate_population(n_organisms, n_params):
    attrs, int_by_attr = param_sol(n_params)
    fit_function = lambda x: fitness_funct(x)
    random_function = lambda x: random_funct(x)

    n_queens_sol = Solution(fit_function, random_function, show_function, mutation_function, 
                            add_sols, subs_sols, mult_sols, div_sols, attrs, int_by_attr)
    
    n_queens_sol.particle_dict = {'pBest':0, 'pBest_fitness':0, 'velocity':0}

    population = []
    
    for i in range(n_organisms):
        population.append(copy.deepcopy(n_queens_sol))
        population[i].randomize_function()
        
    return population


# In[74]:


attrs, int_by_attr = param_sol(5)
fit_function = lambda x: fitness_funct(x)
random_function = lambda x: random_funct(x)

n_queens_sol = Solution(fit_function, random_function, show_function, mutation_function, 
                        add_sols, subs_sols, mult_sols, div_sols, attrs, int_by_attr)

population = generate_population(100, 32)

population_ACS = generate_population(100, 16)


# ### PSO ALGORITHM

# In[120]:


def PSO_speed(weight_factor, particle, best_particle_pos, vmax, vmin, individual_learn, social_learn):
    
    # Lambdas
    speed_lamb = lambda x: int(weight_factor * x[1] + individual_learn * random.random() * (x[2]-x[0]) + 
                            social_learn * random.random() * (x[3]-x[0])) 
    
    dgt_lamb = lambda x: vmax if x >= vmax else vmin if x <= vmin else x
    
    # [xi, vi, Pi, Pg]
    particle_elements = [[particle.attribute_list[i], 
                          particle.particle_dict['velocity'][i], 
                          particle.particle_dict['pBest'][i], 
                          best_particle_pos[i]] for i in range(len(particle.attribute_list))]
    
    #return list(map(dgt_lamb, list(map(speed_lamb, particle_elements))))
    return list(map(speed_lamb, particle_elements))

def PSO_position(particle):
 
    # Lambdas
    post_lamb = lambda x: int(x[0] + x[1]) 
    
    particle_elements = [[particle.attribute_list[i], 
                          particle.particle_dict['velocity'][i]] for i in range(len(particle.attribute_list))]
    
    final_pos = []
    for element in list(map(post_lamb, particle_elements)):
        if element < 1:
            final_pos.append(1)
        elif element > len(particle.attribute_list):
            final_pos.append(len(particle.attribute_list))
        else:
            final_pos.append(element)
    
    return final_pos

def PSO_init(population, v_max, v_min):
        
    for particle in population:
        particle.particle_dict['pBest'] = copy.deepcopy(particle.attribute_list)
        particle.particle_dict['velocity'] = [random.uniform(v_min,v_max) for i in range(len(particle.attribute_list))]
        particle.particle_dict['pBest_fitness'] = 99999

@print_basic_info
def PSO_algorithm(population, max_iters, init=True, decreasing_rate=0.5, ind_rate=1, soc_rate=2, v_max=3, v_min=1):   
    """
    Pseudocode:
    Begin;
        Generate random population of N solutions
        (particles);
        For each individual i2N: calculate fitness (i);
        Initialize the value of the weight factor, u;
            For each particle;
                Set pBest as the best position of particle i;
                If fitness (i) is better than pBest;
                pBest(i)Zfitness (i);
        End;
        Set gBest as the best fitness of all particles;
        For each particle;
            Calculate particle velocity according to Eq. (3);
            Update particle position according to Eq. (4);
        End;
        Update the value of the weight factor, u;
    Check if terminationZtrue;
    End;
    """
    
    weight_fact = 0.5
    
    # Initialization
    if (init):
        PSO_init(population, v_max, v_min)
    
    printProgressBar(0, max_iters)
    for i in range(max_iters):
        # PSO itself
        
        if (i % 1000 == 0):
            weight_fact *= decreasing_rate
        
        for particle in population:
            pBest = particle.particle_dict['pBest']
            pBest_fit = particle.particle_dict['pBest_fitness']

            if particle.fitness_function() < pBest_fit:
                particle.particle_dict['pBest'] = copy.deepcopy(particle.attribute_list)
                particle.particle_dict['pBest_fitness'] = particle.fitness_function()
        
        fitness_vals = [particle.fitness_function() for particle in population]
        gBest = min(fitness_vals)
        
        best_part = population[fitness_vals.index(gBest)]

        for particle in population:
            particle.particle_dict['velocity'] = PSO_speed(weight_fact, particle, best_part.attribute_list, 
                                                           v_max, v_min, ind_rate, soc_rate)
            particle.attribute_list = copy.deepcopy(PSO_position(particle))
            
            
        printProgressBar(i, max_iters)


#%%

PSO_algorithm(population, 1, True, 0.5, 1, 2, 10, 0.01)
# PSO_algorithm(population, 10, False, 0.5, 1, 2, 10, 0.01)
# PSO_algorithm(population, 10, False, 0.5, 1, 2, 10, 0.01)
# PSO_algorithm(population, 2000, False, 0.7, 3, 5, 100, 0.001)
# PSO_algorithm(population, 100, False, 0.7, 3, 5, 2000, 20)




fitness_vals = [particle.fitness_function() for particle in population]
print(fitness_vals)

print(population[7].attribute_list)
print(fitness_vals.index(min(fitness_vals)))
print(min(fitness_vals))



#%%

def print_basic_info(func):
    
    def compute(*args, **kwargs):
        
        func(*args, **kwargs)
        
        print('#' * 100)
        print('Basic statitics from algorithm')
        print(f'\t\t>> Best fitness value: {min([part.fitness_function() for part in population])}')
        print(f'\t\t>> Mean fitness value: {mean([part.fitness_function() for part in population])}')
        print(f'\t\t>> Std-dev fitness value: {standard_desviation([part.fitness_function() for part in population])}\n')
        
        print('Fitness Values desglosed')
        print([part.fitness_function() for part in population])
        
        print('#' * 100)
        
    return compute


def compute_time(func):
    
    def inner(*args, **kwargs):
        print('begin time measurement...')
        print('*' * 30)
        start = datetime.now()
        
        func(*args, **kwargs)
        
        ended = datetime.now()
        print(f'This function took {ended-start} time')
        print('\n')
        print('*' * 30)
        
    return inner

def precision(nat_alg, total_iters=100):
    
    def compute(*args, **kwargs):
        print('init measurement...\n')
        
        last_fit = [particle.fitness_function() for particle in population]
        precision_computed = []
                
        for i in range(total_iters):
            nat_alg(*args, **kwargs)
            sols_summary = [True if last_fit[index] - particle.fitness_function() > 0 else False for index, particle in enumerate(population)]
            
            precision_computed.append(sols_summary.count(True)/len(sols_summary))
            
        print('*' * 100)
        print(f'\nPrecision of algorithm: {mean(precision_computed)}\nPrecisions obtained: {precision_computed}\n')
        
    return compute

def recall(nat_alg, total_iters=100):
    
    def compute(*args, **kwargs):
        print('init measurement...\n')
        
        last_fit = []
        precision_computed = []
        
        last_fit = [particle.fitness_function() for particle in population]
        
        for i in range(total_iters):
            nat_alg(*args, **kwargs)
            sols_summary = [True if last_fit[index] - particle.fitness_function() < last_fit[index] else False for index, particle in enumerate(population)]
            precision_computed.append(sols_summary.count(True)/len(sols_summary))
            
        print('*' * 100)
        print(f'\nPrecision of algorithm: {mean(precision_computed)}\nPrecisions obtained: {precision_computed}\n')
        
    return compute
        
        
precision_PSO = precision(PSO_algorithm, 20)(population, 100, False, 0.7, 3, 5, 1002, 0.1)

#%%
precision_SOS = precision(SOS_algorithm, 5)(10, 1, population)

#%%

# ### SOS ALGORITHM


mult_sol_float = lambda attr, value: map(lambda x: x*value, attr)
div_list_int   = lambda attr, value: map(lambda x: x/value, attr)

def _mutualism_phase(Xi_index, X_best, population):
    # Organisms selection
    population_copy = copy.deepcopy(population)
    Xi = population[Xi_index]
    
    Xj       = random.choice(population_copy[0:Xi_index]+population_copy[Xi_index+1:])
    Xj_index = population_copy.index(Xj)

    Xi_final    = Xi
    Xj_final    = Xj

    # Beneficial factor choose
    BF_list = [1,2]
    BF1 = random.choice(BF_list)
    BF_list.remove(BF1)
    BF2 = BF_list[0]

    # Compute new mutualist organisms
    mutual_vector = div_list_int(Xi + Xj, 2)
    
    Xi_new = copy.deepcopy(Xi)
    Xj_new = copy.deepcopy(Xj)

    profit_i = mult_sol_float(list(map(lambda x: x[0]*x[1], 
                                       list(zip(X_best.attribute_list, mutual_vector)))), 
                                       random.uniform(0,1))
                                   
    profit_j = mult_sol_float(list(map(lambda x: x[0]*x[1], 
                                       list(zip(X_best.attribute_list, mutual_vector)))), 
                                       random.uniform(0,1))
    
    Xi_new.atrribute_list = list(map(lambda x: x[0]+x[1], zip(Xi_new.attribute_list, profit_i)))
    Xj_new.attribute_list = list(map(lambda x: x[0]+x[1], zip(Xj_new.attribute_list, profit_j)))
    
    if (Xi.fitness_function() < Xi_new.fitness_function()):
        Xi_final = copy.deepcopy(Xi_new)
        
    if (Xj.fitness_function() < Xj_new.fitness_function()):
        Xj_final = copy.deepcopy(Xj_new)

    population[Xi_index] = copy.deepcopy(Xi_final)         
    population[Xj_index] = copy.deepcopy(Xj_final)         

def _commensalism_phase(Xi_index, X_best, population):
    population_copy = copy.deepcopy(population)
    Xi = population[Xi_index]
    
    Xj          = random.choice(population_copy[0:Xi_index]+population_copy[Xi_index+1:])

    Xi_final    = Xi
    
    Xi_new = copy.deepcopy(Xi)
    profit_i = mult_sol_float(X_best - Xj, random.uniform(-1,1))
    
    Xi_new.attribute_list = list(map(lambda x: x[0]+x[1], list(zip(Xi.attribute_list, profit_i))))
        
    if (Xi.fitness_function() < Xi_new.fitness_function()):
        Xi_final = Xi_new

    population[Xi_index] = copy.deepcopy(Xi_final) 

def _parasitism_phase(Xi_index, population):
    population_copy = copy.deepcopy(population)
    Xi = population[Xi_index]
    
    Xj          = random.choice(population_copy[0:Xi_index]+population_copy[Xi_index+1:])
    
    Xj_final    = Xj

    Xi.mutation_function()

    if (Xi.fitness_function() > Xj.fitness_function()):
        Xj_final = Xi

    population[population_copy.index(Xj)] = copy.deepcopy(Xj_final)         

@print_basic_info
def SOS_algorithm(iterations:int, termination_criteria, population):
    n_iteration = 0       
    
    printProgressBar(0, iterations)
    while(n_iteration < iterations):

        for i in range(len(population)):
            fitness_vals    = [organism.fitness_function() for organism in population]
            gBest           = min(fitness_vals)
        
            Xi_index = i % len(population)
            X_best      = population[fitness_vals.index(gBest)]
             
            _mutualism_phase(Xi_index, X_best, population)

            _commensalism_phase(Xi_index, X_best, population)

            _parasitism_phase(Xi_index, population)
        
        printProgressBar(n_iteration, iterations)


        n_iteration += 1


# In[84]:


get_ipython().run_cell_magic('time', '', 'SOS_algorithm(20, lambda x:1, population)')


# In[85]:


fitness_vals = [particle.fitness_function() for particle in population]
print(fitness_vals)

print(population[8].attribute_list)
print(population[9].attribute_list)

print(fitness_vals.index(min(fitness_vals)))
print(min(fitness_vals))


# ### ACS ALGORITHM

# In[425]:


def pseudo_random_proportional_rule(solution, pheromones, alpha, beta, nodes_not_discovered):
    last_parameter_assign = len(solution.attribute_list)-1
    posible_nodes = []

    sol_aux = copy.deepcopy(solution)
    sol_aux.attribute_list.append(-1)
        
    for i in nodes_not_discovered:
        sol_aux.attribute_list[-1] = i        
              
        posible_nodes.append((math.pow(pheromones[solution.attribute_list[last_parameter_assign]][i], alpha) *
                              math.pow(sol_aux.fitness_function(), beta)))
        
        
    denom = functools.reduce(lambda a, b: a+b, posible_nodes)
    
    if denom != 0:
        posible_nodes = list(map(lambda x: x/denom, posible_nodes))
    
    return posible_nodes

def best_node_selection(solution, pheromones, alpha, beta, nodes_not_discovered):
    last_parameter_assign = len(solution.attribute_list)-1
    posible_nodes = []

    sol_aux = copy.deepcopy(solution)
    sol_aux.attribute_list.append(-1)
    
    for i in nodes_not_discovered:
        sol_aux.attribute_list[-1] = i        
              
        posible_nodes.append((math.pow(pheromones[solution.attribute_list[last_parameter_assign]][i], alpha) *
                              math.pow(sol_aux.fitness_function(), beta)))
        
    return posible_nodes

def ACS_init_pheromones(pheromones, sol_dims):
    if pheromones == []:
        pheromones = np.full((sol_dims,sol_dims), init_pheromone)
        pheromones[np.eye(sol_dims) == 1] = 0

    return pheromones

def add_step_to_solution(population):
    for solution_index, solution in enumerate(population):
        solution.attribute_list.append(0) 

def check_possible_next_step(solution):
    sol_dims = len(solution.interval_by_attr)

    nodes_not_discovered = [i for i in range(sol_dims-1)]
    nodes_not_discovered = nodes_not_discovered[:solution.attribute_list[0]] + nodes_not_discovered[solution.attribute_list[0]+1:]

    return nodes_not_discovered

def ACS_algorithm(total_iters, termination_criteria, population, q0, 
                  phi, init_pheromone, persistence, alpha, beta, pheromones=[]):
    
    n_iter = 0
    
    sol_dims = len(population[0].interval_by_attr)

    pheromones = ACS_init_pheromones(pheromones, sol_dims)
    updating_pheromones = lambda phero: (1-persistence)*phero

    #while termination_criteria(population):
    printProgressBar(0, total_iters)
    while n_iter < total_iters:
        
        add_step_to_solution(population)           
                    
        for ant in population:
            nodes_not_discovered = check_possible_next_step(ant)
            
            while nodes_not_discovered != []:
                # Compute pseudo-random proporcional rule                
                if random.uniform(0,1) <= q0:
                    nodes_viable = best_node_selection(ant, pheromones, alpha, beta,
                                                       nodes_not_discovered)
                    
                    node_sel = min(nodes_viable)
                    new_state  = nodes_viable.index(node_sel)

                    ant.attribute_list.append(nodes_not_discovered[new_state])
                    nodes_not_discovered = nodes_not_discovered[:new_state] + nodes_not_discovered[new_state+1:]
                    
                else:
                    nodes_viable = pseudo_random_proportional_rule(ant, pheromones, alpha, beta,
                                                                   nodes_not_discovered)
                    
                    node_sel = choice(nodes_not_discovered, 1, nodes_viable)
                    new_state  = node_sel[0]
                    
                    ant.attribute_list.append(new_state)
                
                    nodes_not_discovered.remove(new_state)
                    
                last_state = ant.attribute_list[-2]
     
                pheromones[last_state][new_state] = (((1-phi) * pheromones[last_state][new_state]) + 
                                                                            (phi * init_pheromone))

        fitness_vals = [ant.fitness_function() for ant in population]
        gBest = min(fitness_vals)
        
        best_ant = population[fitness_vals.index(gBest)]
        
        pheromones_updated = pheromones.copy()
        pheromones_updated = np.array([updating_pheromones(xi) for xi in pheromones_updated])
        
        best_ant_path = zip(best_ant.attribute_list[:len(best_ant.attribute_list)-1], best_ant.attribute_list[1:])
        
        for tuple_nodes in best_ant_path:
            pheromones_updated[tuple_nodes[0]][tuple_nodes[1]] = ((1 - persistence) * (pheromones_updated[tuple_nodes[0]][tuple_nodes[1]]) +
                                                                 (((1/gBest) * persistence)))
        if n_iter != total_iters-1:    
            for ant in population:
                ant.attribute_list = []
        
        pheromones = pheromones_updated.copy()
        
        n_iter += 1
        printProgressBar(n_iter, total_iters)    
        
    return pheromones


# In[444]:


get_ipython().run_cell_magic('time', '', "\nfor solution in population_ACS:\n    solution.attribute_list = []\n\n#phero = ACS_algorithm(10, '', population_ACS[0:28], 0.8, 0.2, 30, 1, 0.4, 1.5, 0.3)\nphero = ACS_algorithm(100, '', population_ACS, 1, 1, 30, 1, 0.5, 0.4, 1)")


# In[445]:


phero 


# In[446]:


fitness_vals = [particle.fitness_function() for particle in population_ACS[0:28]]
print(fitness_vals)

print(population_ACS[17].attribute_list)
print(fitness_vals.index(min(fitness_vals)))
print(min(fitness_vals))


# ### Fish swarm algorithm

# In[ ]:



def _foraging_behaviour(art_fish):
    actual_fit = art_fish.fitnesss_function()
    
    
    

def AFSA_algorithm(population, visual, step):
    
    pass


# ### Cuckoo Search Algorithm

# In[ ]:


"""
Cuckoo Search Algorithm 
begin
    Objective function  f(x), x=( x1,..., xd)T;
    Initial a population of n host nests xi (i=1,2,...,n);
        while (t < Maximum Generation) or (stop criterion);
            Get a cuckoo (say i) randomly     
            and generate a new solution by Lévy flights;           
            
            Evaluate its quality/fitness; Fi
            Choose a nest among n (say j ) randomly;
            
            if (Fi > Fj),                   
                Replace j by the new solution; 
            end           
            
            Abandon a fraction (Pa) of worse nests 
            [and build new ones at new locations via Lévy flights]; 
            
            Keep the best solutions (or nests with quality solutions);
            
            Rank the solutions and find the current best; 
            
        end while
        Post process results and visualization; 
    end 
"""

def Levy_flight(t, t0, alpha):
    
    CDF = 0
    PDF = 0
    
    if t >= t0:
        CDF = 1 - pow((t/t0), (alpha * -1))
        PDF = alpha * (pow(t0, alpha)/pow(t, 1+alpha))
    
    return CDF, PDF

def CSA_algorithm(init_population, max_iters, stop_criteria):
    
    pop_array = []
    
    for iteration in range(max_iters):
        
        pass

# In[2]:

def WOA_init(population):
    
    a = np.full(shape=[1, len(population[0].attribute_list)])

    return a    
    

def WOA_encircle_search(actual_whale, whale_to_update, parameter_A, parameter_C):
"""
A ver en general encircle prey y search son literalmente el mismo metodo, por eso lo
hago asi todo junto. Que esta muy bien lo de los algoritmos inspirados en naturaleza
pero esta mejor no tener codigo clon a punta pala.
"""
    parameter_D = np.linalg.norm(np.multiply(parameter_C, whale_to_update.attribute_list) - actual_whale.attribute_list)
    return whale_to_update.attribute_list - np.multiply(parameter_A, parameter_D)

def WOA_attack(whale, best_whale, constant_b, parameter_l):
    parameter_D = np.linalg.norm(best_whale.attribute_list - whale.attribute_list)
    return np.multiply(np.multiply(parameter_D, np.exp(constant_b*parameter_l)), np.cos(2.0*np.pi*parameter_l)) + best_whale.attribute_list


def WOA_algorithm(total_iters, population, a, a_step, b, A, C):

    n_iter = 0
    
    fitness_results = [whale.fitness_function() for whale in population]
    best_whale = population[fitness_results.index(min(fitness_results))]

    a = WAO_init(population)
    
    printProgressBar(0, total_iters)

    while n_iter < total_iters:
        for whale_index, whale in enumerate(population):
            a -= a_step
            A = 2 * a - np.random.rand(*a.shape) - a
            C = 2 * np.random.rand(*a.shape)
            
            l = np.random.rand(*a.shape)
            p = random.uniform(0, 1)
            
            if p < 0.5:
                #bubble net hunting
                if np.linalg.norm(A) < 1:
                    new_attributes = WOA_encircle_search(whale, best_whale, A, C)

                else:
                    random_whale = random.choice(population[0:whale_index]+population[whale_index+1:])
                    new_attributes = WOA_encircle_search(whale, random_whale, A, C)
            else:
                new_attributes = WOA_attack(actual_whale, best_whale, b, l)
            
            actual_whale.attribute_list = new_attributes
                
        fitness_results = [whale.fitness_function() for whale in population]
        best_whale = population[fitness_results.index(min(fitness_results))]

        total_iters += 1
                

#%%

def MSA_init(population):

    for monkey in population:



# In[3]:


size = 50000
x, y = mist.get_levy_flight(size, mode='2D')


# In[4]:


print(x)
print(y)


# ## Niapas de ingeniero

# In[48]:





# In[47]:



def param_sol(size):
    # attrs = list(range(0, size))
    attrs = [n for n in range(1, size)]
    int_by_attr = [(1, size) for number in range(1, size)]
    
    return attrs, int_by_attr


def fitness_funct(self):
    sum_lamb    = lambda x: abs(x[0]-x[1])
    
    attack = 0

    for queen_col, queen_row in enumerate(self.attribute_list):
        queen_pos = [queen_col+1, queen_row]
        print(queen_pos)
        
        # conf_attacks = lambda x: (queen_pos[1] == x)
        
        conf_attacks_dia = lambda x: (x == sum_lamb(queen_pos))
        
        if queen_pos[1] == 1:
            print(queen_pos)
            
            if queen_pos[0] == 1:
                conf_attacks_dia = lambda x:((x % (queen_pos[1]+1 + queen_pos[0]+1) == 0) or 
                                     (queen_pos[1]+1 + queen_pos[0]+1) % x == 0)
            
            elif queen_pos[0] == len(self.attribute_list):
                conf_attacks_dia = lambda x: ((x % (queen_pos[1]+1 + queen_pos[0]-1) == 0) or 
                                     (queen_pos[1]+1 + queen_pos[0]-1) % x == 0) 

            else:
                conf_attacks_dia = lambda x: ((x % (queen_pos[1]+1 + queen_pos[0]+1) == 0) or 
                                     (x % (queen_pos[1]+1 + queen_pos[0]-1) == 0) or 
                                     ((queen_pos[1]+1 + queen_pos[0]+1) % x == 0) or 
                                     ((queen_pos[1]+1 + queen_pos[0]-1) % x == 0))
            
        elif queen_pos[1] == len(self.attribute_list):
            
            if queen_pos[0] == 1:
                conf_attacks_dia = lambda x: ((x % (queen_pos[1]-1 + queen_pos[0]+1) == 0) or
                                         ((queen_pos[1]-1 + queen_pos[0]+1) % x == 0))
            
            elif queen_pos[0] == len(self.attribute_list):
                conf_attacks_dia = lambda x: ((x % (queen_pos[1]-1 + queen_pos[0]-1) == 0) or 
                                     ((queen_pos[1]-1 + queen_pos[0]-1) % x == 0))

            else:
                conf_attacks_dia = lambda x: ((x % (queen_row-1 + queen_pos[0]-1) == 0) or 
                                     (x % (queen_pos[1]-1 + queen_pos[0]+1) == 0) or
                                     ((queen_pos[1]-1 + queen_pos[0]-1) % x == 0) or
                                     ((queen_pos[1]-1 + queen_pos[0]+1) % x == 0))
            
        else:

            if queen_pos[0] == 1:
                conf_attacks_dia = lambda x: ((x % (queen_pos[1]+1 + queen_pos[0]+1) == 0) or 
                                     (x % (queen_pos[1]-1 + queen_pos[0]+1) == 0) or
                                     ((queen_pos[1]+1 + queen_pos[0]+1) % x == 0) or 
                                     ((queen_pos[1]-1 + queen_pos[0]+1) % x == 0))
            
            elif queen_pos[0] == len(self.attribute_list):
                conf_attacks_dia = lambda x: ((x % (queen_pos[1]-1 + queen_pos[0]-1) == 0) or 
                                     (x % (queen_pos[1]+1 + queen_pos[0]-1) == 0) or 
                                     ((queen_pos[1]-1 + queen_pos[0]-1) % x == 0) or
                                     ((queen_pos[1]+1 + queen_pos[0]-1) % x == 0))

            else:
                conf_attacks_dia = lambda x: ((x % (queen_pos[1]-1 + queen_pos[0]-1) == 0) or 
                                     (x % (queen_pos[1]+1 + queen_pos[0]+1) == 0) or 
                                     (x % (queen_pos[1]+1 + queen_pos[0]-1) == 0) or 
                                     (x % (queen_pos[1]-1 + queen_pos[0]+1) == 0) or
                                     ((queen_pos[1]-1 + queen_pos[0]-1) % x == 0) or
                                     ((queen_pos[1]+1 + queen_pos[0]+1) % x == 0) or 
                                     ((queen_pos[1]+1 + queen_pos[0]-1) % x == 0) or 
                                     ((queen_pos[1]-1 + queen_pos[0]+1) % x == 0))
        

                
        attacks_1 = list(map(conf_attacks, self.attribute_list[:queen_col] + self.attribute_list[queen_col+1 :]))
        
        aux = [[index+1, element] for index, element in enumerate(self.attribute_list)]
        
        
        print(aux[:queen_col] + aux[queen_col+1 :])
        print(list(map(sum_lamb, aux[:queen_col] + aux[queen_col+1 :])))
        
        attacks_2 = list(map(conf_attacks_dia, list(map(sum_lamb, aux[:queen_col] + aux[queen_col+1 :]))))
        print(attacks_2)
        
        attack += (attacks_1.count(True) + attacks_2.count(True))
        
        
    return attack / len(self.attribute_list)

attrs, int_by_attr = param_sol(4)

print(attrs)

n_queens_sol = Organism(attrs, int_by_attr)
n_queens_sol._fitness_function = lambda x: fitness_funct(x)

n_queens_sol.fitness_function()

