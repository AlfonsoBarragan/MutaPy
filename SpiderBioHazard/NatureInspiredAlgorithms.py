import copy
import numpy as np
import random


from Galfgets import printProgressBar
from Solution import Solution

# Methods to modify solutions in algorithm execution
def _clean_population_attr(population:list) -> void:
    for solution in population:
        solution.attribute_list = []

def _add_step_to_population(population:list) -> void:
    for solution_index, solution in enumerate(population):
        solution.attribute_list.append(0) 

def _check_possible_next_step(solution:Solution) -> list:
    sol_dims = len(solution.interval_by_attr)

    nodes_not_discovered = [i for i in range(sol_dims-1)]
    nodes_not_discovered = nodes_not_discovered[:solution.attribute_list[0]] + nodes_not_discovered[solution.attribute_list[0]+1:]

    return nodes_not_discovered

# Particle Swarm Optimization Algorithm 
## Paper's title: A new optimizer using particle swarm theory
## DOI: 10.1109/MHS.1995.494215 
## Authors: R. Eberhart, J. Kennedy 
## Year: 1995 

def _PSO_speed(weight_factor:float, particle:Solution, best_particle_pos:list, vmax:float, vmin:float, individual_learn:float, social_learn:float) -> list:
    
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

def _PSO_position(particle:Solution) -> list:
 
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

def _PSO_init(population:list, v_max:float, v_min:float) -> None:
        
    for particle in population:
        particle.particle_dict['pBest'] = copy.deepcopy(particle.attribute_list)
        particle.particle_dict['velocity'] = [random.uniform(v_min,v_max) for i in range(len(particle.attribute_list))]
        particle.particle_dict['pBest_fitness'] = 99999
    
def PSO_algorithm(population:list, max_iters:int, init:bool=True, decreasing_rate:float=0.5, ind_rate:float=1, soc_rate:float=2, v_max:float=3, v_min:float=1) -> None:   
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
        _PSO_init(population, v_max, v_min)
    
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
            particle.particle_dict['velocity'] = _PSO_speed(weight_fact, particle, best_part.attribute_list, 
                                                           v_max, v_min, ind_rate, soc_rate)
            particle.attribute_list = copy.deepcopy(_PSO_position(particle))
            
            
        printProgressBar(i, max_iters)

# Symbiotic Organisms Search Algorithm
## Paper's title: Symbiotic Organisms Search: A new metaheuristic optimization algorithm
## DOI: https://doi.org/10.1016/j.compstruc.2014.03.007
## Authors: Min-Yuan Cheng, Doddy Prayogo
## Year: 2014
mult_sol_float = lambda attr, value: map(lambda x: x*value, attr)
div_list_int   = lambda attr, value: map(lambda x: x/value, attr)

def _mutualism_phase(Xi_index:int, X_best:int, population:list) -> None:
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

def _commensalism_phase(Xi_index:int, X_best:int, population:list) -> None:
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

def _parasitism_phase(Xi_index:int, population:list) -> None:
    population_copy = copy.deepcopy(population)
    Xi = population[Xi_index]
    
    Xj          = random.choice(population_copy[0:Xi_index]+population_copy[Xi_index+1:])
    
    Xj_final    = Xj

    Xi.mutation_function()

    if (Xi.fitness_function() > Xj.fitness_function()):
        Xj_final = Xi

    population[population_copy.index(Xj)] = copy.deepcopy(Xj_final)         

def SOS_algorithm(iterations:int, termination_criteria:callable, population:list) -> None:
    n_iteration = 0       

    while(n_iteration < iterations):
    
        for i in range(len(population)):
            fitness_vals    = [organism.fitness_function() for organism in population]
            gBest           = min(fitness_vals)
        
            Xi_index = i % len(population)
            X_best      = population[fitness_vals.index(gBest)]
             
            _mutualism_phase(Xi_index, X_best, population)

            _commensalism_phase(Xi_index, X_best, population)

            _parasitism_phase(Xi_index, population)

        if termination_criteria(population):
            break

        n_iteration += 1

# Symbiotic Organisms Search Algorithm
## Paper's title: Symbiotic Organisms Search: A new metaheuristic optimization algorithm
## DOI: https://doi.org/10.1016/j.compstruc.2014.03.007
## Authors: Min-Yuan Cheng, Doddy Prayogo
## Year: 2014
def _pseudo_random_proportional_rule(solution:Solution, pheromones:np.array, alpha:float, 
                                    beta:float, nodes_not_discovered:list) -> list:

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

def _best_node_selection(solution:Solution, pheromones:np.array, alpha:float, 
                        beta:float, nodes_not_discovered:list) -> list:
                        
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

def ACS_algorithm(total_iters:int, termination_criteria:callable, population:list, q0:float, 
                  phi:float, init_pheromone:float, persistence:float, alpha:float, 
                  beta:float, pheromones:list=[]) -> np.array:
    
    n_iter = 0
    
    sol_dims = len(population[0].interval_by_attr)

    pheromones = ACS_init_pheromones(pheromones, sol_dims)

    
    updating_pheromones = lambda phero: (1-persistence)*phero

    #while termination_criteria(population):
    printProgressBar(0, total_iters)
    while n_iter < total_iters:
        
        _add_step_to_population(population)           
                    
        for ant in population:
            nodes_not_discovered = check_possible_next_step(ant)

            while nodes_not_discovered != []:
                # Compute pseudo-random proporcional rule                
                if random.uniform(0,1) <= q0:
                    nodes_viable = _best_node_selection(ant, pheromones, alpha, beta,
                                                       nodes_not_discovered)
                    
                    node_sel = min(nodes_viable)
                    new_state  = nodes_viable.index(node_sel)

                    ant.attribute_list.append(nodes_not_discovered[new_state])
                    nodes_not_discovered = nodes_not_discovered[:new_state] + nodes_not_discovered[new_state+1:]
                    
                else:
                    nodes_viable = _pseudo_random_proportional_rule(ant, pheromones, alpha, beta,
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

# Fish swarm algorithm 
## Paper's title: Symbiotic Organisms Search: A new metaheuristic optimization algorithm
## DOI: https://doi.org/10.1016/j.compstruc.2014.03.007
## Authors: Min-Yuan Cheng, Doddy Prayogo
## Year: 2014


# Whale optimization algorithm 
## Paper's title: The  Whale  Optimization  Algorithm
## DOI: https://doi.org/10.1016/j.advengsoft.2016.01.008
## Authors: Seyedali Mirjaliliab, Andrew Lewisa
## Year: 2016

def _WOA_init(population):
    a = np.full(shape=[1, len(population[0].attribute_list)])

    return a    
    

def _WOA_encircle_search(actual_whale, whale_to_update, parameter_A, parameter_C):
"""
A ver en general encircle prey y search son literalmente el mismo metodo, por eso lo
hago asi todo junto. Que esta muy bien lo de los algoritmos inspirados en naturaleza
pero esta mejor no tener codigo clon a punta pala.
"""
    parameter_D = np.linalg.norm(np.multiply(parameter_C, whale_to_update.attribute_list) - actual_whale.attribute_list)
    return whale_to_update.attribute_list - np.multiply(parameter_A, parameter_D)

def _WOA_attack(whale, best_whale, constant_b, parameter_l):
    parameter_D = np.linalg.norm(best_whale.attribute_list - whale.attribute_list)
    return np.multiply(np.multiply(parameter_D, np.exp(constant_b*parameter_l)), np.cos(2.0*np.pi*parameter_l)) + best_whale.attribute_list


def WOA_algorithm(total_iters, population, a, a_step, b, A, C):

    n_iter = 0
    
    fitness_results = [whale.fitness_function() for whale in population]
    best_whale = population[fitness_results.index(min(fitness_results))]

    a = _WAO_init(population)
    
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
                    new_attributes = _WOA_encircle_search(whale, best_whale, A, C)

                else:
                    random_whale = random.choice(population[0:whale_index]+population[whale_index+1:])
                    new_attributes = _WOA_encircle_search(whale, random_whale, A, C)
            else:
                new_attributes = _WOA_attack(actual_whale, best_whale, b, l)
            
            actual_whale.attribute_list = new_attributes
                
        fitness_results = [whale.fitness_function() for whale in population]
        best_whale = population[fitness_results.index(min(fitness_results))]

        total_iters += 1
                
