import copy
import math
import numpy as np
import random
import functools


from Galfgets import printProgressBar
from Solution import Solution

# Methods to modify solutions in algorithm execution
def _clean_population_attr(population:list) -> None:
    for solution in population:
        solution.attribute_list = []

def _add_step_to_population(population:list) -> None:
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

def _PSO_init(population:list, v_max:float, v_min:float, problem_kind:str='min') -> None:

    if problem_kind == 'min':
        best_fitness_default = 99999
    else:
        best_fitness_default = -99999

    for particle in population:
        particle.particle_dict['pBest'] = copy.deepcopy(particle.attribute_list)
        particle.particle_dict['velocity'] = [random.uniform(v_min,v_max) for i in range(len(particle.attribute_list))]
        particle.particle_dict['pBest_fitness'] = best_fitness_default
    
def PSO_algorithm(population:list, max_iters:int, init:bool=True, decreasing_rate:float=0.5, ind_rate:float=1, soc_rate:float=2, v_max:float=3, v_min:float=1, problem_kind:str='min') -> None:   
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
        _PSO_init(population, v_max, v_min, problem_kind)
    
    printProgressBar(0, max_iters)
    for i in range(max_iters):
        # PSO itself
        
        if (i % 1000 == 0):
            weight_fact *= decreasing_rate
        
        for particle in population:
            pBest = particle.particle_dict['pBest']
            pBest_fit = particle.particle_dict['pBest_fitness']

            if problem_kind == 'min':
                if particle.fitness_function() < pBest_fit:
                    particle.particle_dict['pBest'] = copy.deepcopy(particle.attribute_list)
                    particle.particle_dict['pBest_fitness'] = particle.fitness_function()
            else:
                if particle.fitness_function() > pBest_fit:
                    particle.particle_dict['pBest'] = copy.deepcopy(particle.attribute_list)
                    particle.particle_dict['pBest_fitness'] = particle.fitness_function()

        fitness_vals = [particle.fitness_function() for particle in population]
        
        if problem_kind == 'min':
            gBest = min(fitness_vals)
        else:
            gBest = max(fitness_vals)

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

def _mutualism_phase(Xi_index:int, X_best:int, population:list, problem_kind:str='min') -> None:
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
    
    if problem_kind == 'min':
        if (Xi.fitness_function() < Xi_new.fitness_function()):
            Xi_final = copy.deepcopy(Xi_new)
            
        if (Xj.fitness_function() < Xj_new.fitness_function()):
            Xj_final = copy.deepcopy(Xj_new)
    else:
        if (Xi.fitness_function() > Xi_new.fitness_function()):
            Xi_final = copy.deepcopy(Xi_new)
            
        if (Xj.fitness_function() > Xj_new.fitness_function()):
            Xj_final = copy.deepcopy(Xj_new)

    population[Xi_index] = copy.deepcopy(Xi_final)         
    population[Xj_index] = copy.deepcopy(Xj_final)         

def _commensalism_phase(Xi_index:int, X_best:int, population:list, problem_kind:str='min') -> None:
    population_copy = copy.deepcopy(population)
    Xi = population[Xi_index]
    
    Xj          = random.choice(population_copy[0:Xi_index]+population_copy[Xi_index+1:])

    Xi_final    = Xi
    
    Xi_new = copy.deepcopy(Xi)
    profit_i = mult_sol_float(X_best - Xj, random.uniform(-1,1))
    
    Xi_new.attribute_list = list(map(lambda x: x[0]+x[1], list(zip(Xi.attribute_list, profit_i))))
    
    if problem_kind == 'min':
        if (Xi.fitness_function() < Xi_new.fitness_function()):
            Xi_final = Xi_new
    else:
        if (Xi.fitness_function() > Xi_new.fitness_function()):
        Xi_final = Xi_new

    population[Xi_index] = copy.deepcopy(Xi_final) 

def _parasitism_phase(Xi_index:int, population:list, problem_kind:str='min') -> None:
    population_copy = copy.deepcopy(population)
    Xi = population[Xi_index]
    
    Xj          = random.choice(population_copy[0:Xi_index]+population_copy[Xi_index+1:])
    
    Xj_final    = Xj

    Xi.mutation_function()
    if problem_kind == 'min':            
        if (Xi.fitness_function() > Xj.fitness_function()):
            Xj_final = Xi
    else:            
        if (Xi.fitness_function() < Xj.fitness_function()):
            Xj_final = Xi

    population[population_copy.index(Xj)] = copy.deepcopy(Xj_final)         

def SOS_algorithm(iterations:int, termination_criteria:callable, population:list, problem_kind:str='min') -> None:
    n_iteration = 0       

    while(n_iteration < iterations):
    
        for i in range(len(population)):
            fitness_vals    = [organism.fitness_function() for organism in population]
            gBest           = min(fitness_vals)
        
            Xi_index = i % len(population)
            X_best      = population[fitness_vals.index(gBest)]
             
            _mutualism_phase(Xi_index, X_best, population, problem_kind)

            _commensalism_phase(Xi_index, X_best, population, problem_kind)

            _parasitism_phase(Xi_index, population, problem_kind)

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

def ACS_init_pheromones(sol_dims, init_pheromone):
    pheromones = np.full((sol_dims,sol_dims), init_pheromone)
    pheromones[np.eye(sol_dims) == 1] = 0
    
    return pheromones

def ACS_algorithm(total_iters, termination_criteria, population, q0, 
                  phi, init_pheromone, persistence, alpha, beta, pheromones):
    
    n_iter = 0
    
    sol_dims = len(population[0].interval_by_attr)
    
    if len(pheromones) == 0:
        pheromones = ACS_init_pheromones(sol_dims, init_pheromone)
    
    evaporate_pheromones = lambda phero: (1-persistence)*phero
    
    _clean_population_attr(population)    

    #while termination_criteria(population):
    printProgressBar(0, total_iters)
    while n_iter < total_iters:
        
        _add_step_to_population(population)           
                    
        for ant in population:
            nodes_not_discovered = _check_possible_next_step(ant)
                        
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
                    
                    node_sel = random.choice(nodes_not_discovered, 1, nodes_viable)
                    new_state  = node_sel[0]
                    
                    ant.attribute_list.append(new_state)
                
                    nodes_not_discovered.remove(new_state)
                    
                last_state = ant.attribute_list[-2]
     
                pheromones[last_state][new_state] = (((1-phi) * pheromones[last_state][new_state]) + 
                                                                            (phi * init_pheromone))

        fitness_vals = [ant.fitness_function() for ant in population]
        gBest = min(fitness_vals)
        
        best_ant = population[fitness_vals.index(gBest)]
        
        pheromones_updated = copy.deepcopy(pheromones)
        pheromones_updated = np.multiply(pheromones_updated, np.full((sol_dims,sol_dims), 1 - persistence))
        
        best_ant_path = zip(best_ant.attribute_list[:len(best_ant.attribute_list)-1], best_ant.attribute_list[1:])
        
        for tuple_nodes in best_ant_path:
            pheromones_updated[tuple_nodes[0]][tuple_nodes[1]] = ((1 - persistence) * (pheromones_updated[tuple_nodes[0]][tuple_nodes[1]]) +
                                                                 (((1/gBest) * persistence)))
        if n_iter != total_iters-1:    
            _clean_population_attr(population)
        
        pheromones = copy.deepcopy(pheromones_updated)
        
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
def _WOA_init(population:list, parameter_a:float):
    a = np.full(shape=[1, len(population[0].attribute_list)], fill_value=parameter_a)
    return a    
    

def _WOA_encircle_search(actual_whale:Solution, whale_to_update:Solution, parameter_A:float, parameter_C:float):
    parameter_D = np.absolute(np.subtract(np.multiply(parameter_C, whale_to_update.attribute_list), actual_whale.attribute_list))
    return np.subtract(whale_to_update.attribute_list, np.multiply(parameter_A, parameter_D))

def _WOA_attack(whale:Solution, best_whale:Solution, constant_b:float, parameter_l:float):
    parameter_D = np.absolute(np.subtract(best_whale.attribute_list, whale.attribute_list))
    np1 = np.multiply(np.multiply(parameter_D, np.exp(constant_b*parameter_l)), np.cos(2.0*np.pi*parameter_l))
    
    return np.add(np1, best_whale.attribute_list)

def _WOA_compute_A(parameter_a:float):
    return np.subtract(np.subtract(np.multiply(parameter_a, 2), np.random.rand(*parameter_a.shape)), parameter_a)

def _WOA_compute_C(parameter_a:float):
    return np.multiply(np.random.rand(*parameter_a.shape) ,2)

def _WOA_amend_whale(whale:Solution):
    for attr_index, attribute in enumerate(whale.attribute_list):

        if (attribute < whale.interval_by_attr[attr_index][0]) or (attribute > whale.interval_by_attr[attr_index][1]):
            whale.attribute_list[attr_index] = attribute % whale.interval_by_attr[attr_index][1]

def WOA_algorithm(total_iters:int, termination_criteria:callable, population:list, a_value:float, a_step:float, b:float):

    n_iter = 0
    
    fitness_results = [whale.fitness_function() for whale in population]
    best_whale = population[fitness_results.index(min(fitness_results))]

    a = _WOA_init(population, a_value)
    a_value_cpy = a_value
    
    printProgressBar(0, total_iters)

    while n_iter < total_iters:
        for whale_index, whale in enumerate(population):
            
            if a_value_cpy - a_step >= 0:
                a -= np.full(shape=a.shape, fill_value=a_step)
                a_value_cpy -= a_step
                
            else:
                a = np.full(shape=a.shape, fill_value=0)

            A = _WOA_compute_A(a)
            C = _WOA_compute_C(a)
            
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
                new_attributes = _WOA_attack(whale, best_whale, b, l)
            
            whale.attribute_list = [int(x) for x in new_attributes.tolist()[0]]
            _WOA_amend_whale(whale)
            
        fitness_results = [whale.fitness_function() for whale in population]
        best_whale = population[fitness_results.index(min(fitness_results))]

        if termination_criteria(population):
            break

        n_iter += 1
        
        printProgressBar(n_iter, total_iters)

# Multi-verse optimization algorithm
## Multi-Verse Optimizer: a nature-inspired algorithm for global optimization
## DOI: https://doi.org/10.1007/s00521-015-1870-7
## Authors: Seyedali Mirjaliliab, Seyed Mohammad Mirjalili, Abdolreza Hatamlou
## Year: 2016
def _MVO_compute_WEP(l: float, L: float, minimum: float, maximum: float):
    return minimum + l * ((maximum-minimum)/L)

def _MVO_compute_TDR(l: float, L: float, p: float):
    return 1 - ((pow(l, 1/p))/(pow(L, 1/p)))

def _MVO_sort_universes(population, problem_kind:str='min'):
    inflation_rate_list = [universe.fitness_function()
                           for universe in population]
    inflation_rate_list_sorted = copy.deepcopy(inflation_rate_list)

    if (problem_kind == 'min'):
        inflation_rate_list_sorted.sort()

    else:
        inflation_rate_list_sorted.sort(reverse=True)

    sorted_universes = [inflation_rate_list.index(
        inflation_rate_value) for inflation_rate_value in inflation_rate_list_sorted]

    return sorted_universes

def _MVO_normalize_infl_rate(population):
    inflation_rate_list = [universe.fitness_function()
                           for universe in population]

    infl_rate_max = max(inflation_rate_list)
    infl_rate_min = min(inflation_rate_list)

    normalized_infl_rate_list = [(infl_rate - infl_rate_min)/(
        infl_rate_max-infl_rate_min) for infl_rate in inflation_rate_list]

    return np.array(normalized_infl_rate_list)

def _roulette_wheel_selection(weights):
    accumulation = np.cumsum(weights)
    p = random.random() * accumulation[-1]

    chosen_index = -1

    for index in range(0, len(accumulation)):

        if accumulation[index] > p:
            chosen_index = index
            break

    choice = chosen_index
    return choice

def MVO_algorithm(total_iters:int, termination_criteria:callable, population:list, minimum:float, maximum:float, p:float, problem_kind:str='min'):
    n_iter = 0
    printProgressBar(0, total_iters)

    while n_iter < total_iters:
        # Initialize WER, TDR, Best_universe
        wep = _MVO_compute_WEP(n_iter, total_iters, minimum, maximum)
        tdr = _MVO_compute_TDR(n_iter, total_iters, p)

        sorted_universes_index = _MVO_sort_universes(population, problem_kind)
        sorted_universes = [population[sort_index]
                            for sort_index in sorted_universes_index]
        
        best_universe = copy.deepcopy(sorted_universes[0])

        norm_infl_rate = _MVO_normalize_infl_rate(sorted_universes)

        for univ_index, universe in enumerate(sorted_universes[1:]):
            black_hole_index = univ_index

            for parameter_index, parameter in enumerate(universe.attribute_list):
                r1 = random.uniform(0, 1)

                if r1 < norm_infl_rate[univ_index]:
                    white_hole_index = _roulette_wheel_selection(-norm_infl_rate)
                    sorted_universes[black_hole_index].attribute_list[parameter_index] = sorted_universes[
                        white_hole_index].attribute_list[parameter_index]

                r2 = random.uniform(0, 1)

                if r2 < wep:
                    r3 = random.uniform(0, 1)
                    r4 = random.uniform(0, 1)

                    if r3 < 0.5:
                        aux_val = (best_universe.attribute_list[parameter_index] + tdr *
                                   (best_universe.interval_by_attr[parameter_index][1] - best_universe.interval_by_attr[parameter_index][0]) *
                                   r4 + best_universe.interval_by_attr[parameter_index][0])

                    else:
                        aux_val = (best_universe.attribute_list[parameter_index] - tdr *
                                   (best_universe.interval_by_attr[parameter_index][1] - best_universe.interval_by_attr[parameter_index][0]) *
                                   r4 + best_universe.interval_by_attr[parameter_index][0])

                    if aux_val < universe.interval_by_attr[parameter_index][0]:
                        aux_val = universe.interval_by_attr[parameter_index][0]
                    elif aux_val > universe.interval_by_attr[parameter_index][1]:
                        aux_val = universe.interval_by_attr[parameter_index][1]

                    sorted_universes[univ_index].attribute_list[parameter_index] = aux_val
        
        if termination_criteria(population):
            break

        printProgressBar(n_iter, total_iters)
        population = sorted_universes
        n_iter += 1


