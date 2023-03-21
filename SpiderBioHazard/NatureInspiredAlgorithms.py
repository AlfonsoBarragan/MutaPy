import copy
import math
import numpy as np
import random
import functools


from Galfgets.GraphicsTools import printProgressBar
from .Solution import Solution
from numpy.random import choice

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
        
    # [xi, vi, Pi, Pg]
    particle_elements = [[particle.attribute_list[i], 
                          particle.particle_dict['velocity'][i], 
                          particle.particle_dict['pBest'][i], 
                          best_particle_pos[i]] for i in range(len(particle.attribute_list))]
    
    return [speed_lamb(x) for x in particle_elements]

def _PSO_position(particle:Solution) -> list:
 
    # Lambdas
    post_lamb = lambda x: int(x[0] + x[1]) 
    
    particle_elements = [[particle.attribute_list[i], 
                          particle.particle_dict['velocity'][i]] for i in range(len(particle.attribute_list))]
    
    final_pos = []
    for element_index, element in enumerate([post_lamb(x) for x in particle_elements]):
        if element < particle.interval_by_attr[element_index][0]:
            final_pos.append(particle.interval_by_attr[element_index][0])
        elif element > particle.interval_by_attr[element_index][1]:
            final_pos.append(particle.interval_by_attr[element_index][1])
        else:
            final_pos.append(element)
    
    return final_pos

def _PSO_init(population:list, v_max:float, v_min:float, problem_kind:str='min') -> None:

    if problem_kind == 'min':
        best_fitness_default = 99999
    else:
        best_fitness_default = -99999

    for particle in population:
        particle.particle_dict = {'pBest':0, 'pBest_fitness':0, 'velocity':0}
        particle.particle_dict['pBest'] = copy.deepcopy(particle.attribute_list)
        particle.particle_dict['velocity'] = [random.uniform(v_min,v_max) for i in range(len(particle.attribute_list))]
        particle.particle_dict['pBest_fitness'] = best_fitness_default
    
def PSO_algorithm(total_iters:int, termination_criteria:callable, population:list, init:bool=True, 
                  decreasing_rate:float=0.5, ind_rate:float=1, soc_rate:float=2, 
                  v_max:float=3, v_min:float=1, problem_kind:str='min', verbose:bool=False) -> None:   
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
    
    if verbose:
        printProgressBar(0, total_iters)

    for i in range(total_iters):
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
            
        if verbose:  
            printProgressBar(i, total_iters)

        if termination_criteria(population):
            break

# Symbiotic Organisms Search Algorithm
## Paper's title: Symbiotic Organisms Search: A new metaheuristic optimization algorithm
## DOI: https://doi.org/10.1016/j.compstruc.2014.03.007
## Authors: Min-Yuan Cheng, Doddy Prayogo
## Year: 2014

def _mutualism_phase(Xi_index:int, X_best:int, population:list, problem_kind:str='min') -> None:
    population_copy = [x for x in range(len(population))]

    Xi = population[Xi_index]  
    Xj_index = random.choice(population_copy)
    
    while Xj_index == Xi_index:
        Xj_index = random.choice(population_copy)
    
    Xj = population[Xj_index]

    Xi_final    = Xi
    Xj_final    = Xj

    # Beneficial factor choose
    BF_list = [1,2]
    BF1 = random.choice(BF_list)
    BF_list.remove(BF1)
    BF2 = BF_list[0]

    # Compute new mutualist organisms
    mutual_vector = (Xi.attribute_list + Xj.attribute_list) / 2
    
    Xi_new = copy.deepcopy(Xi)
    Xj_new = copy.deepcopy(Xj)

    profit_i = (X_best.attribute_list - (mutual_vector * BF1)) * random.uniform(0,1)
    profit_j = (X_best.attribute_list - (mutual_vector * BF2)) * random.uniform(0,1)
     
    Xi_new.attribute_list = Xi_new.attribute_list + profit_i
    Xj_new.attribute_list = Xj_new.attribute_list + profit_j
    
    
    if problem_kind == 'min':
        if (Xi.fitness_function() < Xi_new.fitness_function()):
            Xi_final = copy.deepcopy(Xi_new)
        
        if (Xj.fitness_function() < Xj_new.fitness_function()):
            Xj_final = copy.deepcopy(Xj_new)
    elif problem_kind == 'max':
        if (Xi.fitness_function() > Xi_new.fitness_function()):
            Xi_final = copy.deepcopy(Xi_new)
        
        if (Xj.fitness_function() > Xj_new.fitness_function()):
            Xj_final = copy.deepcopy(Xj_new)
    
    population[Xi_index] = copy.deepcopy(Xi_final)         
    population[Xj_index] = copy.deepcopy(Xj_final)

def _commensalism_phase(Xi_index:int, X_best:int, population:list, problem_kind:str='min') -> None:
    population_copy = [x for x in range(len(population))]

    Xi = population[Xi_index]
    Xj_index = random.choice(population_copy)
    
    while Xj_index == Xi_index:
        Xj_index = random.choice(population_copy)
        
    Xj          = population[Xj_index]
    Xi_final    = Xi    
    
    Xi_new      = copy.deepcopy(Xi)
    profit_i    = (X_best.attribute_list - Xj.attribute_list) * random.uniform(-1,1)

    Xi_new.attribute_list = Xi.attribute_list + profit_i
        
    if problem_kind == 'min':
        if (Xi.fitness_function() < Xi_new.fitness_function()):
            Xi_final = Xi_new
    elif problem_kind == 'max':
        if (Xi.fitness_function() > Xi_new.fitness_function()):
            Xi_final = Xi_new

    population[Xi_index] = copy.deepcopy(Xi_final) 

def _parasitism_phase(Xi_index:int, population:list, problem_kind:str='min') -> None:
    population_copy = [x for x in range(len(population))]
    Xi_new = copy.deepcopy(population[Xi_index])
    Xj_index = random.choice(population_copy)
    
    while Xj_index == Xi_index:
        Xj_index = random.choice(population_copy)
        
    Xj          = population[Xj_index]
    Xj_final    = Xj

    Xi_new.mutation_function()

    if problem_kind == 'min':
        if (Xi_new.fitness_function() < Xj.fitness_function()):
            Xj_final = Xi_new
    elif problem_kind == 'max':
        if (Xi_new.fitness_function() > Xj.fitness_function()):
            Xj_final = Xi_new
    
    population[Xj_index] = copy.deepcopy(Xj_final)        

def SOS_algorithm(iterations:int, termination_criteria:callable, 
                  population:list, problem_kind:str='min') -> None:
    n_iteration = 0       

    while(n_iteration < iterations):
    
        for i in range(len(population)):
            fitness_vals    = [organism.fitness_function() for organism in population]
            
            if problem_kind == 'min':
                gBest           = min(fitness_vals)
            else:
                gBest           = max(fitness_vals)
        
            Xi_index = i
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
def _pseudo_random_proportional_rule(solution:Solution, pheromones:np.array, alpha:float, beta:float, nodes_not_discovered:list) -> list:
    posible_nodes = []
    sol_aux = copy.deepcopy(solution)
    
    # Añadir control para evitar la repeticion de nodos que ya hayan sido explorados
    for index, value in enumerate(nodes_not_discovered):
        sol_aux.attribute_list = np.insert(sol_aux.attribute_list, solution.attribute_list.size, value)        
                
        posible_nodes.append((math.pow(pheromones[solution.attribute_list.size - 1][index], alpha) *
                              math.pow(sol_aux.fitness_function(), beta)))
        sol_aux = copy.deepcopy(solution)

        
    denom = functools.reduce(lambda a, b: a+b, posible_nodes)
    
    if denom != 0:
        posible_nodes = list(map(lambda x: x/denom, posible_nodes))
    else:
        prob_equiv = 1/len(posible_nodes)
        posible_nodes = [prob_equiv for x in posible_nodes]
    
    return posible_nodes

def best_node_selection(solution:Solution, pheromones:np.array, alpha:float, beta:float, nodes_not_discovered:list) -> list:
    posible_nodes = []

    sol_aux = copy.deepcopy(solution)
    
    for index, value in enumerate(nodes_not_discovered):
        sol_aux.attribute_list = np.insert(sol_aux.attribute_list, solution.attribute_list.size , value)     
        
        posible_nodes.append((math.pow(pheromones[solution.attribute_list.size - 1][index], alpha) *
                              math.pow(sol_aux.fitness_function(), beta)))
    return posible_nodes

def _init_pheromones(init_pheromone:int, population: list) -> np.array:
    pheromones = np.full((population[0].interval_by_attr.shape[0], 
                          (population[0].interval_by_attr[0][1] - population[0].interval_by_attr[0][0]) + 1),
                          init_pheromone)
    return pheromones

def _init_ants_population(population:list) -> None:
    for i, ant in enumerate(population):
        ant.attribute_list = np.array([])
        ant.attribute_list = np.insert(ant.attribute_list, 0, i % ant.interval_by_attr.shape[0])

def ACS_algorithm(total_iters:int, termination_criteria:callable, population:list, q0:float, 
                  phi:float, init_pheromone:int, persistence:float, alpha:float, beta:float, 
                  pheromones:np.array=np.array([]), problem_kind:str='min',
                  verbose:bool=False) -> np.array:
    n_iter = 0
    
    if pheromones.size == 0:
        pheromones = _init_pheromones(init_pheromone, population)

    if verbose:
        printProgressBar(0, total_iters)
    
    while n_iter < total_iters:
        _init_ants_population(population)
        
        for ant in population:
            nodes_not_discovered = [i for i in range(population[0].interval_by_attr.shape[0])]
            index = nodes_not_discovered.index(ant.attribute_list[0])
            nodes_not_discovered = np.append(nodes_not_discovered[:index], nodes_not_discovered[index+1:])

            while nodes_not_discovered.size != 0:
                # Compute pseudo-random proporcional rule                
                if random.uniform(0,1) <= q0:
                    nodes_viable = best_node_selection(ant, pheromones, alpha, beta,
                                                       nodes_not_discovered)
                    
                    if problem_kind == 'min':
                        node_sel = min(nodes_viable)
                    else:
                        node_sel = max(nodes_viable)
                        
                    new_state  = nodes_viable.index(node_sel)
                    ant.attribute_list = np.append(ant.attribute_list, nodes_not_discovered[new_state])
                    nodes_not_discovered = np.append(nodes_not_discovered[:new_state], nodes_not_discovered[new_state + 1:])                    
                    
                else:
                    nodes_viable = pseudo_random_proportional_rule(ant, pheromones, alpha, beta,
                                                                   nodes_not_discovered)
                    node_sel = choice(nodes_not_discovered, 1, p=nodes_viable)
                    new_state  = np.where(nodes_not_discovered == node_sel[0])[0][0]
                    
                    
                    ant.attribute_list = np.append(ant.attribute_list, nodes_not_discovered[new_state])
                    nodes_not_discovered = np.append(nodes_not_discovered[:new_state], nodes_not_discovered[new_state + 1:])                    
                    
                last_state = ant.attribute_list.size - 1
                
                new_state = np.where(np.array(list(range(ant.interval_by_attr[last_state][1] + 1))) == ant.attribute_list[-1])[0][0]
                pheromones[last_state][new_state] = (((1-phi) * pheromones[last_state][new_state]) + 
                                            (phi * init_pheromone))
     
        fitness_vals = [ant.fitness_function() for ant in population]
        
        if problem_kind == 'min':
            gBest = min(fitness_vals)
        else:
            gBest = max(fitness_vals)
        
        best_ant = population[fitness_vals.index(gBest)]
        
        pheromones_updated = pheromones.copy()
        pheromones_updated = np.array([updating_pheromones(xi, persistence) for xi in pheromones_updated])
                
        for index, attribute in enumerate(best_ant.attribute_list):
            index_1 = np.where(np.array(list(range(best_ant.interval_by_attr[index][1] + 1))) == best_ant.attribute_list[index])[0][0]

            pheromones_updated[index][index_1] = ((1 - persistence) * (pheromones_updated[index][index_1]) +
                                                                 (((1/gBest) * persistence)))            
        
        pheromones = pheromones_updated.copy()
        
        n_iter += 1

        if termination_criteria(population):
            break
        
        if verbose:
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
def _WOA_init(population:list, parameter_a:float) -> np.array:
    a = np.full(shape=[1, len(population[0].attribute_list)], fill_value=parameter_a)
    return a    
    

def _WOA_encircle_search(actual_whale:Solution, whale_to_update:Solution,
                         parameter_A:float, parameter_C:float) -> np.array:
    parameter_D = np.absolute(np.subtract(np.multiply(parameter_C, whale_to_update.attribute_list), actual_whale.attribute_list))
    return np.subtract(whale_to_update.attribute_list, np.multiply(parameter_A, parameter_D))

def _WOA_attack(whale:Solution, best_whale:Solution, 
                constant_b:float, parameter_l:float) -> np.array:
    parameter_D = np.absolute(np.subtract(best_whale.attribute_list, whale.attribute_list))
    np1 = np.multiply(np.multiply(parameter_D, np.exp(constant_b*parameter_l)), np.cos(2.0*np.pi*parameter_l))
    
    return np.add(np1, best_whale.attribute_list)

def _WOA_compute_A(parameter_a:float) -> np.array:
    return np.subtract(np.subtract(np.multiply(parameter_a, 2), np.random.rand(*parameter_a.shape)), parameter_a)

def _WOA_compute_C(parameter_a:float) -> np.array:
    return np.multiply(np.random.rand(*parameter_a.shape) ,2)

def _WOA_amend_whale(whale:Solution) -> None:
    for attr_index, attribute in enumerate(whale.attribute_list):

        if (attribute < whale.interval_by_attr[attr_index][0]) or (attribute > whale.interval_by_attr[attr_index][1]):
            whale.attribute_list[attr_index] = attribute % whale.interval_by_attr[attr_index][1]

def WOA_algorithm(total_iters:int, termination_criteria:callable,
                  problem_kind:str,population:list, a_value:float, 
                  a_step:float, b:float, verbose:bool=False) -> None:

    n_iter = 0
    
    fitness_results = [whale.fitness_function() for whale in population]
    
    if problem_kind == 'min':
        gBest = min(fitness_results)
    else:
        gBest = max(fitness_results)

    best_whale = population[fitness_results.index(gBest)]

    a = _WOA_init(population, a_value)
    a_value_cpy = a_value
    
    if verbose:
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
        
        if verbose:
            printProgressBar(n_iter, total_iters)

        
# Multi-verse optimization algorithm
## Multi-Verse Optimizer: a nature-inspired algorithm for global optimization
## DOI: https://doi.org/10.1007/s00521-015-1870-7
## Authors: Seyedali Mirjaliliab, Seyed Mohammad Mirjalili, Abdolreza Hatamlou
## Year: 2016
def _MVO_compute_WEP(l: float, L: float, minimum: float, maximum: float) -> float:
    return minimum + l * ((maximum-minimum)/L)

def _MVO_compute_TDR(l: float, L: float, p: float) -> float:
    return 1 - ((pow(l, 1/p))/(pow(L, 1/p)))

def _MVO_sort_universes(population:list, problem_kind:str='min') -> list:
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

def _MVO_normalize_infl_rate(population:list) -> np.array:
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

def MVO_algorithm(total_iters:int, termination_criteria:callable, population:list, 
                  minimum:float, maximum:float, p:float, problem_kind:str='min', 
                  verbose:bool=False) -> None:
    n_iter = 0
    
    if verbose:
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

        n_iter += 1
        
        if verbose:
            printProgressBar(n_iter, total_iters)

        population = sorted_universes


# Cuckoo search algorithm
## Cuckoo Search via Levy Flights
## DOI: https://doi.org/10.1007/s00521-015-1870-7
## Authors: Xin-She Yang, Suash Deb
## Year: 2010

def _correct_solution_limits(solution:Solution, type_data:str='int') -> None:
    for attr_index, attr in enumerate(solution.attribute_list):
        if (attr > solution.interval_by_attr[attr_index][1]):
            solution.attribute_list[attr_index] = solution.interval_by_attr[attr_index][1]
        
        elif (attr < solution.interval_by_attr[attr_index][0]):
            solution.attribute_list[attr_index] = solution.interval_by_attr[attr_index][0]
    
    try:
        exec(f"solution.attribute_list = solution.attribute_list.astype({type_data})")
    except Exception as e:
        print(e)

def levy_flight_move(best_cuckoo:Solution, cuckoo:Solution) -> Solution:
    beta = 3/2
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)

    u = np.random.normal(0, sigma, np.shape(cuckoo.attribute_list))
    v = np.random.normal(0, 1, np.shape(cuckoo.attribute_list))
    
    step = u / abs(v) ** (1 / beta)
    step_size = 0.01 * np.array(cuckoo.attribute_list) - np.array(best_cuckoo.attribute_list) * step
    
    
    flying_cuckoo = copy.deepcopy(cuckoo)
    flying_cuckoo.attribute_list = flying_cuckoo.attribute_list + step_size
        
    return flying_cuckoo
    
def CSA_algorithm(total_iters:int, termination_criteria:callable, population:list,
                  verbose:bool=False, problem_kind:str='min') -> None:
    
    if verbose:
        printProgressBar(0, total_iters)

    for iteration in range(total_iters):
        population_copy = [x for x in range(len(population))]
        
        fitness_vals = [x.fitness_function() for x in population]

        if problem_kind == 'min':
            X_best = population[min(fitness_vals)]
        elif problem_kind == 'max':
            X_best = population[max(fitness_vals)]
        
        Xi_index = random.choice(population_copy)
        
        flying_cuckoo = levy_flight_move(X_best, population[Xi_index])
        _correct_solution_limits(flying_cuckoo, 'int')
        
        Xj_index = random.choice(population_copy)
        
        while Xj_index == Xi_index:
            Xj_index = random.choice(population_copy)
        
        if flying_cuckoo.fitness_function() < population[Xj_index].fitness_function():
            population[Xj_index] = copy.deepcopy(flying_cuckoo)
        
        if verbose:
            printProgressBar(iteration, max_iters)

        if termination_criteria(population):
            break
        
    
    