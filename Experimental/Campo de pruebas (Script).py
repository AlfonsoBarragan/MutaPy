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


def clean_population_attr(population):
    for solution in population:
        solution.attribute_list = []


def add_step_to_solution(population):
    for solution_index, solution in enumerate(population):
        solution.attribute_list.append(0)


def check_possible_next_step(solution):
    sol_dims = len(solution.interval_by_attr)

    nodes_not_discovered = [i for i in range(sol_dims-1)]
    nodes_not_discovered = nodes_not_discovered[:solution.attribute_list[0]
                                                ] + nodes_not_discovered[solution.attribute_list[0]+1:]

    return nodes_not_discovered

# %%


def print_basic_info(func):

    def compute(*args, **kwargs):

        func(*args, **kwargs)

        fitness_vals = [part.fitness_function() for part in population]
        print('#' * 100)
        print('Basic statitics from algorithm')
        print(
            f'\t\t>> Best fitness value: {min([part.fitness_function() for part in population])}')
        print(
            f'\t\t>> Mean fitness value: {mean([part.fitness_function() for part in population])}')
        print(
            f'\t\t>> Std-dev fitness value: {standard_desviation([part.fitness_function() for part in population])}\n')

        print('Fitness Values desglosed')
        print(fitness_vals)

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
            sols_summary = [True if last_fit[index] - particle.fitness_function(
            ) > 0 else False for index, particle in enumerate(population)]

            precision_computed.append(
                sols_summary.count(True)/len(sols_summary))

        print('*' * 100)
        print(
            f'\nPrecision of algorithm: {mean(precision_computed)}\nPrecisions obtained: {precision_computed}\n')

    return compute


def recall(nat_alg, total_iters=100):

    def compute(*args, **kwargs):
        print('init measurement...\n')

        last_fit = []
        precision_computed = []

        last_fit = [particle.fitness_function() for particle in population]

        for i in range(total_iters):
            nat_alg(*args, **kwargs)
            sols_summary = [True if last_fit[index] - particle.fitness_function(
            ) < last_fit[index] else False for index, particle in enumerate(population)]
            precision_computed.append(
                sols_summary.count(True)/len(sols_summary))

        print('*' * 100)
        print(
            f'\nPrecision of algorithm: {mean(precision_computed)}\nPrecisions obtained: {precision_computed}\n')

    return compute

# In[42]:


int_by_attr = [number for number in range(0, 244)]


def sum_lamb(x): return x[0]+x[1]

#aux = list(map(sum_lamb, int_by_attr))


print(int_by_attr[:1] + int_by_attr[1+1:])


# In[BenchMark Functions]:

def funct_ackley(self):
    variable_num = len(self.attribute_list)

    tmp1 = 20.-20.*np.exp(-0.2*np.sqrt(1./variable_num *
                          np.sum(np.square(self.attribute_list))))
    tmp2 = np.e-np.exp(1./variable_num *
                       np.sum(np.cos(self.attribute_list*2.*np.pi)))
    return tmp1+tmp2


def funct_sphere(self):
    return np.sum(np.square(self.attribute_list))


def funct_rosenbrock(self):
    f = 0
    for i in range(len(self.attribute_list)-1):
        f += 100*np.power(self.attribute_list[i+1]-np.power(
            self.attribute_list[i], 2), 2)+np.power(self.attribute_list[i]-1, 2)
    return f


def funct_beale(self):
    tmp1 = np.power(
        1.5 - self.attribute_list[0] + self.attribute_list[0] * self.attribute_list[1], 2)
    tmp2 = np.power(
        2.25 - self.attribute_list[0] + self.attribute_list[0] * np.power(self.attribute_list[1], 2), 2)
    tmp3 = np.power(
        2.625 - self.attribute_list[0] + self.attribute_list[0] * np.power(self.attribute_list[1], 3), 2)
    return tmp1+tmp2+tmp3

# In[73]:


def param_sol(size):
    # attrs = list(range(0, size))
    attrs = [n for n in range(0, size)]
    int_by_attr = [(0, size-1) for number in range(0, size)]

    return attrs, int_by_attr

def param_sol2(size):
    # attrs = list(range(0, size))
    attrs = [n for n in range(0, size)]
    int_by_attr = [(-100, 100) for number in range(0, size)]

    return attrs, int_by_attr

def fitness_score(self):
    # check duplicates
    attacks_same_row = 0

    if len(np.unique(self.attribute_list)) != len(self.attribute_list):
        duplicates = np.unique(self.attribute_list)
        for value in duplicates:
            attacks_same_row += (np.sum(self.attribute_list ==
                                 value) - 1) * np.sum(self.attribute_list == value)

    attacks_dia = 0
    for val_ind, value in enumerate(self.attribute_list):
        pos_ori = [val_ind, value]

        aux = np.append(
            self.attribute_list[val_ind+1:], self.attribute_list[:val_ind])

        for val2_ind, value2 in enumerate(aux):
            if abs(pos_ori[0] - val2_ind) == abs(pos_ori[1] - value2):
                attacks_dia += 1

    out_of_bounds = 0

    for value_ind, value in enumerate(self.attribute_list):
        if value < self.interval_by_attr[value_ind][0] or value > self.interval_by_attr[value_ind][1]:
            out_of_bounds += 20

    return attacks_dia/2 + attacks_same_row + out_of_bounds


def fitness_funct(self):
    def sum_lamb(x): return abs(x[0]-x[1])
    attack = 0

    for queen_col, queen_row in enumerate(self.attribute_list):
        queen_pos = [queen_col, queen_row]

        def conf_attacks_dia(x): return abs(
            x[0] - queen_pos[0]) - abs(x[1] - queen_pos[1])

        def conf_attacks_line(x): return (x == queen_pos[1])

        aux = [[index+1, element]
               for index, element in enumerate(self.attribute_list)]

        attacks_1 = map(conf_attacks_dia, aux[:queen_col] + aux[queen_col+1:])
        attacks_2 = map(conf_attacks_line, np.append(
            self.attribute_list[:queen_col], self.attribute_list[queen_col+1:]))

        attacks_1 = np.array(list(attacks_1))
        attacks_2 = np.array(list(attacks_2))

        final_attacks = [True for index, _ in enumerate(attacks_1) if (
            attacks_1[index] == 0) or (attacks_2[index] == True)]

        attack += (final_attacks.count(True))

        if queen_row > self.interval_by_attr[queen_col][1] or queen_row < self.interval_by_attr[queen_col][0]:
            attack += 9999999999

    return attack


def fitness_funct2(self):
    def sum_lamb(x): return abs(x[0]-x[1])
    attack = 0

    for queen_col, queen_row in enumerate(self.attribute_list):
        queen_pos = [queen_col+1, queen_row]

        def conf_attacks_dia(x): return abs(
            x[0] - queen_pos[0]) - abs(x[1] - queen_pos[1])

        def conf_attacks_line(x): return (x == queen_pos[1])

        aux = [[index+1, element]
               for index, element in enumerate(self.attribute_list)]

        attacks_1 = map(conf_attacks_dia, aux[:queen_col] + aux[queen_col+1:])
        attacks_2 = map(
            conf_attacks_line, self.attribute_list[:queen_col] + self.attribute_list[queen_col+1:])

        attacks_1 = list(attacks_1)
        attacks_2 = list(attacks_2)

        final_attacks = [True for index, _ in enumerate(attacks_1) if (
            attacks_1[index] == 0) or (attacks_2[index] == True)]

        attack += (final_attacks.count(True))

    return attack


def random_funct(self):

    for index, attr in enumerate(self.attribute_list):
        self.attribute_list[index] = random.randint(
            self.interval_by_attr[index][0], self.interval_by_attr[index][1])


def random_funct2(self):

    for index, attr in enumerate(self.attribute_list):
        self.attribute_list[index] = random.randint(-100, 100)


def show_function(self):

    mat = [['–' for x in range(self.interval_by_attr[0][1]+1)]
           for y in range(self.interval_by_attr[0][1]+1)]

    for column, row in enumerate(self.attribute_list):
        mat[column][row] = 'Q'

    for r in mat:
        print(str(r).replace(',', '').replace('\'', ''))
    print()


def mutation_function(self):
    n_dim = random.randint(0, len(self.attribute_list))

    dim = list(range(len(self.attribute_list)))

    for i in range(n_dim):
        random_dim = random.choice(dim)
        dim.remove(random_dim)

        self.attribute_list[random_dim] = random.randint(self.interval_by_attr[random_dim][0],
                                                         self.interval_by_attr[random_dim][1])
    # TODO


def add_sols(self, other):
    def add(x): return x[0]+x[1]
    pair_sols = list(zip(self.attribute_list, other.attribute_list))

    return list(map(add, pair_sols))


def subs_sols(self, other):
    def subs(x): return x[0]-x[1]
    pair_sols = list(zip(self.attribute_list, other.attribute_list))

    return list(map(subs, pair_sols))


def mult_sols(self, other):
    def mult(x): return x[0]*x[1]
    pair_sols = list(zip(self.attribute_list, other))

    return list(map(mult, pair_sols))


def div_sols(self, other):
    def div(x): return x[0]/x[1]
    pair_sols = list(zip(self, other))

    return list(map(div, pair_sols))


def generate_population(n_organisms, n_params):
    attrs, int_by_attr = param_sol2(n_params)
    def fit_function(x): return funct_sphere(x)
    def random_function(x): return random_funct(x)

    n_queens_sol = Solution(fit_function, random_function, show_function, mutation_function,
                            add_sols, subs_sols, mult_sols, div_sols, np.array(attrs), int_by_attr)

    n_queens_sol.particle_dict = {
        'pBest': 0, 'pBest_fitness': 0, 'velocity': 0}

    population = []

    for i in range(n_organisms):
        population.append(copy.deepcopy(n_queens_sol))
        population[i].randomize_function()

    return population


# In[74]:


attrs, int_by_attr = param_sol(5)
def fit_function(x): return fitness_funct(x)
def random_function(x): return random_funct(x)


n_queens_sol = Solution(fit_function, random_function, show_function, mutation_function,
                        add_sols, subs_sols, mult_sols, div_sols, attrs, int_by_attr)

population = generate_population(100, 50)

population_ACS = generate_population(100, 16)


# ### PSO ALGORITHM

# In[120]:


def PSO_speed(weight_factor, particle, best_particle_pos, vmax, vmin, individual_learn, social_learn):

    # Lambdas
    def speed_lamb(x): return int(weight_factor * x[1] + individual_learn * random.random() * (x[2]-x[0]) +
                                  social_learn * random.random() * (x[3]-x[0]))

    def dgt_lamb(x): return vmax if x >= vmax else vmin if x <= vmin else x

    # [xi, vi, Pi, Pg]
    particle_elements = [[particle.attribute_list[i],
                          particle.particle_dict['velocity'][i],
                          particle.particle_dict['pBest'][i],
                          best_particle_pos[i]] for i in range(len(particle.attribute_list))]

    # return list(map(dgt_lamb, list(map(speed_lamb, particle_elements))))
    return list(map(speed_lamb, particle_elements))


def PSO_position(particle):

    # Lambdas
    def post_lamb(x): return int(x[0] + x[1])

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
        particle.particle_dict['pBest'] = copy.deepcopy(
            particle.attribute_list)
        particle.particle_dict['velocity'] = [random.uniform(
            v_min, v_max) for i in range(len(particle.attribute_list))]
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
                particle.particle_dict['pBest'] = copy.deepcopy(
                    particle.attribute_list)
                particle.particle_dict['pBest_fitness'] = particle.fitness_function(
                )

        fitness_vals = [particle.fitness_function() for particle in population]
        gBest = min(fitness_vals)

        best_part = population[fitness_vals.index(gBest)]

        for particle in population:
            particle.particle_dict['velocity'] = PSO_speed(weight_fact, particle, best_part.attribute_list,
                                                           v_max, v_min, ind_rate, soc_rate)
            particle.attribute_list = copy.deepcopy(PSO_position(particle))

        printProgressBar(i, max_iters)


# %%

# PSO_algorithm(population, 1, True, 0.5, 1, 2, 10, 0.01)
# PSO_algorithm(population, 10, False, 0.5, 1, 2, 10, 0.01)
# PSO_algorithm(population, 10, False, 0.5, 1, 2, 10, 0.01)
# PSO_algorithm(population, 2000, False, 0.7, 3, 5, 100, 0.001)
PSO_algorithm(population, 100, False, 0.7, 1, 900, 10, 0.01)


fitness_vals = [particle.fitness_function() for particle in population]
print(fitness_vals)

print(fitness_vals.index(min(fitness_vals)))
print(min(fitness_vals))
print(population[fitness_vals.index(min(fitness_vals))].attribute_list)


# precision_PSO = precision(PSO_algorithm, 20)(
#     population, 100, False, 0.7, 3, 5, 1002, 0.1)

# %%
precision_SOS = precision(SOS_algorithm, 5)(10, 1, population)

# %%

# ### SOS ALGORITHM


def mult_sol_float(attr, value): return map(lambda x: x*value, attr)


def div_list_int(attr, value): return map(lambda x: x/value, attr)


def _mutualism_phase(Xi_index, X_best, population):
    # Organisms selection
    population_copy = copy.deepcopy(population)
    Xi = population[Xi_index]

    Xj = random.choice(
        population_copy[0:Xi_index]+population_copy[Xi_index+1:])
    Xj_index = population_copy.index(Xj)

    Xi_final = Xi
    Xj_final = Xj

    # Beneficial factor choose
    BF_list = [1, 2]
    BF1 = random.choice(BF_list)
    BF_list.remove(BF1)
    BF2 = BF_list[0]

    # Compute new mutualist organisms
    mutual_vector = div_list_int(Xi + Xj, 2)

    Xi_new = copy.deepcopy(Xi)
    Xj_new = copy.deepcopy(Xj)

    profit_i = mult_sol_float(list(map(lambda x: x[0]*x[1],
                                       list(zip(X_best.attribute_list, mutual_vector)))),
                              random.uniform(0, 1))

    profit_j = mult_sol_float(list(map(lambda x: x[0]*x[1],
                                       list(zip(X_best.attribute_list, mutual_vector)))),
                              random.uniform(0, 1))

    Xi_new.atrribute_list = list(
        map(lambda x: x[0]+x[1], zip(Xi_new.attribute_list, profit_i)))
    Xj_new.attribute_list = list(
        map(lambda x: x[0]+x[1], zip(Xj_new.attribute_list, profit_j)))

    if (Xi.fitness_function() < Xi_new.fitness_function()):
        Xi_final = copy.deepcopy(Xi_new)

    if (Xj.fitness_function() < Xj_new.fitness_function()):
        Xj_final = copy.deepcopy(Xj_new)

    population[Xi_index] = copy.deepcopy(Xi_final)
    population[Xj_index] = copy.deepcopy(Xj_final)


def _commensalism_phase(Xi_index, X_best, population):
    population_copy = copy.deepcopy(population)
    Xi = population[Xi_index]

    Xj = random.choice(
        population_copy[0:Xi_index]+population_copy[Xi_index+1:])

    Xi_final = Xi

    Xi_new = copy.deepcopy(Xi)
    profit_i = mult_sol_float(X_best - Xj, random.uniform(-1, 1))

    Xi_new.attribute_list = list(
        map(lambda x: x[0]+x[1], list(zip(Xi.attribute_list, profit_i))))

    if (Xi.fitness_function() < Xi_new.fitness_function()):
        Xi_final = Xi_new

    population[Xi_index] = copy.deepcopy(Xi_final)


def _parasitism_phase(Xi_index, population):
    population_copy = copy.deepcopy(population)
    Xi = population[Xi_index]

    Xj = random.choice(
        population_copy[0:Xi_index]+population_copy[Xi_index+1:])

    Xj_final = Xj

    Xi.mutation_function()

    if (Xi.fitness_function() > Xj.fitness_function()):
        Xj_final = Xi

    population[population_copy.index(Xj)] = copy.deepcopy(Xj_final)


@print_basic_info
def SOS_algorithm(iterations: int, termination_criteria, population):
    n_iteration = 0

    printProgressBar(0, iterations)
    while(n_iteration < iterations):

        for i in range(len(population)):
            fitness_vals = [organism.fitness_function()
                            for organism in population]
            gBest = min(fitness_vals)

            Xi_index = i % len(population)
            X_best = population[fitness_vals.index(gBest)]

            _mutualism_phase(Xi_index, X_best, population)

            _commensalism_phase(Xi_index, X_best, population)

            _parasitism_phase(Xi_index, population)

        printProgressBar(n_iteration, iterations)

        n_iteration += 1


# In[84]:


get_ipython().run_cell_magic('time', '', 'SOS_algorithm(2000, lambda x:1, population)')


# In[85]:


fitness_vals = [particle.fitness_function() for particle in population]
print(fitness_vals)

print(population[8].attribute_list)
print(population[9].attribute_list)

print(fitness_vals.index(min(fitness_vals)))
print(min(fitness_vals))

# %%


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


def ACS_init_pheromones(sol_dims, init_pheromone):
    pheromones = np.full((sol_dims, sol_dims), init_pheromone)
    pheromones[np.eye(sol_dims) == 1] = 0

    return pheromones


def ACS_algorithm(total_iters, termination_criteria, population, q0,
                  phi, init_pheromone, persistence, alpha, beta, pheromones):

    n_iter = 0

    sol_dims = len(population[0].interval_by_attr)

    if len(pheromones) == 0:
        pheromones = ACS_init_pheromones(sol_dims, init_pheromone)

    def evaporate_pheromones(phero): return (1-persistence)*phero

    clean_population_attr(population)

    # while termination_criteria(population):
    printProgressBar(0, total_iters)
    while n_iter < total_iters:

        add_step_to_solution(population)

        for ant in population:
            nodes_not_discovered = check_possible_next_step(ant)

            while nodes_not_discovered != []:
                # Compute pseudo-random proporcional rule
                if random.uniform(0, 1) <= q0:
                    nodes_viable = best_node_selection(ant, pheromones, alpha, beta,
                                                       nodes_not_discovered)

                    node_sel = min(nodes_viable)
                    new_state = nodes_viable.index(node_sel)

                    ant.attribute_list.append(nodes_not_discovered[new_state])
                    nodes_not_discovered = nodes_not_discovered[:new_state] + \
                        nodes_not_discovered[new_state+1:]

                else:
                    nodes_viable = pseudo_random_proportional_rule(ant, pheromones, alpha, beta,
                                                                   nodes_not_discovered)

                    node_sel = choice(nodes_not_discovered, 1, nodes_viable)
                    new_state = node_sel[0]

                    ant.attribute_list.append(new_state)

                    nodes_not_discovered.remove(new_state)

                last_state = ant.attribute_list[-2]

                pheromones[last_state][new_state] = (((1-phi) * pheromones[last_state][new_state]) +
                                                     (phi * init_pheromone))

        fitness_vals = [ant.fitness_function() for ant in population]
        gBest = min(fitness_vals)

        best_ant = population[fitness_vals.index(gBest)]

        pheromones_updated = copy.deepcopy(pheromones)
        pheromones_updated = np.multiply(
            pheromones_updated, np.full((sol_dims, sol_dims), 1 - persistence))

        best_ant_path = zip(best_ant.attribute_list[:len(
            best_ant.attribute_list)-1], best_ant.attribute_list[1:])

        for tuple_nodes in best_ant_path:
            pheromones_updated[tuple_nodes[0]][tuple_nodes[1]] = ((1 - persistence) * (pheromones_updated[tuple_nodes[0]][tuple_nodes[1]]) +
                                                                  (((1/gBest) * persistence)))
        if n_iter != total_iters-1:
            clean_population_attr(population)

        pheromones = copy.deepcopy(pheromones_updated)

        n_iter += 1
        printProgressBar(n_iter, total_iters)

    return pheromones

# In[445]:


phero = []

file = open('results.txt', 'w')

for i in range(20):
    phero = ACS_algorithm(100, '', population_ACS, 0.3,
                          0.2, 30, 0.4, 0.4, 1.5, phero)

    fitness_vals = [particle.fitness_function() for particle in population_ACS]
    print(fitness_vals)

    print(
        f'ant_attributes: {population_ACS[fitness_vals.index(min(fitness_vals))].attribute_list}\nant_index: {fitness_vals.index(min(fitness_vals))} ant_fitness: {min(fitness_vals)}')
    print(f'fitness mean: {mean(fitness_vals)}')

    population_ACS[fitness_vals.index(min(fitness_vals))].show_solution()

    file.write(f'iteration: {i} (100 subiterations)\n')
    file.write(
        f'ant_attributes: {population_ACS[fitness_vals.index(min(fitness_vals))].attribute_list}\nant_index: {fitness_vals.index(min(fitness_vals))} ant_fitness: {min(fitness_vals)}\n')
    file.write(f'fitness mean: {mean(fitness_vals)}\n')

    file.write('######################################\n')

file.close()

# pheromones = ACS_algorithm(10, '', population_ACS, 0.5, 1, 30, 1, 0.5, 0.4, phero)


# In[446]:


fitness_vals = [particle.fitness_function() for particle in population_ACS]
print(fitness_vals)

print(
    f'ant_attributes: {population_ACS[fitness_vals.index(min(fitness_vals))].attribute_list}\nant_index: {fitness_vals.index(min(fitness_vals))} ant_fitness: {min(fitness_vals)}')
print(f'fitness mean: {mean(fitness_vals)}')
population_ACS[fitness_vals.index(min(fitness_vals))].show_solution()


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


def WOA_init(population, parameter_a):
    a = np.full(shape=[1, len(population[0].attribute_list)],
                fill_value=parameter_a)
    return a


def WOA_encircle_search(actual_whale, whale_to_update, parameter_A, parameter_C):
    parameter_D = np.absolute(np.subtract(np.multiply(
        parameter_C, whale_to_update.attribute_list), actual_whale.attribute_list))
    return np.subtract(whale_to_update.attribute_list, np.multiply(parameter_A, parameter_D))


def WOA_attack(whale, best_whale, constant_b, parameter_l):
    parameter_D = np.absolute(np.subtract(
        best_whale.attribute_list, whale.attribute_list))
    np1 = np.multiply(np.multiply(parameter_D, np.exp(
        constant_b*parameter_l)), np.cos(2.0*np.pi*parameter_l))

    return np.add(np1, best_whale.attribute_list)


def WOA_compute_A(parameter_a):
    return np.subtract(np.subtract(np.multiply(parameter_a, 2), np.random.rand(*parameter_a.shape)), parameter_a)


def WOA_compute_C(parameter_a):
    return np.multiply(np.random.rand(*parameter_a.shape), 2)


def WOA_amend_whale(whale):
    for attr_index, attribute in enumerate(whale.attribute_list):

        if (attribute < whale.interval_by_attr[attr_index][0]) or (attribute > whale.interval_by_attr[attr_index][1]):
            whale.attribute_list[attr_index] = attribute % whale.interval_by_attr[attr_index][1]

@print_basic_info
def WOA_algorithm(total_iters, population, a_value, a_step, b):

    n_iter = 0

    fitness_results = [whale.fitness_function() for whale in population]
    best_whale = population[fitness_results.index(min(fitness_results))]

    a = WOA_init(population, a_value)
    a_value_cpy = a_value

    printProgressBar(0, total_iters)

    while n_iter < total_iters:
        for whale_index, whale in enumerate(population):

            if a_value_cpy - a_step >= 0:
                a -= np.full(shape=a.shape, fill_value=a_step)
                a_value_cpy -= a_step

            else:
                a = np.full(shape=a.shape, fill_value=0)

            A = WOA_compute_A(a)
            C = WOA_compute_C(a)

            l = np.random.rand(*a.shape)
            p = random.uniform(0, 1)

            if p < 0.5:
                # bubble net hunting
                if np.linalg.norm(A) < 1:
                    new_attributes = WOA_encircle_search(
                        whale, best_whale, A, C)

                else:
                    random_whale = random.choice(
                        population[0:whale_index]+population[whale_index+1:])
                    new_attributes = WOA_encircle_search(
                        whale, random_whale, A, C)
            else:
                new_attributes = WOA_attack(whale, best_whale, b, l)

            whale.attribute_list = [int(x) for x in new_attributes.tolist()[0]]
            WOA_amend_whale(whale)

        fitness_results = [whale.fitness_function() for whale in population]
        best_whale = population[fitness_results.index(min(fitness_results))]

        n_iter += 1

        printProgressBar(n_iter, total_iters)


def VDWOA_algorithm(total_iters, population, a_value, a_step, b, epsilon):

    n_iter = 0

    fitness_results = [whale.fitness_function() for whale in population]
    best_whale = population[fitness_results.index(min(fitness_results))]

    a = WOA_init(population, a_value)
    a_value_cpy = a_value

    printProgressBar(0, total_iters)

    while n_iter < total_iters:
        for whale_index, whale in enumerate(population):

            if a_value_cpy - a_step >= 0:
                a -= np.full(shape=a.shape, fill_value=a_step)
                a_value_cpy -= a_step

            else:
                a = np.full(shape=a.shape, fill_value=0)

            A = WOA_compute_A(a)
            C = WOA_compute_C(a)

            l = np.random.rand(*a.shape)
            p = random.uniform(0, 1)

            if p < 0.5:
                # bubble net hunting
                if np.linalg.norm(A) < 1:

                    if a==1: # Corregir luego, esto esta mal
                        new_attributes = WOA_encircle_search(
                            whale, best_whale, A, C)

                else:
                    random_whale = random.choice(
                        population[0:whale_index]+population[whale_index+1:])
                    new_attributes = WOA_encircle_search(
                        whale, random_whale, A, C)
            else:
                new_attributes = WOA_attack(whale, best_whale, b, l)

            whale.attribute_list = [int(x) for x in new_attributes.tolist()[0]]
            WOA_amend_whale(whale)

        fitness_results = [whale.fitness_function() for whale in population]
        best_whale = population[fitness_results.index(min(fitness_results))]

        n_iter += 1

        printProgressBar(n_iter, total_iters)


# %%
def MVU_algorithm(total_iters, population, minimum, maximum, p, order: str = "min"):
    n_iter = 0
    printProgressBar(0, total_iters)

    while n_iter < total_iters:
        # Initialize WER, TDR, Best_universe
        wep = compute_WEP(n_iter, total_iters, minimum, maximum)
        tdr = compute_TDR(n_iter, total_iters, p)

        sorted_universes_index = sort_universes(population, order)
        sorted_universes = [population[sort_index]
                            for sort_index in sorted_universes_index]
        
        best_universe = copy.deepcopy(sorted_universes[0])

        norm_infl_rate = normalize_infl_rate(sorted_universes)

        for univ_index, universe in enumerate(sorted_universes[1:]):
            black_hole_index = univ_index

            for parameter_index, parameter in enumerate(universe.attribute_list):
                r1 = random.uniform(0, 1)

                if r1 < norm_infl_rate[univ_index]:
                    white_hole_index = roulette_wheel_selection(-norm_infl_rate)
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

        printProgressBar(n_iter, total_iters)
        population = sorted_universes
        n_iter += 1


def compute_WEP(l: float, L: float, minimum: float, maximum: float):
    return minimum + l * ((maximum-minimum)/L)


def compute_TDR(l: float, L: float, p: float):
    return 1 - ((pow(l, 1/p))/(pow(L, 1/p)))


def sort_universes(population, order: str = "min"):
    inflation_rate_list = [universe.fitness_function()
                           for universe in population]
    inflation_rate_list_sorted = copy.deepcopy(inflation_rate_list)

    if (order == "min"):
        inflation_rate_list_sorted.sort()

    else:
        inflation_rate_list_sorted.sort(reverse=True)

    sorted_universes = [inflation_rate_list.index(
        inflation_rate_value) for inflation_rate_value in inflation_rate_list_sorted]

    return sorted_universes


def normalize_infl_rate(population):
    inflation_rate_list = [universe.fitness_function()
                           for universe in population]

    infl_rate_max = max(inflation_rate_list)
    infl_rate_min = min(inflation_rate_list)

    normalized_infl_rate_list = [(infl_rate - infl_rate_min)/(
        infl_rate_max-infl_rate_min) for infl_rate in inflation_rate_list]

    return np.array(normalized_infl_rate_list)


def roulette_wheel_selection(weights):
    accumulation = np.cumsum(weights)
    p = random.random() * accumulation[-1]

    chosen_index = -1

    for index in range(0, len(accumulation)):

        if accumulation[index] > p:
            chosen_index = index
            break

    choice = chosen_index
    return choice


# %%

fitness_vals = [universe.fitness_function() for universe in population]
print(fitness_vals)

print(
    f'universe_attributes: {population[fitness_vals.index(min(fitness_vals))].attribute_list}\nuniverse_index: {fitness_vals.index(min(fitness_vals))} universe_fitness: {min(fitness_vals)}')
print(f'fitness mean: {mean(fitness_vals)}')
population[fitness_vals.index(min(fitness_vals))].show_solution()

# %%
fitness_results = [whale.fitness_function() for whale in population]
best_whale = population[fitness_results.index(min(fitness_results))]

evolution.append(mean(fitness_results))

evolution = []

print('------------------------------------')


print(f'best_fitness = {best_whale.fitness_function()}\nattributes = {best_whale.attribute_list}\nfitness_mean = {mean(fitness_results)}')
print('------------------------------------')

for i in range(100):
    WOA_algorithm(10000, population, 2.0, 0.2, 1)

    fitness_results = [whale.fitness_function() for whale in population]
    best_whale = population[fitness_results.index(min(fitness_results))]

    evolution.append(mean(fitness_results))

    print(
        f'best_fitness = {best_whale.fitness_function()}\nattributes = {best_whale.attribute_list}\nfitness_mean = {mean(fitness_results)}')
    print('------------------------------------')


# %%

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
    def sum_lamb(x): return abs(x[0]-x[1])

    attack = 0

    for queen_col, queen_row in enumerate(self.attribute_list):
        queen_pos = [queen_col+1, queen_row]
        print(queen_pos)

        # conf_attacks = lambda x: (queen_pos[1] == x)

        def conf_attacks_dia(x): return (x == sum_lamb(queen_pos))

        if queen_pos[1] == 1:
            print(queen_pos)

            if queen_pos[0] == 1:
                def conf_attacks_dia(x): return ((x % (queen_pos[1]+1 + queen_pos[0]+1) == 0) or
                                                 (queen_pos[1]+1 + queen_pos[0]+1) % x == 0)

            elif queen_pos[0] == len(self.attribute_list):
                def conf_attacks_dia(x): return ((x % (queen_pos[1]+1 + queen_pos[0]-1) == 0) or
                                                 (queen_pos[1]+1 + queen_pos[0]-1) % x == 0)

            else:
                def conf_attacks_dia(x): return ((x % (queen_pos[1]+1 + queen_pos[0]+1) == 0) or
                                                 (x % (queen_pos[1]+1 + queen_pos[0]-1) == 0) or
                                                 ((queen_pos[1]+1 + queen_pos[0]+1) % x == 0) or
                                                 ((queen_pos[1]+1 + queen_pos[0]-1) % x == 0))

        elif queen_pos[1] == len(self.attribute_list):

            if queen_pos[0] == 1:
                def conf_attacks_dia(x): return ((x % (queen_pos[1]-1 + queen_pos[0]+1) == 0) or
                                                 ((queen_pos[1]-1 + queen_pos[0]+1) % x == 0))

            elif queen_pos[0] == len(self.attribute_list):
                def conf_attacks_dia(x): return ((x % (queen_pos[1]-1 + queen_pos[0]-1) == 0) or
                                                 ((queen_pos[1]-1 + queen_pos[0]-1) % x == 0))

            else:
                def conf_attacks_dia(x): return ((x % (queen_row-1 + queen_pos[0]-1) == 0) or
                                                 (x % (queen_pos[1]-1 + queen_pos[0]+1) == 0) or
                                                 ((queen_pos[1]-1 + queen_pos[0]-1) % x == 0) or
                                                 ((queen_pos[1]-1 + queen_pos[0]+1) % x == 0))

        else:

            if queen_pos[0] == 1:
                def conf_attacks_dia(x): return ((x % (queen_pos[1]+1 + queen_pos[0]+1) == 0) or
                                                 (x % (queen_pos[1]-1 + queen_pos[0]+1) == 0) or
                                                 ((queen_pos[1]+1 + queen_pos[0]+1) % x == 0) or
                                                 ((queen_pos[1]-1 + queen_pos[0]+1) % x == 0))

            elif queen_pos[0] == len(self.attribute_list):
                def conf_attacks_dia(x): return ((x % (queen_pos[1]-1 + queen_pos[0]-1) == 0) or
                                                 (x % (queen_pos[1]+1 + queen_pos[0]-1) == 0) or
                                                 ((queen_pos[1]-1 + queen_pos[0]-1) % x == 0) or
                                                 ((queen_pos[1]+1 + queen_pos[0]-1) % x == 0))

            else:
                def conf_attacks_dia(x): return ((x % (queen_pos[1]-1 + queen_pos[0]-1) == 0) or
                                                 (x % (queen_pos[1]+1 + queen_pos[0]+1) == 0) or
                                                 (x % (queen_pos[1]+1 + queen_pos[0]-1) == 0) or
                                                 (x % (queen_pos[1]-1 + queen_pos[0]+1) == 0) or
                                                 ((queen_pos[1]-1 + queen_pos[0]-1) % x == 0) or
                                                 ((queen_pos[1]+1 + queen_pos[0]+1) % x == 0) or
                                                 ((queen_pos[1]+1 + queen_pos[0]-1) % x == 0) or
                                                 ((queen_pos[1]-1 + queen_pos[0]+1) % x == 0))

        attacks_1 = list(map(
            conf_attacks, self.attribute_list[:queen_col] + self.attribute_list[queen_col+1:]))

        aux = [[index+1, element]
               for index, element in enumerate(self.attribute_list)]

        print(aux[:queen_col] + aux[queen_col+1:])
        print(list(map(sum_lamb, aux[:queen_col] + aux[queen_col+1:])))

        attacks_2 = list(map(conf_attacks_dia, list(
            map(sum_lamb, aux[:queen_col] + aux[queen_col+1:]))))
        print(attacks_2)

        attack += (attacks_1.count(True) + attacks_2.count(True))

    return attack / len(self.attribute_list)


attrs, int_by_attr = param_sol(4)

print(attrs)

n_queens_sol = Organism(attrs, int_by_attr)
n_queens_sol._fitness_function = lambda x: fitness_funct(x)

n_queens_sol.fitness_function()
