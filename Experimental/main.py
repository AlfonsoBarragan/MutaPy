# Import section

import sos_algorithm
import hunter_spider_algorithm as hs
import math
import functools 

# Declaration section

    #TODO

if __name__ == '__main__':

    print('campo de pruebas')
    black_widow = hs.Hunting_Spider(10)
    prey_aux    = hs.Prey([[0,1], [0,1], [0,1]])

    black_widow.create_web(prey_aux)

    coordinates_oper        = lambda x1, x2: math.pow((x2 - x1), 2)
    sum_funct               = lambda x, y: x+y
    distance_between_points = lambda p1, p2: math.sqrt(functools.reduce(sum_funct, map(coordinates_oper, p1, p2)))
    

    # TODO implementacion de ejemplos para ver que tal va
