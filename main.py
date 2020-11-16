import numpy as np
import random
import array
import numpy
from copy import deepcopy
import queue as queue

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import matplotlib.pyplot as plt
import time

# directions
LEFT = 1
RIGHT = 2
UP = 3
DOWN = 4

START = 3
EXIT = 4

PENALTY_WEIGHT = 2.0
DISTANCE_WEIGHT = 2.0

map = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
])

# map = np.array([
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
#     [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
#     [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
#     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
#     [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
# ])
#
# map = np.array([
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
# ])


def get_distance_from_start(path):
    penalty_points = 0
    current_pos_x = 1
    current_pos_y = 1
    for i in range(len(path)):
        x = get_shifted_cord_x_start(current_pos_x, path[i])
        y = get_shifted_cord_y_start(current_pos_y, path[i])
        if map[current_pos_y][current_pos_x] == 1:
            penalty_points += 1
        else:
            current_pos_x, current_pos_y = x, y

    distance = get_absolute_distance_from_start(current_pos_x, current_pos_y)
    return get_weighted_value(distance, penalty_points)


def get_distance(path):
    current_pos_x = 1
    current_pos_y = 1
    map_with_position = deepcopy(map)
    penalty = 0
    for i in range(len(path)):
        current_pos_x = get_shifted_cord_x_start(current_pos_x, path[i])
        current_pos_y = get_shifted_cord_y_start(current_pos_y, path[i])
        map_with_position[current_pos_y][current_pos_x] = i
        if map[current_pos_y][current_pos_x] == 1:
            penalty += 1

    print("Map:")
    print(map)
    print("++++++++++++++")
    print(map_with_position)
    print("Pos: " + str(current_pos_x) + " " + str(current_pos_y))
    print("PEN: " + str(penalty))
    return get_absolute_distance_from_start(current_pos_x, current_pos_y)


def get_distance_from_end(path):
    path_from_end = path[::-1]
    penalty_points = 0
    current_pos_x = 10
    current_pos_y = 10

    for i in range(len(path_from_end)):
        x = get_shifted_cord_x_end(current_pos_x, path[i])
        y = get_shifted_cord_y_end(current_pos_y, path[i])
        if map[current_pos_y][current_pos_x] == 1:
            penalty_points += 1
        else:
            current_pos_x, current_pos_y = x, y

    distance = get_absolute_distance_from_end(current_pos_x, current_pos_y)
    return distance


def get_shifted_cord_x_start(x, direction):
    if direction == LEFT:
        return x - 1
    if direction == RIGHT:
        return x + 1
    return x


def get_shifted_cord_x_end(x, direction):
    if direction == LEFT:
        return x + 1
    if direction == RIGHT:
        return x - 1
    return x


def get_shifted_cord_y_start(y, direction):
    if direction == UP:
        return y - 1
    if direction == DOWN:
        return y + 1
    return y


def get_shifted_cord_y_end(y, direction):
    if direction == UP:
        return y + 1
    if direction == DOWN:
        return y - 1
    return y


def get_absolute_distance_from_start(x, y):
    return abs(10 - x) + abs(10 - y)


def get_absolute_distance_from_end(x, y):
    return (x + y) - 2


def get_weighted_value(distance, penalty):
    value = -(penalty * PENALTY_WEIGHT) - (distance * DISTANCE_WEIGHT)
    return value


def read_chromosome(chromosome):
    res = ""
    for i in range(len(chromosome)):
        if i % 10 == 0:
            res += "\n"
        if chromosome[i] == UP:
            res += " " + "UP"
        if chromosome[i] == DOWN:
            res += " " + "DOWN"
        if chromosome[i] == LEFT:
            res += " " + "LEFT"
        if chromosome[i] == RIGHT:
            res += " " + "RIGHT"

    # print(res)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_int", random.randint, 1, 4)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, 40)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def fitness_v1(individual):
    value = get_distance_from_start(individual)
    return value,


def get_penalty_for_moving_to_visited_node(path):
    penalty_points = 0
    for i in range(int(len(path) // 1.4)):
        if abs(path[i] - path[i + 1]) == 1:
            if not ((path[i] == 2 and path[i + 1] == 3) or (path[i] == 3 and path[i + 1] == 2)):
                penalty_points += 1
    return penalty_points


def fitness_v2(individual):
    value = get_distance_from_start(individual) - get_penalty_for_moving_to_visited_node(individual)
    return value,


def add_nodes_to_queue(route_queue, map, route_map, x, y, map_width):
    if map[y][x + 1] != 1 and route_map[x + 1 + (y * map_width)] == 0:
        route_queue.put(x + 1 + (y * map_width))
        route_map[x + 1 + (y * map_width)] = x + (y * map_width)
    if map[y][x - 1] != 1 and route_map[x - 1 + (y * map_width)] == 0:
        route_queue.put(x - 1 + (y * map_width))
        route_map[x - 1 + (y * map_width)] = x + (y * map_width)
    if map[y + 1][x] != 1 and route_map[x + ((y + 1) * map_width)] == 0:
        route_queue.put(x + ((y + 1) * map_width))
        route_map[x + ((y + 1) * map_width)] = x + (y * map_width)
    if map[y - 1][x] != 1 and route_map[x + ((y - 1) * map_width)] == 0:
        route_queue.put(x + ((y - 1) * map_width))
        route_map[x + ((y - 1) * map_width)] = x + (y * map_width)


def search_route(map, start_pos_x, start_pos_y, end_pos_x, end_pos_y):
    # print(map)
    map_width = len(map)
    route_map = [0 for i in range(map_width * map_width)]
    end_node_cords = end_pos_x + (map_width * end_pos_y)
    route_queue = queue.Queue()
    route_queue.put(start_pos_x + (start_pos_y * map_width))
    route_map[start_pos_x + (start_pos_y * map_width)] = -1
    while True:
        current_node = route_queue.get()
        if current_node == end_node_cords:
            break
        add_nodes_to_queue(route_queue, map, route_map, current_node % map_width, current_node // map_width, map_width)
    # print(route_map)
    read_route(map, start_pos_x, start_pos_y, end_pos_x, end_pos_y, route_map)


def read_route(map, start_pos_x, start_pos_y, end_pos_x, end_pos_y, path):
    map_width = len(map)
    result = []
    start_node = start_pos_x + (start_pos_y * map_width)
    current = end_pos_x + (end_pos_y * map_width)
    result.append(current)
    while start_node != current:
        current = path[current]
        result.append(current)
    result = result[::-1]
    get_directions_from_route(result, map)


def get_directions_from_route(route, map):
    map_length = len(map)
    result = ""
    for index in range(len(route) - 1):
        result += {
            1: 'RIGHT, ',
            -1: 'LEFT, ',
            map_length: 'DOWN, ',
            -map_length: 'UP, ',
        }[route[index + 1] - route[index]]
    # print(result)


MU = 1500
LAMBDA = 1500
MUT_PB = 0.2
CXPB = 0.8
NGEN = 100

GEN_ARR = [i for i in range(NGEN + 1)]
# POP_ARR = [50, 100, 200, 500, 1000, 1200, 1500]
POP_ARR = [1500]
MUT_ARR = [0.01, 0.02, 0.05, 0.1, 0.2]


def get_individals_from_log(log):
    results = []
    for i in range(len(log)):
        results.append(log[i]['max'])
    return results


def get_avg_from_10_rows(row):
    row_sum = 0
    for i in range(len(row)):
        row_sum += row[i]
    return row_sum / len(row)


def get_avg_from_tests(tests):
    result = [-1000 for y in range(len(tests[0]))]
    for i in range(len(tests[0])):
        result[i] = (tests[0][i] + tests[1][i] + tests[2][i] + tests[3][i] + tests[4][i] + tests[5][i] + tests[6][i] + tests[7][i] + tests[8][i] + tests[9][i]) / 10
    return result


def find_optimum():
    individuals = [[] for o in range(10)]
    for population in POP_ARR:
        for mut in MUT_ARR:
            for i in range(0, 10):
                pop, log, hof = do_ga(population, mut)
                individuals[i] = get_individals_from_log(log)
            avg_individuals = get_avg_from_tests(individuals)
            plt.plot(GEN_ARR, avg_individuals, '-', label=f'{population}, {mut}')
        plt.ylabel('Fitness')
        plt.xlabel('Generacje')
        plt.title(f'Optymalizacja parametrów algorytmu (input_2_f_v2_{population})')
        plt.legend()
        plt.autoscale()
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        bottom, top = plt.ylim()
        plt.ylim(bottom, 5)
        plt.axhline(y=-5, color='r', linestyle='-')
        fig.savefig(f'cx80input_2_f_v2_{population}.png', dpi=100)
        plt.close(fig)
        plt.close()


def do_ga(population, mut_pb):
    toolbox.register("evaluate", fitness_v2)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, indpb=mut_pb, low=1, up=4)
    toolbox.register("select", tools.selTournament, tournsize=2)
    pop = toolbox.population(n=population)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("max", numpy.max)

    pop, log = algorithms.eaMuPlusLambda(population=pop, toolbox=toolbox, mu=population, lambda_=population, cxpb=0.8,
                                         mutpb=mut_pb, ngen=NGEN, stats=stats, halloffame=hof, verbose=False)
    return pop, log, hof


def measure_time():
    avg_time = 0
    for i in range(20):
        start = time.time()
        do_ga(1500, 0.05)
        end = time.time()
        avg_time += end - start
    avg_time = avg_time / 20
    print(avg_time)


def main():
    random.seed(20) # It is for case of showing how it works
    # find_optimum()
    pop, log, hof = do_ga(1500, 0.1)
    get_distance(hof[0])
    # get_distance(hof[0])
    # search_route(map, 1, 1, 10, 10)
    # measure_time()

if __name__ == "__main__":
    main()
