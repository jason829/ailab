import random as rd
import numpy as np
import math

# Defines the initial population with specified size
# Returns population
def i_pop(size, chromosome):
    pop = []
    for inx in range(size):
        pop.append(rd.choices(range(2), k = chromosome))
    
    return pop

# Returns the fitness of individual
def fitness_f(individual):
    return sum(individual)

# Roulette wheel
def Roulette_wheel(pop,fitness):
    parents=[]
    fitotal=sum(fitness)
    normalized=[x/fitotal for x in fitness]
    print('normalized fitness')
    print('________________________')
    print(normalized)
    print('________________________')

    f_cumulative=[]
    index=0
    for n_value in normalized:
        index+=n_value
        f_cumulative.append(index)

    pop_size=len(pop)
    print('cumulative fitness')
    print('________________________')
    print(f_cumulative)
    print('________________________')

    for index2 in range(pop_size):
        rand_n=rd.uniform(0,1)
        individual_n=0
        for fitvalue in f_cumulative:
            if(rand_n<=fitvalue):
                parents.append(pop[individual_n])
                break
            individual_n+=1
    return parents
    
def mutate(chromo):
    for idx in range(len(chromo)):
        if rd.random() < 0.3:
            chromo = chromo[:idx] + [1-chromo[idx]] + chromo[idx + 1:]
    
    return chromo

def mating_crossover(parent_a,parent_b):
    offspring=[]
    cut_point=rd.randint(1, len(parent_a) -1)

    offspring.append(parent_a[:cut_point] + parent_b[cut_point:])
    offspring.append(parent_b[:cut_point] + parent_a[cut_point:])
    return offspring

def main():
    size = 10
    chromosome = 8
    population = i_pop(size,chromosome)
        
    fitness_all = []
    for individual in population:
        fitness_all.append(fitness_f(individual))
    
    parents = Roulette_wheel(population, fitness_all)

    children = []
    for i in range(len(parents) - 1):
        offspring = mating_crossover(parents[i], parents[i+1])
        for child in offspring:
            children.append(mutate(child))
    
    print("***Children produced*** \n", children)
    
main()