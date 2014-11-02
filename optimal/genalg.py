###############################################################################
#The MIT License (MIT)
#
#Copyright (c) 2014 Justin Lovinger
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
###############################################################################

import random
import copy

import gaoperators

class GenAlg:
    """Peform genetic algorithm optimization with a given fitness function."""

    def __init__(self, fitness_function, chromosome_size, 
                 population_size=20, generations=100, mutation_chance=0.02, crossover_chance=0.7, 
                 selection_function=gaoperators.roulette_selection, crossover_function=gaoperators.one_point_crossover,
                 **kwargs):
        """Create an object that performs genetic algorithm optimization with a given fitness function.

        Args:
            fitness_function: A function representing the problem to solve, must return a fitness value.
            chromosome_size: The number of genes (bits) in every chromosome.
            population_size: The number of chromosomes in every generation
            generations: The number of generations to optimize before stopping
            mutation_chance: the chance that a bit will be flipped during mutation
            crossover_chance: the chance that two parents will be crossed during crossover
            selection_function: A function that will select parents for crossover and mutation
            crossover_function: A function that will cross two parents
        """
        #set paramaters for users problem
        self.fitness_function = fitness_function
        self.additional_parameters = kwargs #parameters for the users fitness function
        self.chromosome_size = chromosome_size

        #set genetic algorithm paramaters
        self.population_size = population_size
        if self.population_size % 2 == 1: #if population size is odd
            self.population_size += 1 #make population size even
        self.generations = generations
        self.mutation_chance = mutation_chance
        self.crossover_chance = crossover_chance
        self.selection_function = selection_function
        self.crossover_function = crossover_function

        #enable logging by default
        self.logging = True

        #set initial values that are used internally
        self.evaluation_runs = 0
        self.solution_found = False
        self.best_chromosome = None
        self._fitness_dict = {}

    def run_genalg(self):
        """Find the optimal inputs for a given fitness function.
        
        Returns:
            list; A list of genes (0, 1) representing the best solution.
        """
        self._fitness_dict = {}
        self.evaluation_runs = 0
        self.solution_found = False

        best_chromosome = {'chromosome': [], 'fitness': 0.0}
        population = create_initial_population(self.chromosome_size, self.population_size)

        for generation in range(self.generations):
            fitnesses, finished = self.get_fitnesses(population)
            if max(fitnesses) > best_chromosome['fitness']:
                best_chromosome['fitness'] = max(fitnesses)
                best_chromosome['chromosome'] = copy.copy(population[fitnesses.index(max(fitnesses))])

            if self.logging:
                print ('Generation: ' + str(generation))
                print ('Avg Fitness: ' + str(sum(fitnesses)/len(fitnesses)))
                print ('Best Fitness: ' + str(best_chromosome['fitness']))

            if finished:
                self.solution_found = True
                break

            population = new_population(population, fitnesses, self.mutation_chance, self.crossover_chance, self.selection_function, self.crossover_function)

        self._fitness_dict = {}

        self.best_chromosome = best_chromosome['chromosome']
        return self.best_chromosome

    def get_fitnesses(self, population):
        """Get the fitness for every chromosome in a population."""
        fitnesses = []
        finished = False
        for chromosome in population:
            try:
                #attempt to retrieve the fitness from the internal fitness memory
                fitness = self._fitness_dict[str(chromosome)]
            except KeyError:
                #if the fitness is not remembered
                #calculate the fitness, pass in any saved user arguments
                fitness = self.fitness_function(chromosome, **self.additional_parameters)
                #if the user supplied fitness function includes the "finished" flag
                #unpack the results into the finished flag and the fitness
                if isinstance(fitness, tuple):
                    finished = fitness[1]
                    fitness = fitness[0]
                self._fitness_dict[str(chromosome)] = fitness
                self.evaluation_runs += 1 #keep track of how many times fitness is evaluated

            fitnesses.append(fitness)
            if finished:
                break

        return fitnesses, finished

def create_initial_population(chromosome_length, population_size):
    """Create a random initial population of chromosomes.

    Args:
        chromosome_length: an integer representing the length of each chromosome.
        population_size: an integer representing the number of chromosomes in the population.

    Returns:
        list; A list of random chromosomes.
    """
    population = []

    for i in range(population_size): #for every chromosome
        chromosome = []
        for j in range(chromosome_length): #for every bit in the chromosome
            chromosome.append(random.randint(0, 1)) #randomly add a 0 or a 1
        population.append(chromosome) #add the chromosome to the population

    return population

def new_population(population, fitnesses, mutation_chance=0.02, crossover_chance=0.7, selection_function='roulette', crossover_function='one_point'):
    """Perform all genetic algorithm operations on a population, and return a new population.

    population must have an even number of chromosomes.

    Args:
        population: A list of binary lists, ex. [[0,1,1,0], [1,0,1,0]]
        fitness: A list of fitnesses that corrospond with chromosomes in the population, ex. [1.2, 10.8]
        mutation_chance: the chance that a bit will be flipped during mutation
        crossover_chance: the chance that two parents will be crossed during crossover
        selection_function: A function that will select parents for crossover and mutation
        crossover_function: A function that will cross two parents

    Returns:
        list; A new population of chromosomes, that should be more fit.
    """

    #selection
    fitness_sum = sum(fitnesses)
    #generate probabilities
    #creates a list of increasing values. 
    #The greater the gap between two values, the greater the probability. Ex. [0.1, 0.23, 0.56, 1.0]
    prob_sum = 0.0
    probabilities = []
    for fitness in fitnesses:
        if fitness < 0:
            raise ValueError("Fitness cannot be negative, fitness = {}.".format(fitness))
        prob_sum += (fitness/fitness_sum)
        probabilities.append(prob_sum)
    probabilities[-1] += 0.0001 #to compensate for rounding errors

    #create the population of parents that will be crossed and mutated
    #this population can contain duplicates of chromosomes
    intermediate_population = selection_function(population, probabilities)

    #crossover
    new_population = crossover(intermediate_population, crossover_chance, crossover_function) #returns new population

    #mutation
    gaoperators.mutate(new_population, mutation_chance) #mutates list in place

    #return new population
    return new_population

def crossover(population, crossover_chance, crossover_operator):
    """Perform crossover on a population, return the new crossedover population."""

    #population = copy.deepcopy(old_population)

    new_population = []
    for i in range(0, len(population), 2): #for every other index
        parents = [population[i], population[i+1]] #take parents from every set of 2 in the population

        crossover = random.uniform(0.0, 1.0)

        if crossover <= crossover_chance: #if crossover takes place
            new_population.extend(crossover_operator(parents)) #add the children to the new population
        else:
            new_population.append(parents[0][:])
            new_population.append(parents[1][:])

    return new_population

if __name__ == '__main__':
    """Example usage of this library."""
    import math
    import time

    import gahelpers

    #define functions for determining fitness
    def chromosome_to_inputs(chromosome):
        #Helpful functions from gahelpers is used to convert binary to floats
        x1 = gahelpers.binary_to_float(chromosome[0:16], -5, 5)
        x2 = gahelpers.binary_to_float(chromosome[16:32], -5, 5)
        return x1, x2

    #the first argument must always be the chromosome.
    #Additional arguments can optionally come after chromosome
    def get_fitness(chromosome, offset): 
        #Turn our chromosome of bits into floating point values
        x1, x2 = chromosome_to_inputs(chromosome)

        #Ackley's function
        #A common mathematical optimization problem
        output = -20*math.exp(-0.2*math.sqrt(0.5*(x1**2+x2**2)))-math.exp(0.5*(math.cos(2*math.pi*x1)+math.cos(2*math.pi*x2)))+20+math.e
        output += offset

        #You can prematurely stop the genetic algorithm by returning True as the second return value
        #Here, we consider the problem solved if the output is <= 0.01
        if output <= 0.01:
            finished = True
        else:
            finished = False

        #Because this function is trying to minimize the output, a smaller output has a greater fitness
        fitness = 1/output

        return fitness, finished

    #Setup and run the genetic algorithm, using our fitness function, and a chromosome size of 32
    #Additional fitness function arguments are added as keyword arguments
    my_genalg = GenAlg(get_fitness, 32, offset=0) #Yes, offset is completely pointless, but it demonstrates additional arguments
    best_chromosome = my_genalg.run_genalg()
    print chromosome_to_inputs(best_chromosome)