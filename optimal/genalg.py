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
"""Genetic algorithm metaheuristic.

Contains base implementation of genetic algorithms.
Specific crossover, selection, etc. functions are implemented in gaoperators.py
"""

import random

from optimal import gaoperators, optimize

class GenAlg(optimize.StandardOptimizer):
    """Canonical Genetic Algorithm

    Perform genetic algorithm optimization with a given fitness function."""

    def __init__(self, fitness_function, chromosome_size, population_size=20,
                 max_iterations=100, mutation_chance=0.02, crossover_chance=0.7,
                 selection_function=gaoperators.roulette_selection,
                 crossover_function=gaoperators.one_point_crossover,
                 **kwargs):
        """Create an object that optimizes a given fitness function with GenAlg.

        Args:
            fitness_function: A function representing the problem to solve,
                              must return a fitness value.
            chromosome_size: The number of genes (bits) in every chromosome.
            population_size: The number of chromosomes in every generation
            max_iterations: The number of iterations to optimize before stopping
            mutation_chance: the chance that a bit will be flipped during mutation
            crossover_chance: the chance that two parents will be crossed during crossover
            selection_function: A function that will select parents for crossover and mutation
            crossover_function: A function that will cross two parents
        """
        super(GenAlg, self).__init__(fitness_function, chromosome_size, population_size,
                                     max_iterations, **kwargs)

        #set genetic algorithm paramaters
        self._mutation_chance = mutation_chance
        self._crossover_chance = crossover_chance
        self._selection_function = selection_function
        self._crossover_function = crossover_function

        # Meta optimize parameters
        self._meta_parameters['_mutation_chance'] = {'type': 'float', 'min': 0.0, 'max': 1.0}
        self._meta_parameters['_crossover_chance'] = {'type': 'float', 'min': 0.0, 'max': 1.0}
        self._meta_parameters['_selection_function'] = {'type': 'discrete',
                                                        'values': [gaoperators.roulette_selection,
                                                                   gaoperators.stochastic_selection]}
        self._meta_parameters['_crossover_function'] = {'type': 'discrete',
                                                        'values': [gaoperators.one_point_crossover,
                                                                   gaoperators.uniform_crossover]}

    def initial_population(self):
        return optimize.make_population(self._population_size, optimize.random_solution_binary,
                                        self._solution_size)

    def next_population(self, population, fitnesses):
        return _new_population_genalg(population, fitnesses,
                                      self._mutation_chance, self._crossover_chance,
                                      self._selection_function, self._crossover_function)

def _new_population_genalg(population, fitnesses, mutation_chance=0.02, crossover_chance=0.7,
                           selection_function='roulette', crossover_function='one_point'):
    """Perform all genetic algorithm operations on a population, and return a new population.

    population must have an even number of chromosomes.

    Args:
        population: A list of binary lists, ex. [[0,1,1,0], [1,0,1,0]]
        fitness: A list of fitnesses that correspond with chromosomes in the population,
                 ex. [1.2, 10.8]
        mutation_chance: the chance that a bit will be flipped during mutation
        crossover_chance: the chance that two parents will be crossed during crossover
        selection_function: A function that will select parents for crossover and mutation
        crossover_function: A function that will cross two parents

    Returns:
        list; A new population of chromosomes, that should be more fit.
    """
    # Selection
    fitness_sum = sum(fitnesses)

    # Generate probabilities
    # Creates a list of increasing values.
    # The greater the gap between two values, the greater the probability.
    # Ex. [0.1, 0.23, 0.56, 1.0]
    prob_sum = 0.0
    probabilities = []
    for fitness in fitnesses:
        if fitness < 0:
            raise ValueError("Fitness cannot be negative, fitness = {}.".format(fitness))
        prob_sum += (fitness/fitness_sum)
        probabilities.append(prob_sum)
    probabilities[-1] += 0.0001 #to compensate for rounding errors

    # Create the population of parents that will be crossed and mutated.
    intermediate_population = selection_function(population, probabilities)

    # Crossover
    new_population = _crossover(intermediate_population, crossover_chance, crossover_function)

    # Mutation
    gaoperators.random_flip_mutate(new_population, mutation_chance) # mutates list in place

    # Return new population
    return new_population

def _crossover(population, crossover_chance, crossover_operator):
    """Perform crossover on a population, return the new crossed-over population."""
    new_population = []
    for i in range(0, len(population), 2): # for every other index
        # Take parents from every set of 2 in the population
        # Wrap index if out of range
        try:
            parents = [population[i], population[i+1]]
        except IndexError:
            parents = [population[i], population[0]]

        crossover = random.uniform(0.0, 1.0)

        if crossover <= crossover_chance: # if crossover takes place
            # Add children to the new population
            new_population.extend(crossover_operator(parents))
        else:
            new_population.append(parents[0][:])
            new_population.append(parents[1][:])

    return new_population
