###############################################################################
# The MIT License (MIT)
#
# Copyright (c) 2014 Justin Lovinger
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################
"""Genetic algorithm metaheuristic.

Contains base implementation of genetic algorithms.
Specific crossover, selection, etc. functions are implemented in gaoperators.py
"""
import random
import logging

from optimal import gaoperators, optimize, common


class GenAlg(optimize.StandardOptimizer):
    """Canonical Genetic Algorithm

    Perform genetic algorithm optimization with a given fitness function."""

    def __init__(self,
                 chromosome_size,
                 population_size=20,
                 mutation_chance=0.02,
                 crossover_chance=0.7,
                 selection_function=gaoperators.tournament_selection,
                 crossover_function=gaoperators.one_point_crossover):
        """Create an object that optimizes a given fitness function with GenAlg.

        Args:
            chromosome_size: The number of genes (bits) in every chromosome.
            population_size: The number of chromosomes in every generation
            mutation_chance: the chance that a bit will be flipped during mutation
            crossover_chance: the chance that two parents will be crossed during crossover
            selection_function: A function that will select parents for crossover and mutation
            crossover_function: A function that will cross two parents
        """
        super(GenAlg, self).__init__(chromosome_size, population_size)

        if chromosome_size == 1 and crossover_chance > 0.0:
            logging.warning('Crossover not supported with chromosome_size == 1. ' \
                            'Crossover is disabled.')
            crossover_chance = 0.0

        # Set genetic algorithm parameters
        self._mutation_chance = mutation_chance
        self._crossover_chance = crossover_chance
        self._selection_function = selection_function
        self._crossover_function = crossover_function

        # Meta optimize parameters
        self._hyperparameters['_mutation_chance'] = {
            'type': 'float',
            'min': 0.0,
            'max': 1.0
        }
        self._hyperparameters['_crossover_chance'] = {
            'type': 'float',
            'min': 0.0,
            'max': 1.0
        }
        self._hyperparameters['_selection_function'] = {
            'type':
            'discrete',
            'values': [
                gaoperators.roulette_selection,
                gaoperators.stochastic_selection,
                gaoperators.tournament_selection
            ]
        }
        self._hyperparameters['_crossover_function'] = {
            'type':
            'discrete',
            'values':
            [gaoperators.one_point_crossover, gaoperators.uniform_crossover]
        }

    def initial_population(self):
        return common.make_population(self._population_size,
                                      common.random_binary_solution,
                                      self._solution_size)

    def next_population(self, population, fitnesses):
        return _new_population_genalg(
            population, fitnesses, self._mutation_chance,
            self._crossover_chance, self._selection_function,
            self._crossover_function)


def _new_population_genalg(population,
                           fitnesses,
                           mutation_chance=0.02,
                           crossover_chance=0.7,
                           selection_function=gaoperators.tournament_selection,
                           crossover_function=gaoperators.one_point_crossover):
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
    # Create the population of parents that will be crossed and mutated.
    intermediate_population = selection_function(population, fitnesses)

    # Crossover
    new_population = _crossover(intermediate_population, crossover_chance,
                                crossover_function)

    # Mutation
    # Mutates chromosomes in place
    gaoperators.random_flip_mutate(new_population, mutation_chance)

    # Return new population
    return new_population


def _crossover(population, crossover_chance, crossover_operator):
    """Perform crossover on a population, return the new crossed-over population."""
    new_population = []
    for i in range(0, len(population), 2):  # For every other index
        # Take parents from every set of 2 in the population
        # Wrap index if out of range
        try:
            parents = (population[i], population[i + 1])
        except IndexError:
            parents = (population[i], population[0])

        # If crossover takes place
        if random.uniform(0.0, 1.0) <= crossover_chance:
            # Add children to the new population
            new_population.extend(crossover_operator(parents))
        else:
            new_population.extend(parents)

    return new_population
