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
"""Population-based incremental learning (PBIL) algorithm

This is a simple estimation of distribution algorithm (EDA),
that evolves a probability vector, and samples from this vector.
"""
import numpy

from optimal.optimize import StandardOptimizer
from optimal.algorithms import crossentropy


class PBIL(StandardOptimizer):
    """Population-based incremental learning (PBIL) algorithm.
    
    Args:
        solution_size: Number of bits in every solution.
        population_size: Number of solutions in every iteration.
        adjust_rate: Rate at which probability vector will move
            towards best solution, every iteration.
        mutation_chance: Chance for any given probability to mutate,
            every iteration.
            Defaults to 1 / solution_size.
        mutation_adjust_rate: Rate at which a probability will move
            towards a random probability, if it mutates.
    """
    def __init__(self,
                 solution_size,
                 population_size=20,
                 adjust_rate=0.1,
                 mutation_chance=None,
                 mutation_adjust_rate=0.05):
        super(PBIL, self).__init__(solution_size, population_size)

        if mutation_chance is None:
            mutation_chance = 1.0 / solution_size

        # PBIL hyperparameters
        self._adjust_rate = adjust_rate
        self._mutation_chance = mutation_chance
        self._mutation_adjust_rate = mutation_adjust_rate

        self._hyperparameters['_adjust_rate'] = {
            'type': 'float',
            'min': 0.0,
            'max': 1.0
        }
        self._hyperparameters['_mutation_chance'] = {
            'type': 'float',
            'min': 0.0,
            'max': 1.0
        }
        self._hyperparameters['_mutation_adjust_rate'] = {
            'type': 'float',
            'min': 0.0,
            'max': 1.0
        }

        # PBIL parameters
        self._probability_vec = None # Initialize in initialize function

    def initialize(self):
        """Initialize algorithm parameters before each optimization run.

        This method is optional, but useful for some algorithms
        """
        self._probability_vec = 0.5 * numpy.ones(self._solution_size)

    def initial_population(self):
        """Make the initial population before each optimization run.

        Returns:
            list; a list of solutions.
        """
        return [
            _sample(self._probability_vec)
            for _ in range(self._population_size)
        ]

    def next_population(self, population, fitnesses):
        """Make a new population after each optimization iteration.

        Args:
            population: The population current population of solutions.
            fitnesses: The fitness associated with each solution in the population
        Returns:
            list; a list of solutions.
        """
        # Update probability vector
        self._probability_vec = _adjust_probability_vec_best(
            population, fitnesses, self._probability_vec, self._adjust_rate)

        # Mutate probability vector
        _mutate_probability_vec(self._probability_vec, self._mutation_chance,
                                self._mutation_adjust_rate)

        # Return new samples
        return [
            _sample(self._probability_vec)
            for _ in range(self._population_size)
        ]


def _sample(probability_vec):
    """Return random binary string, with given probabilities."""
    return map(int,
               numpy.random.random(probability_vec.size) <= probability_vec)


def _adjust_probability_vec_best(population, fitnesses, probability_vec,
                                 adjust_rate):
    """Shift probabilities towards the best solution."""
    best_solution = max(zip(fitnesses, population))[1]

    # Shift probabilities towards best solution
    return _adjust(probability_vec, best_solution, adjust_rate)


def _mutate_probability_vec(probability_vec, mutation_chance, mutation_adjust_rate):
    """Randomly adjust probabilities.

    WARNING: Modifies probability_vec argument.
    """
    bits_to_mutate = numpy.random.random(probability_vec.size) <= mutation_chance
    probability_vec[bits_to_mutate] = _adjust(
        probability_vec[bits_to_mutate],
        numpy.random.random(numpy.sum(bits_to_mutate)), mutation_adjust_rate)


def _adjust(old_value, new_value, rate):
    return old_value + rate * (new_value - old_value)
