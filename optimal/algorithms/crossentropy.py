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
"""Cross entropy (CE) algorithm for optimization.

This algorithm generates solutions from a probability density function (pdf),
and adjusts changes the pdf depending on the fitnesses.
"""

import random
import math
import operator

from optimal import optimize


class CrossEntropy(optimize.StandardOptimizer):
    """Cross entropy optimization."""

    def __init__(self,
                 solution_size,
                 population_size=20,
                 pdfs=None,
                 quantile=0.9):
        super(CrossEntropy, self).__init__(solution_size, population_size)

        # Cross entropy variables
        if pdfs:
            self.pdfs = pdfs
        else:
            # Create a default set of pdfs
            self.pdfs = _random_pdfs(solution_size)
        self.pdf = None  # Values initialize in initialize function

        # Quantile is easier to use as an index offset (from max)
        # Higher the quantile, the smaller this offset
        # Setter will automatically set this offset
        self.__quantile = None
        self.__quantile_offset = None
        self._quantile = quantile

        # Meta optimize parameters
        self._hyperparameters['_quantile'] = {
            'type': 'float',
            'min': 0.0,
            'max': 1.0
        }

    def initialize(self):
        # Start with a random pdf
        self.pdf = random.choice(self.pdfs)

    def initial_population(self):
        # Initial population is a uniform random sample
        return _sample(self.pdf, self._population_size)

    def next_population(self, population, fitnesses):
        # Update pdf, then sample new population
        self.pdf = _update_pdf(population, fitnesses, self.pdfs,
                               self.__quantile_offset)

        # New population is randomly sampled, independent of old population
        return _sample(self.pdf, self._population_size)

    # Setters and getters for quantile, so quantile_offset is automatically set
    @property
    def _quantile(self):
        return self.__quantile

    @_quantile.setter
    def _quantile(self, value):
        self.__quantile = value
        self.__quantile_offset = _get_quantile_offset(self._population_size,
                                                      value)


def _get_quantile_offset(num_values, quantile):
    return int((num_values - 1) * (1.0 - quantile))


def _random_pdfs(solution_size, num_pdfs=None):
    if num_pdfs is None:
        num_pdfs = solution_size * 4

    pdfs = []
    for _ in range(num_pdfs):
        # Create random pdf
        pdfs.append([random.uniform(0.0, 1.0) for _ in range(solution_size)])
    return pdfs


def _sample(probabilities, population_size):
    """Return a random population, drawn with regard to a set of probabilities"""
    population = []
    for _ in range(population_size):
        solution = []
        for probability in probabilities:
            # probability of 1.0: always 1
            # probability of 0.0: always 0
            if random.uniform(0.0, 1.0) < probability:
                solution.append(1)
            else:
                solution.append(0)
        population.append(solution)
    return population


def _prod(iterable):
    return reduce(operator.mul, iterable, 1)


def _chance(solution, pdf):
    """Return the chance of obtaining a solution from a pdf.

    The probability of many independant weighted "coin flips" (one for each bit)
    """
    # 1.0 - abs(bit - p) gives probability of bit given p
    return _prod([1.0 - abs(bit - p) for bit, p in zip(solution, pdf)])


def _pdf_value(pdf, population, fitnesses, fitness_threshold):
    """Give the value of a pdf.

    This represents the likelihood of a pdf generating solutions
    that exceed the threshold.
    """
    # Add the chance of obtaining a solution from the pdf
    # when the fitness for that solution exceeds a threshold
    value = 0.0
    for solution, fitness in zip(population, fitnesses):
        if fitness >= fitness_threshold:
            # 1.0 + chance to avoid issues with chance of 0
            value += math.log(1.0 + _chance(solution, pdf))

    # The official equation states that value is now divided by len(fitnesses)
    # however, this is unnecessary when we are only obtaining the best pdf,
    # because every solution is of the same size
    return value


def _best_pdf(pdfs, population, fitnesses, fitness_threshold):
    # We can use the built in max function
    # we just need to provide a key that provides the value of the
    # stochastic program defined for cross entropy
    return max(
        pdfs,
        key=
        lambda pdf: _pdf_value(pdf, population, fitnesses, fitness_threshold))


def _get_quantile_cutoff(values, quantile_offset):
    return sorted(values)[-(quantile_offset + 1)]


def _update_pdf(population, fitnesses, pdfs, quantile):
    """Find a better pdf, based on fitnesses."""
    # First we determine a fitness threshold based on a quantile of fitnesses
    fitness_threshold = _get_quantile_cutoff(fitnesses, quantile)

    # Then check all of our possible pdfs with a stochastic program
    return _best_pdf(pdfs, population, fitnesses, fitness_threshold)
