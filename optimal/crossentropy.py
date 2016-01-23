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
import math
import operator

import numpy

from optimal import optimize

class CrossEntropy(optimize.Optimizer):

    def __init__(self, fitness_function, solution_size,
                 population_size = 20, max_iterations = 100, 
                 pdfs=None, quantile=0.9,
                 **kwargs):
        optimize.Optimizer.__init__(self, fitness_function, population_size, 
                                    max_iterations, **kwargs)
        
        # Parameters for user problem
        self.solution_size = solution_size

        # Cross entropy variables
        if pdfs:
            self.pdfs = pdfs
        else:
            # Create a default set of pdfs
            self.pdfs = random_pdfs(solution_size)
        self.pdf = None # Values initialize in initialize function

        # Quantile is easier to use as an index offset (from max)
        # Higher the quantile, the smaller this offset
        # Setter will automatically set this offset
        self._quantile = None
        self._quantile_offset = None
        self.quantile = quantile

        # Meta optimize parameters
        self.meta_parameters['quantile'] = {'type': 'float', 'min': 0.0, 'max': 1.0}

    def initialize(self):
        # Start with a random pdf
        self.pdf = random.choice(self.pdfs)

    def create_initial_population(self, population_size):
        # Initial population is a uniform random sample
        return sample(self.pdf, population_size)

    def new_population(self, population, fitnesses):
        # Update pdf, then sample new population
        self.pdf = update_pdf(population, fitnesses, self.pdfs, self._quantile_offset)

        # New population is randomly sampled, independent of old population
        return sample(self.pdf, self.population_size)

    # Setters and getters for quantile, so quantile_offset is automatically set
    @property
    def quantile(self):
        return self._quantile

    @quantile.setter
    def quantile(self, value):
        self._quantile = value
        self._quantile_offset = get_quantile_offset(self.population_size, value)

def get_quantile_offset(num_values, quantile):
    return int((num_values-1) * (1.0-quantile))

def random_pdfs(solution_size, num_pdfs=None):
    if num_pdfs == None:
        num_pdfs = solution_size*4

    pdfs = []
    for i in range(num_pdfs):
        # Create random pdf
        pdfs.append([random.uniform(0.0, 1.0) for i in range(solution_size)])
    return pdfs

def sample(probabilities, population_size):
    """Return a random population, drawn with regard to a set of probabilities"""
    population = []
    for i in range(population_size):
        solution = []
        for p in probabilities:
            # p of 1.0: always 1.0
            # p of 0.0: always 0.0
            if random.uniform(0.0, 1.0) < p:
                solution.append(1)
            else:
                solution.append(0)
        population.append(solution)
    return population

def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def chance(solution, pdf):
    """Return the chance of obtaining a solution from a pdf.
    
    The probability of many independant weighted "coin flips" (one for each bit)
    """
    # 1.0 - abs(bit - p) gives probability of bit given p
    return prod([1.0 - abs(bit - p) for bit, p in zip(solution, pdf)])

def pdf_value(pdf, population, fitnesses, fitness_threshold):
    """Give the value of a pdf.

    This represents the likelihood of a pdf generating solutions 
    that exceed the threshold."""
    # Add the chance of obtaining a solution from the pdf
    # when the fitness for that solution exceeds a threshold
    value = 0.0
    for solution, fitness in zip(population, fitnesses):
        if fitness >= fitness_threshold:
            value += math.log(1.0+chance(solution, pdf)) #1.0 + chance to avoid issues with chance of 0

    # The official equation states that value is now divided by len(fitnesses)
    # however, this is unnecessary when we are only obtaining the best pdf,
    # because every solution is of the same size
    return value

def best_pdf(pdfs, population, fitnesses, fitness_threshold):
    # We can use the built in max function
    # we just need to provide a key that provides the value of the 
    # stochastic program defined for cross entropy
    return max(pdfs, key=lambda pdf: pdf_value(pdf, population, fitnesses, fitness_threshold))

def get_quantile_cutoff(values, quantile_offset):
    return sorted(values)[-(quantile_offset+1)]

def update_pdf(population, fitnesses, pdfs, quantile):
    """Find a better pdf, based on fitnesses."""
    # First we determine a fitness threshold based on a quantile of fitnesses
    fitness_threshold = get_quantile_cutoff(fitnesses, quantile)

    # Then check all of our possible pdfs with a stochastic program
    return best_pdf(pdfs, population, fitnesses, fitness_threshold)
