import random
import numpy
import math

import optimize

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

def random_pdfs(solution_size):
    pdfs = []
    for i in range(solution_size*2):
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

def chance(solution, pdf):
    """Return the chance of obtaining a solution from a pdf.
    
    This is the average of the chance of each bit given the probability for that bit.
    """
    c = 0.0
    for bit, p in zip(solution, pdf):
        c += 1.0 - abs(bit - p)
    return c / len(solution)

def pdf_value(pdf, population, fitnesses, fitness_threshold):
    """Give the value of a pdf.

    This represents the likelihood of a pdf generating solutions 
    that exceed the threshold."""
    # Add the chance of obtaining a solution from the pdf
    # when the fitness for that solution exceeds a threshold
    value = 0.0
    for solution, fitness in zip(population, fitnesses):
        if fitness >= fitness_threshold:
            value += math.log(chance(solution, pdf))

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

if __name__ == '__main__':
    """Example usage of this library.
    
    See examplefunctions.py for instructions on how to create a fitness function
    """
    import examplefunctions

    # Setup and run the probabilistic evolution, using our fitness function, 
    # and a chromosome size of 32
    # Additional fitness function arguments are added as keyword arguments
    ce = CrossEntropy(examplefunctions.ackley, 32, 
                       decode_func=examplefunctions.ackley_binary)
    best_solution = ce.optimize()
    print examplefunctions.ackley_binary(best_solution)