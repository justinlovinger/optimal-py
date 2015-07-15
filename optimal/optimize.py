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

import copy

class Optimizer:
    """Base class for optimization algorithms."""

    def __init__(self, fitness_function, population_size=20, max_iterations=100,
                 **kwargs):
        """Create an object that performs genetic algorithm optimization with a given fitness function.

        Args:
            fitness_function: A function representing the problem to solve, must return a fitness value.
            population_size: The number of solutions in every generation
            max_iterations: The number of iterations to optimize before stopping
        """
        #set paramaters for users problem
        self.fitness_function = fitness_function
        self.additional_parameters = kwargs #parameters for the users fitness function

        #set general algorithm paramaters
        self.population_size = population_size
        self.max_iterations = max_iterations

        # Parameters for metaheuristic optimization
        self.meta_parameters = {'population_size': {'min': 2, 'max': 1000}}

        #enable logging by default
        self.logging = True

        #set initial values that are used internally
        self.iteration = 0
        self.evaluation_runs = 0
        self.best_solution = None
        self.best_fitness = None
        self.solution_found = False
        self._fitness_dict = {}

        # Parameters for algorithm specific functions
        self.initial_pop_args = []
        self.new_pop_args = []

    def initialize(self):
        pass

    def create_initial_population(self, *args, **kwargs):
        raise NotImplementedError("create_initial_population is not implemented.")

    def new_population(self, *args, **kwargs):
        raise NotImplementedError("new_population is not implemented.")

    def optimize(self):
        """Find the optimal inputs for a given fitness function.
        
        Returns:
            list; The best solution, as it is encoded.
        """
        self.evaluation_runs = 0
        self.solution_found = False
        self._fitness_dict = {}

        best_solution = {'solution': [], 'fitness': 0.0}
        self.initialize()
        population = self.create_initial_population(self.population_size)

        for self.iteration in range(self.max_iterations):
            fitnesses, finished = self.get_fitnesses(population)
            if max(fitnesses) > best_solution['fitness']:
                best_solution['fitness'] = max(fitnesses)
                best_solution['solution'] = copy.copy(population[fitnesses.index(max(fitnesses))])

            if self.logging:
                print ('Iteration: ' + str(self.iteration))
                print ('Avg Fitness: ' + str(sum(fitnesses)/len(fitnesses)))
                print ('Best Fitness: ' + str(best_solution['fitness']))

            if finished:
                self.solution_found = True
                break

            population = self.new_population(population, fitnesses)

        self._fitness_dict = {}

        self.best_solution = best_solution['solution']
        self.best_fitness = best_solution['fitness']
        return self.best_solution

    def get_fitnesses(self, population):
        """Get the fitness for every solution in a population."""
        fitnesses = []
        finished = False
        for solution in population:
            try:
                #attempt to retrieve the fitness from the internal fitness memory
                fitness = self._fitness_dict[str(solution)]
            except KeyError:
                #if the fitness is not remembered
                #calculate the fitness, pass in any saved user arguments
                fitness = self.fitness_function(solution, **self.additional_parameters)
                #if the user supplied fitness function includes the "finished" flag
                #unpack the results into the finished flag and the fitness
                if isinstance(fitness, tuple):
                    finished = fitness[1]
                    fitness = fitness[0]
                self._fitness_dict[str(solution)] = fitness
                self.evaluation_runs += 1 #keep track of how many times fitness is evaluated

            fitnesses.append(fitness)
            if finished:
                break

        return fitnesses, finished

    def meta_optimize(self):
        """Optimize parameters for a given problem."""