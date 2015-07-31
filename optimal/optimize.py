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
import helpers

class Optimizer(object):
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
        self.meta_parameters = {'population_size': {'type': 'int', 'min': 2, 'max': 1026}}

        #enable logging by default
        self.logging = True

        #set initial values that are used internally
        self.iteration = 0
        self.evaluation_runs = 0
        self.best_solution = None
        self.best_fitness = None
        self.solution_found = False
        self._clear_fitness_dict = True
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
        try:
            self.evaluation_runs = 0
            self.solution_found = False

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

            self.best_solution = best_solution['solution']
            self.best_fitness = best_solution['fitness']
        finally:
            if self._clear_fitness_dict:
                self._fitness_dict = {} # Clear memory

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

    def set_hyperparameters(self, parameters):
        for name, value in parameters.iteritems():
            setattr(self, name, value)

    def meta_optimize(self, meta_optimizer=None, parameter_locks=None, 
                      low_memory=True, smoothing=20):
        """Optimize hyperparameters for a given problem.
        
        Args:
            meta_optimizer: an optional optimizer to use for metaoptimiztion.
            parameter_locks: a list of strings, each corrosponding to a hyperparamter that should not be optimized.
            low_memory: disable performance enhancements to save memory (they use a lot of memory otherwise).
            smoothing: int; number of runs to average over for each set of hyperparameters.
        """
        assert smoothing > 0

        if meta_optimizer == None:
            # Initialize default meta optimizer
            # GenAlg is used because it supports both discrete and continous attributes
            import genalg

            # Copy to avoid permanent modification
            meta_parameters = copy.deepcopy(self.meta_parameters) 

            # First, handle parameter locks, since it will modify our
            # meta_parameters dict
            locked_values = _parse_parameter_locks(self, meta_parameters, 
                                                  parameter_locks)

            # We need to know the size of our chromosome, 
            # based on the hyperparameters to optimize
            solution_size = _get_hyperparameter_solution_size(meta_parameters)

            # We also need to create a decode function to transform the binary solution 
            # into parameters for the metaheuristic
            decode = make_hyperparameter_decode_func(locked_values, meta_parameters)
            
            
            # A master fitness dictionary can be stored for use between calls
            # to meta_fitness
            if low_memory:
                master_fitness_dict = None
            else:
                master_fitness_dict = {}

            # Create metaheuristic with computed decode function and soltuion size
            meta_optimizer = genalg.GenAlg(meta_fitness, solution_size,
                                            _decode_func=decode,
                                            _master_fitness_dict=master_fitness_dict,
                                            _optimizer=self,
                                            _runs=smoothing)
        
        # Determine the best hyperparameters with a metaheuristic
        best_solution = meta_optimizer.optimize()
        best_parameters = decode(best_solution)

        # Set the hyperparameters inline
        self.set_hyperparameters(best_parameters)

        # And return
        return best_parameters

def _parse_parameter_locks(optimizer, meta_parameters, parameter_locks):
    # WARNING: meta_parameters is modified inline

    locked_values = {}
    if parameter_locks:
        for name in parameter_locks:
            # store the current optimzier value
            # and remove from our dictionary of paramaters to optimize
            if parameter_locks and name in parameter_locks:
                locked_values[name] = getattr(optimizer, name)
                meta_parameters.pop(name)

    return locked_values

def _get_hyperparameter_solution_size(meta_parameters):
    # WARNING: meta_parameters is modified inline

    solution_size = 0
    for name, parameters in meta_parameters.iteritems():
        if parameters['type'] == 'discrete':
            # Binary encoding of discrete values -> log_2 N
            num_values = len(parameters['values'])
            binary_size = helpers.binary_size(num_values)
        elif parameters['type'] == 'int':
            # Use enough bits to cover range from min to max
            range = parameters['max'] - parameters['min']
            binary_size = helpers.binary_size(range)
        elif parameters['type'] == 'float':
            # Use enough bits to provide fine steps between min and max
            range = parameters['max'] - parameters['min']
            # * 1000 provides 1000 values between each natural number
            binary_size = helpers.binary_size(range*1000)
        else:
            raise ValueError('Parameter type "{}" does not match known values'.format(parameters['type']))

        # Store binary size with parameters for use in decode function
        parameters['binary_size'] = binary_size

        solution_size += binary_size

    return solution_size

def make_hyperparameter_decode_func(locked_values, meta_parameters):
    # Locked parameters are also returned by decode function, but are not
    # based on solution

    def decode(solution):
        # Start with out stationary (locked) paramaters
        hyperparameters = locked_values

        # Obtain moving hyperparameters from binary solution
        index = 0
        for name, parameters in meta_parameters.iteritems():
            # Obtain binary for this hyperparameter
            binary_size = parameters['binary_size']
            binary = solution[index:index+binary_size]
            index += binary_size # Just index to start of next hyperparameter

            # Decode binary
            if parameters['type'] == 'discrete':
                i = helpers.binary_to_int(binary, 
                            max=len(parameters['values'])-1)
                value = parameters['values'][i]
            elif parameters['type'] == 'int':
                value = helpers.binary_to_int(binary, 
                            offset=parameters['min'], max=parameters['max'])
            elif parameters['type'] == 'float':
                value = helpers.binary_to_float(binary, 
                            minimum=parameters['min'], maximum=parameters['max'])
            else:
                raise ValueError('Parameter type "{}" does not match known values'.format(parameters['type']))

            # Store value
            hyperparameters[name] = value

        return hyperparameters

    return decode

def meta_fitness(solution, _decode_func, _optimizer, _master_fitness_dict, _runs=20):
    """Test a metaheuristic with parameters encoded in solution.
    
    Our goal is to minimize number of evaluation runs until a solution is found,
    while maximizing chance of finding solution to the underlying problem
    NOTE: while meta optimization requires a 'known' solution, this solution 
    can be an estimtate to provide the meta optimizer with a sense of progress.
    """
    parameters = _decode_func(solution)
    
    # Create the optimizer with parameters encoded in solution
    optimizer = copy.deepcopy(_optimizer)
    optimizer.set_hyperparameters(parameters)
    optimizer.logging = False

    # Preload fitness dictionary from master, and disable clearing dict
    # NOTE: master_fitness_dict will be modified inline, and therefore,
    # we do not need to take additional steps to update it
    if _master_fitness_dict != None: # None means low memory mode
        optimizer._clear_fitness_dict = False
        optimizer._fitness_dict = _master_fitness_dict
    
    # Because metaheuristics are stochastic, we run the optimizer multiple times, 
    # to obtain an average of performance
    all_evaluation_runs = []
    solutions_found = []
    for i in range(_runs):
        optimizer.optimize()
        all_evaluation_runs.append(optimizer.evaluation_runs)
        if optimizer.solution_found:
            solutions_found.append(1.0)
        else:
            solutions_found.append(0.0)

    # Our main goal is to minimize time the optimizer takes
    fitness = 1.0 / helpers.avg(all_evaluation_runs)

    # Optimizer is heavily penalized for missing solutions
    fitness = fitness * helpers.avg(solutions_found)**2 + 1e-19 # To avoid 0 fitness

    return fitness