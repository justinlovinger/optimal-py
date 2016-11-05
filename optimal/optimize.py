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
"""General optimizer code for any specific algorithm."""

import copy
import random

from optimal import helpers


def random_solution_binary(solution_size):
    """Make a list of random 0s and 1s."""
    return [random.randint(0, 1) for _ in range(solution_size)]


def random_solution_real(solution_size, lower_bounds, upper_bounds):
    """Make a list of random real numbers between lower and upper bounds."""
    return [random.uniform(lower_bounds[i], upper_bounds[i]) for i in range(solution_size)]


def make_population(population_size, solution_generator, *args, **kwargs):
    """Make a population with the supplied generator."""
    return [solution_generator(*args, **kwargs) for _ in range(population_size)]


def _print_fitnesses(iteration, fitnesses, best_solution, frequency=1):
    if iteration == 1 or iteration % frequency == 0:
        print 'Iteration: ' + str(iteration)
        print 'Avg Fitness: ' + str(sum(fitnesses) / len(fitnesses))
        print 'Best Fitness: ' + str(best_solution['fitness'])


class Optimizer(object):
    """Base class for optimization algorithms."""

    def __init__(self, fitness_function,
                 max_iterations=100, **kwargs):
        """Initialize general optimization attributes and bookkeeping

        Args:
            fitness_function: A function representing the problem to solve,
                              must return a fitness value.
            population_size: The number of solutions in every generation
            max_iterations: The number of iterations to optimize before stopping
        """
        # Save users fitness function,
        # parameters for the users fitness function,
        # and decode function
        self._fitness_function = fitness_function
        self._additional_parameters = kwargs
        self._decode_function = lambda x: x

        # Set general algorithm paramaters
        self._max_iterations = max_iterations

        # Parameters for metaheuristic optimization
        self._hyperparameters = {}

        # Enable logging by default
        self.logging = True
        self._logging_func = _print_fitnesses

        # Set initial values that are used internally
        self.__clear_fitness_dict = True
        self.__fitness_dict = {}

        # Bookkeeping
        self.iteration = 0
        self.fitness_runs = 0
        self.best_solution = None
        self.best_fitness = None
        self.solution_found = False

    def __reset_bookkeeping(self):
        """Reset bookkeeping parameters to initial values.

        Call before beginning optimization.
        """
        self.iteration = 0
        self.fitness_runs = 0
        self.best_solution = None
        self.best_fitness = None
        self.solution_found = False

    def initialize(self):
        """Initialize algorithm parameters before each optimization run.

        This method is optional, but useful for some algorithms
        """
        pass

    def initial_population(self):
        """Make the initial population before each optimization run.

        Returns:
            list; a list of solutions.
        """
        raise NotImplementedError("initial_population is not implemented.")

    def next_population(self, population, fitnesses):
        """Make a new population after each optimization iteration.

        Args:
            population: The population current population of solutions.
            fitnesses: The fitness associated with each solution in the population
        Returns:
            list; a list of solutions.
        """
        raise NotImplementedError("new_population is not implemented.")

    def optimize(self):
        """Find the optimal inputs for a given fitness function.

        Returns:
            list; The best solution, as it is encoded.
        """
        self.__reset_bookkeeping()

        # Initialize algorithm
        best_solution = {'solution': [], 'fitness': 0.0}
        self.initialize()
        population = self.initial_population()

        try:
            # Begin optimization loop
            for self.iteration in range(1, self._max_iterations + 1):
                fitnesses, solutions, finished = self.__get_fitnesses(population)

                # If the best fitness from this iteration is better than
                # the global best
                if max(fitnesses) > best_solution['fitness']:
                    # Store the new best solution
                    best_solution['fitness'] = max(fitnesses)
                    best_index = fitnesses.index(best_solution['fitness'])
                    best_solution['solution'] = solutions[best_index][:]

                if self.logging and self._logging_func:
                    self._logging_func(
                        self.iteration, fitnesses, best_solution)

                if finished:
                    self.solution_found = True
                    break

                # Continue optimizing
                population = self.next_population(population, fitnesses)

            # Store best internally, before returning
            self.best_solution = best_solution['solution']
            self.best_fitness = best_solution['fitness']

        # Always clear memory
        finally:
            if self.__clear_fitness_dict:
                self.__fitness_dict = {}  # Clear memory

        return self.best_solution

    def __get_fitnesses(self, population):
        """Get the fitness for every solution in a population."""
        fitnesses = []
        solutions = []
        finished = False
        for encoded_solution in population:
            fitness_key = tuple(encoded_solution)
            try:
                # Attempt to retrieve fitness from cache
                fitness = self.__fitness_dict[fitness_key]

                # Will never be best solution, because we saw it already,
                # so we don't need to decode.
                solution = None
            except KeyError: # Cache miss
                # Decode solution, if user does not provide decode function
                # we simply consider the encoded_solution to be the decoded solution
                solution = self._decode_function(encoded_solution)

                # Get fitness from user defined fitness function,
                # with any argument they provide for it
                fitness = self._fitness_function(
                    solution, **self._additional_parameters)

                # If the user supplied fitness function includes the "finished" flag,
                # unpack the results into the finished flag and the fitness
                if isinstance(fitness, tuple):
                    finished = fitness[1]
                    fitness = fitness[0]
                self.__fitness_dict[fitness_key] = fitness
                self.fitness_runs += 1  # keep track of how many times fitness is evaluated

            fitnesses.append(fitness)
            solutions.append(solution)
            if finished:
                break

        return fitnesses, solutions, finished

    def _set_hyperparameters(self, parameters):
        """Set internal optimization parameters."""
        for name, value in parameters.iteritems():
            try:
                getattr(self, name)
            except AttributeError:
                raise ValueError('Each parameter in parameters must be an attribute. '
                                 '{} is not.'.format(name))
            setattr(self, name, value)

    def _get_hyperparameters(self):
        """Get internal optimization parameters."""
        hyperparameters = {}
        for key in self._hyperparameters:
            hyperparameters[key] = getattr(self, key)
        return hyperparameters

    def optimize_hyperparameters(self, parameter_locks=None, problems=None,
                                 smoothing=20, _meta_optimizer=None, _low_memory=True):
        """Optimize hyperparameters for a given problem.

        Args:
            parameter_locks: a list of strings, each corresponding to a hyperparamter
                             that should not be optimized.
            problems: list of fitness_function, arguments, pairs,
                      allowing optimization based on multiple similar problems.
            low_memory: disable performance enhancements to save memory
                        (they use a lot of memory otherwise).
            smoothing: int; number of runs to average over for each set of hyperparameters.
        """
        if smoothing <= 0:
            raise ValueError('smoothing must be > 0')

        # Copy to avoid permanent modification
        meta_parameters = copy.deepcopy(self._hyperparameters)

        # First, handle parameter locks, since it will modify our
        # meta_parameters dict
        locked_values = _parse_parameter_locks(self, meta_parameters,
                                               parameter_locks)

        # We need to know the size of our chromosome,
        # based on the hyperparameters to optimize
        solution_size = _get_hyperparameter_solution_size(meta_parameters)

        # We also need to create a decode function to transform the binary solution
        # into parameters for the metaheuristic
        decode = _make_hyperparameter_decode_func(
            locked_values, meta_parameters)

        # If the user does not specify a list of problems, default to using
        # only the problem in the optimizer
        if problems is None:
            problems = [(self._fitness_function, self._additional_parameters)]

        # A master fitness dictionary can be stored for use between calls
        # to meta_fitness
        if _low_memory:
            master_fitness_dict = None
        else:
            master_fitness_dict = {}

        additional_parameters = {
            '_decode_func': decode,
            '_optimizer': self,
            '_problems': problems,
            '_runs': smoothing,
            '_master_fitness_dict': master_fitness_dict,
        }
        if _meta_optimizer is None:
            # Initialize default meta optimizer
            # GenAlg is used because it supports both discrete and continous
            # attributes
            import genalg

            # Create metaheuristic with computed decode function and soltuion
            # size
            _meta_optimizer = genalg.GenAlg(
                _meta_fitness, solution_size, **additional_parameters)
        else:
            # Adjust supplied metaheuristic for this problem
            _meta_optimizer._fitness_function = _meta_fitness
            _meta_optimizer._solution_size = solution_size
            _meta_optimizer._additional_parameters = additional_parameters

        # Determine the best hyperparameters with a metaheuristic
        best_solution = _meta_optimizer.optimize()
        best_parameters = decode(best_solution)

        # Set the hyperparameters inline
        self._set_hyperparameters(best_parameters)

        # And return
        return best_parameters


class StandardOptimizer(Optimizer):
    """Adds support for standard metaheuristic hyperparameters."""

    def __init__(self, fitness_function, solution_size, population_size=20,
                 max_iterations=100, **kwargs):
        """Initialize general optimization attributes and bookkeeping

        Args:
            fitness_function: A function representing the problem to solve,
                              must return a fitness value.
            solution_size: The number of values in each solution.
            population_size: The number of solutions in every generation
            max_iterations: The number of iterations to optimize before stopping
        """
        super(StandardOptimizer, self).__init__(
            fitness_function, max_iterations, **kwargs)

        # Set general algorithm paramaters
        self._solution_size = solution_size
        self._population_size = population_size

        # Parameters for metaheuristic optimization
        self._hyperparameters['_population_size'] = {
            'type': 'int', 'min': 2, 'max': 1026}


def _parse_parameter_locks(optimizer, meta_parameters, parameter_locks):
    """Syncronize meta_parameters and locked_values.

    The union of these two sets will have all necessary parameters.
    locked_values will have the parameters specified in parameter_locks.
    """
    # WARNING: meta_parameters is modified inline

    locked_values = {}
    if parameter_locks:
        for name in parameter_locks:
            # Store the current optimzier value
            # and remove from our dictionary of paramaters to optimize
            locked_values[name] = getattr(optimizer, name)
            meta_parameters.pop(name)

    return locked_values


def _get_hyperparameter_solution_size(meta_parameters):
    """Determine size of binary encoding of parameters.

    Also adds binary size information for each parameter.
    """
    # WARNING: meta_parameters is modified inline

    solution_size = 0
    for _, parameters in meta_parameters.iteritems():
        if parameters['type'] == 'discrete':
            # Binary encoding of discrete values -> log_2 N
            num_values = len(parameters['values'])
            binary_size = helpers.binary_size(num_values)
        elif parameters['type'] == 'int':
            # Use enough bits to cover range from min to max
            int_range = parameters['max'] - parameters['min']
            binary_size = helpers.binary_size(int_range)
        elif parameters['type'] == 'float':
            # Use enough bits to provide fine steps between min and max
            float_range = parameters['max'] - parameters['min']
            # * 1000 provides 1000 values between each natural number
            binary_size = helpers.binary_size(float_range * 1000)
        else:
            raise ValueError(
                'Parameter type "{}" does not match known values'.format(parameters['type']))

        # Store binary size with parameters for use in decode function
        parameters['binary_size'] = binary_size

        solution_size += binary_size

    return solution_size


def _make_hyperparameter_decode_func(locked_values, meta_parameters):
    """Create a function that converts the binary solution to parameters."""

    # Locked parameters are also returned by decode function, but are not
    # based on solution

    def decode(solution):
        """Convert solution into dict of hyperparameters."""
        # Start with out stationary (locked) paramaters
        hyperparameters = locked_values

        # Obtain moving hyperparameters from binary solution
        index = 0
        for name, parameters in meta_parameters.iteritems():
            # Obtain binary for this hyperparameter
            binary_size = parameters['binary_size']
            binary = solution[index:index + binary_size]
            index += binary_size  # Just index to start of next hyperparameter

            # Decode binary
            if parameters['type'] == 'discrete':
                i = helpers.binary_to_int(
                    binary, upper_bound=len(parameters['values']) - 1)
                value = parameters['values'][i]
            elif parameters['type'] == 'int':
                value = helpers.binary_to_int(binary,
                                              offset=parameters['min'],
                                              upper_bound=parameters['max'])
            elif parameters['type'] == 'float':
                value = helpers.binary_to_float(binary,
                                                minimum=parameters['min'],
                                                maximum=parameters['max'])
            else:
                raise ValueError(
                    'Parameter type "{}" does not match known values'.format(parameters['type']))

            # Store value
            hyperparameters[name] = value

        return hyperparameters

    return decode


def _meta_fitness(solution, _decode_func, _optimizer, _problems,
                  _master_fitness_dict, _runs=20):
    """Test a metaheuristic with parameters encoded in solution.

    Our goal is to minimize number of evaluation runs until a solution is found,
    while maximizing chance of finding solution to the underlying problem
    NOTE: while meta optimization requires a 'known' solution, this solution
    can be an estimate to provide the meta optimizer with a sense of progress.
    """
    parameters = _decode_func(solution)

    # Create the optimizer with parameters encoded in solution
    optimizer = copy.deepcopy(_optimizer)
    optimizer._set_hyperparameters(parameters)
    optimizer.logging = False

    # Preload fitness dictionary from master, and disable clearing dict
    # NOTE: master_fitness_dict will be modified inline, and therefore,
    # we do not need to take additional steps to update it
    if _master_fitness_dict != None:  # None means low memory mode
        optimizer._Optimizer__clear_fitness_dict = False
        optimizer._Optimizer__fitness_dict = _master_fitness_dict

    # Because metaheuristics are stochastic, we run the optimizer multiple times,
    # to obtain an average of performance
    all_evaluation_runs = []
    solutions_found = []
    for _ in range(_runs):
        for problem in _problems:
            # Set problem
            optimizer._fitness_function = problem[0]
            optimizer._additional_parameters = problem[1]

            # Get performance for problem
            optimizer.optimize()
            all_evaluation_runs.append(optimizer.fitness_runs)
            if optimizer.solution_found:
                solutions_found.append(1.0)
            else:
                solutions_found.append(0.0)

    # Our main goal is to minimize time the optimizer takes
    fitness = 1.0 / helpers.avg(all_evaluation_runs)

    # Optimizer is heavily penalized for missing solutions
    # To avoid 0 fitness
    fitness = fitness * helpers.avg(solutions_found)**2 + 1e-19

    return fitness
