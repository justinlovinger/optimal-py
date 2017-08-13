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
import operator
import collections

from optimal import helpers


class Problem(object):
    """The problem to solve.

    Contains everything needed to decode and calculate the fitness
    of a potential solution to a problem.
    """
    def __init__(self, fitness_function, decode_function=lambda x: x,
                 fitness_args=[], decode_args=[],
                 fitness_kwargs={}, decode_kwargs={}):
        self._fitness_function = fitness_function
        self._decode_function = decode_function

        self._fitness_args = fitness_args
        self._decode_args = decode_args

        self._fitness_kwargs = fitness_kwargs
        self._decode_kwargs = decode_kwargs

    def copy(self, fitness_function=None, decode_function=None,
             fitness_args=None, decode_args=None,
             fitness_kwargs=None, decode_kwargs=None):
        """Return a copy of this problem.

        Optionally replace this problems arguments with those passed in.
        """
        if fitness_function is None:
            fitness_function = self._fitness_function
        if decode_function is None:
            decode_function = self._decode_function
        if fitness_args is None:
            fitness_args = self._fitness_args
        if decode_args is None:
            decode_args = self._decode_args
        if fitness_kwargs is None:
            fitness_kwargs = self._fitness_kwargs
        if decode_kwargs is None:
            decode_kwargs = self._decode_kwargs

        return Problem(
            fitness_function, decode_function=decode_function,
            fitness_args=fitness_args, decode_args=decode_args,
            fitness_kwargs=fitness_kwargs, decode_kwargs=decode_kwargs)

    def get_fitness(self, solution):
        """Return fitness for the given solution."""
        return self._fitness_function(solution, *self._fitness_args, **self._fitness_kwargs)

    def decode_solution(self, encoded_solution):
        """Return solution from an encoded representation."""
        return self._decode_function(encoded_solution, *self._decode_args, **self._decode_kwargs)


class Optimizer(object):
    """Base class for optimization algorithms."""

    def __init__(self):
        """Initialize general optimization attributes and bookkeeping."""
        # Runtime parameters, keep in object, so subclasses can view
        self.__max_iterations = None

        # Parameters for metaheuristic optimization
        self._hyperparameters = {}

        # Enable logging by default
        self.logging = True
        self._logging_func = _print_fitnesses

        # Set caching parameters
        self.cache_encoded_solution = True
        self.cache_decoded_solution = True
        self.clear_cache = True
        self.__encoded_cache = {}
        self.__decoded_cache = {}
        self._get_decoded_key = self._get_decoded_key_type

        # Bookkeeping
        self.iteration = 0
        self.fitness_runs = 0
        self.best_solution = None
        self.best_fitness = None
        self.solution_found = False

    @property
    def _max_iterations(self):
        return self.__max_iterations

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

    def optimize(self, problem, max_iterations=100):
        """Find the optimal inputs for a given fitness function.

        Args:
            problem: An instance of Problem. The problem to solve.
            max_iterations: The number of iterations to optimize before stopping.

        Returns:
            object; The best solution, after decoding.
        """
        if not isinstance(problem, Problem):
            raise TypeError('problem must be an instance of Problem class')

        # Set first, incase optimizer uses _max_iterations in initialization
        self.__max_iterations = max_iterations

        # Initialize algorithm
        self._reset()

        best_solution = {'solution': None, 'fitness': None}
        population = self.initial_population()
        try:
            # Begin optimization loop
            for self.iteration in range(1, self._max_iterations + 1):
                fitnesses, solutions, finished = self._get_fitnesses(problem, population)

                # If the best fitness from this iteration is better than
                # the global best
                best_index, best_fitness = max(enumerate(fitnesses), key=operator.itemgetter(1))
                if best_fitness > best_solution['fitness']:
                    # Store the new best solution
                    best_solution['fitness'] = best_fitness
                    best_solution['solution'] = solutions[best_index]

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
            if self.clear_cache:
                # Clear caches from memory
                self.__encoded_cache = {}
                self.__decoded_cache = {}

                # Reset decoded cache key
                self._get_decoded_key = self._get_decoded_key_type

        return self.best_solution

    def _reset(self):
        self._reset_bookkeeping()
        self.initialize()

    def _reset_bookkeeping(self):
        """Reset bookkeeping parameters to initial values.

        Call before beginning optimization.
        """
        self.iteration = 0
        self.fitness_runs = 0
        self.best_solution = None
        self.best_fitness = None
        self.solution_found = False

    def _get_fitnesses(self, problem, population):
        """Get the fitness for every solution in a population."""
        fitnesses = []
        solutions = []
        finished = False

        # Get fitness for each potential soluion in population
        for encoded_solution in population:
            # First cache level, encoded cache
            encoded_key = tuple(encoded_solution)
            try:
                # Attempt to retrieve fitness from cache
                fitness = self.__encoded_cache[encoded_key]

                # Will never be best solution, because we saw it already,
                # so we don't need to decode.
                solution = None

            except KeyError: # Cache miss
                # Decode solution, if user does not provide decode function
                # we simply consider the encoded_solution to be the decoded solution
                solution = problem.decode_solution(encoded_solution)

                # Second cache level, decoded cache
                decoded_key = self._get_decoded_key(solution)
                try:
                    fitness = self.__decoded_cache[decoded_key]

                    # Add to cache
                    if self.cache_decoded_solution:
                        self.__encoded_cache[encoded_key] = fitness
                    
                except KeyError: # Cache miss
                    # Get fitness from user defined fitness function,
                    # with any argument they provide for it
                    fitness_finished = problem.get_fitness(solution)

                    # If the user supplied fitness function includes the "finished" flag,
                    # unpack the results into the finished flag and the fitness
                    try:
                        fitness, finished = fitness_finished
                    except TypeError: # Not (fitness, finished) tuple
                        fitness = fitness_finished

                    # Add to caches
                    if self.cache_encoded_solution:
                        self.__encoded_cache[encoded_key] = fitness
                    if self.cache_decoded_solution and decoded_key is not None:
                        self.__decoded_cache[decoded_key] = fitness

                    # Bookkeeping
                    self.fitness_runs += 1  # keep track of how many times fitness is evaluated

            # Fitness calculated, add to list
            fitnesses.append(fitness)
            solutions.append(solution) # Remember solution, in case it is the best
            if finished: # Break early if optimization is finished
                break

        return fitnesses, solutions, finished

    def _get_decoded_key_type(self, solution):
        # Start by just trying to hash it
        try:
            {solution: None}
            self._get_decoded_key = self._get_decoded_key_simple
        except:
            # Not hashable
            # Try tuple

            # Before trying tuple, check if dict
            # tuple(dict) will return a tuple of the KEYS only
            if isinstance(solution, dict):
                self._get_decoded_key = self._get_decoded_key_dict
            else:
                # Not dict, try tuple
                try:
                    {tuple(solution): None}
                    self._get_decoded_key = self._get_decoded_key_tuple
                except:
                    # Cannot convert to tuple
                    # Try str
                    try:
                        {str(solution): None}
                        self._get_decoded_key = self._get_decoded_key_str
                    except:
                        # Nothing works, give up
                        self._get_decoded_key = self._get_decoded_key_none

        # Done discovering, return key
        return self._get_decoded_key(solution)

    def _get_decoded_key_simple(self, solution):
        return solution

    def _get_decoded_key_tuple(self, solution):
        return tuple(solution)

    def _get_decoded_key_str(self, solution):
        return str(solution)

    def _get_decoded_key_dict(self, solution):
        return tuple(solution.items())

    def _get_decoded_key_none(self, solution):
        return None

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

    def optimize_hyperparameters(self, problems, parameter_locks=None,
                                 smoothing=20, max_iterations=100,
                                 _meta_optimizer=None, _low_memory=True):
        """Optimize hyperparameters for a given problem.

        Args:
            parameter_locks: a list of strings, each corresponding to a hyperparamter
                             that should not be optimized.
            problems: Either a single problem, or a list of problem instances,
                     allowing optimization based on multiple similar problems.
            smoothing: int; number of runs to average over for each set of hyperparameters.
            max_iterations: The number of iterations to optimize before stopping.
            _low_memory: disable performance enhancements to save memory
                         (they use a lot of memory otherwise).
        """
        if smoothing <= 0:
            raise ValueError('smoothing must be > 0')

        # problems supports either one or many problem instances
        if isinstance(problems, collections.Iterable):
            for problem in problems:
                if not isinstance(problem, Problem):
                    raise TypeError('problem must be Problem instance or list of Problem instances')
        elif isinstance(problems, Problem):
            problems = [problems]
        else:
            raise TypeError('problem must be Problem instance or list of Problem instances')

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
        decode = _make_hyperparameter_decode_func(locked_values, meta_parameters)

        # A master fitness dictionary can be stored for use between calls
        # to meta_fitness
        if _low_memory:
            master_fitness_dict = None
        else:
            master_fitness_dict = {}

        additional_parameters = {
            '_optimizer': self,
            '_problems': problems,
            '_runs': smoothing,
            '_master_fitness_dict': master_fitness_dict,
        }
        META_FITNESS = Problem(_meta_fitness_func, decode_function=decode,
                               fitness_kwargs=additional_parameters)
        if _meta_optimizer is None:
            # Initialize default meta optimizer
            # GenAlg is used because it supports both discrete and continous
            # attributes
            from optimal import GenAlg

            # Create metaheuristic with computed decode function and soltuion
            # size
            _meta_optimizer = GenAlg(solution_size)
        else:
            # Adjust supplied metaheuristic for this problem
            _meta_optimizer._solution_size = solution_size

        # Determine the best hyperparameters with a metaheuristic
        best_parameters = _meta_optimizer.optimize(META_FITNESS, max_iterations=max_iterations)

        # Set the hyperparameters inline
        self._set_hyperparameters(best_parameters)

        # And return
        return best_parameters


class StandardOptimizer(Optimizer):
    """Adds support for standard metaheuristic hyperparameters."""

    def __init__(self, solution_size, population_size=20):
        """Initialize general optimization attributes and bookkeeping

        Args:
            solution_size: The number of values in each solution.
            population_size: The number of solutions in every generation.
        """
        super(StandardOptimizer, self).__init__()

        # Set general algorithm paramaters
        self._solution_size = solution_size
        self._population_size = population_size

        # Parameters for metaheuristic optimization
        self._hyperparameters['_population_size'] = {
            'type': 'int', 'min': 2, 'max': 1026}


def _print_fitnesses(iteration, fitnesses, best_solution, frequency=1):
    if iteration == 1 or iteration % frequency == 0:
        print 'Iteration: ' + str(iteration)
        print 'Avg Fitness: ' + str(sum(fitnesses) / len(fitnesses))
        print 'Best Fitness: ' + str(best_solution['fitness'])


def _parse_parameter_locks(optimizer, meta_parameters, parameter_locks):
    """Synchronize meta_parameters and locked_values.

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
            # + 1 to include max in range
            int_range = parameters['max'] - parameters['min'] + 1
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
        hyperparameters = copy.deepcopy(locked_values)

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


def _meta_fitness_func(parameters, _optimizer, _problems,
                       _master_fitness_dict, _runs=20):
    """Test a metaheuristic with parameters encoded in solution.

    Our goal is to minimize number of evaluation runs until a solution is found,
    while maximizing chance of finding solution to the underlying problem
    NOTE: while meta optimization requires a 'known' solution, this solution
    can be an estimate to provide the meta optimizer with a sense of progress.
    """
    # Create the optimizer with parameters encoded in solution
    optimizer = copy.deepcopy(_optimizer)
    optimizer._set_hyperparameters(parameters)
    optimizer.logging = False

    # Preload fitness dictionary from master, and disable clearing dict
    # NOTE: master_fitness_dict will be modified inline, and therefore,
    # we do not need to take additional steps to update it
    if _master_fitness_dict != None:  # None means low memory mode
        optimizer.clear_cache = False
        optimizer._Optimizer__encoded_cache = _master_fitness_dict

    # Because metaheuristics are stochastic, we run the optimizer multiple times,
    # to obtain an average of performance
    all_evaluation_runs = []
    solutions_found = []
    for _ in range(_runs):
        for problem in _problems:
            # Get performance for problem
            optimizer.optimize(problem)
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
