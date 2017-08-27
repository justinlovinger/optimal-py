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
import functools
try:
    import pickle
    import dill
    import multiprocessing
except ImportError:
    pass

from optimal import helpers


class UnhashableError(Exception):
    """Raise when an object cannot be hashed."""


def _identity(x):
    """Return x."""
    return x


def _print_fitnesses(iteration,
                     population,
                     solutions,
                     fitnesses,
                     best_solution,
                     best_fitness,
                     frequency=1):
    if iteration == 1 or iteration % frequency == 0:
        print 'Iteration: %s\nAvg Fitness: %s\nBest Fitness: %s' % (
            iteration, (sum(fitnesses) / len(fitnesses)), best_fitness)


class Problem(object):
    """The problem to solve.

    Contains everything needed to decode and calculate the fitness
    of a potential solution to a problem.

    Args:
        fitness_function: Function mapping solution to a fitness value.
            Or a (fitness, finished) tuple.
            If finished is True, optimization ends immediately.
        decode_function: Function mapping encoded string to a potential solution.
            fitness_function takes the output of decode_function.
        fitness_args: List of arguments (past the first) for fitness_function.
        decode_args: List of arguments (past the first) for decode_function.
        fitness_kwargs: Dict of keyword arguments for fitness_function.
        decode_kwargs: Dict of keyword arguments for decode_function.
        hash_solution_func: Function mapping solution to unique hash of solution.
    """

    def __init__(self,
                 fitness_function,
                 decode_function=_identity,
                 fitness_args=[],
                 decode_args=[],
                 fitness_kwargs={},
                 decode_kwargs={},
                 hash_solution_func=None):
        self._fitness_function = fitness_function
        self._decode_function = decode_function

        self._fitness_args = fitness_args
        self._decode_args = decode_args

        self._fitness_kwargs = fitness_kwargs
        self._decode_kwargs = decode_kwargs

        self.hash_solution = hash_solution_func

    def copy(self,
             fitness_function=None,
             decode_function=None,
             fitness_args=None,
             decode_args=None,
             fitness_kwargs=None,
             decode_kwargs=None):
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
            fitness_function,
            decode_function=decode_function,
            fitness_args=fitness_args,
            decode_args=decode_args,
            fitness_kwargs=fitness_kwargs,
            decode_kwargs=decode_kwargs)

    def get_fitness(self, solution):
        """Return fitness for the given solution."""
        return self._fitness_function(solution, *self._fitness_args,
                                      **self._fitness_kwargs)

    def decode_solution(self, encoded_solution):
        """Return solution from an encoded representation."""
        return self._decode_function(encoded_solution, *self._decode_args,
                                     **self._decode_kwargs)


class Optimizer(object):
    """Base class for optimization algorithms."""

    def __init__(self):
        """Initialize general optimization attributes and bookkeeping."""
        # Runtime parameters, keep in object, so subclasses can view
        self.__max_iterations = None

        # Parameters for metaheuristic optimization
        self._hyperparameters = {}

        # Set caching parameters
        self.__encoded_cache = {}
        self.__solution_cache = {}
        self._get_solution_key = self._get_solution_key_type

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

    def optimize(self, problem, max_iterations=100,
                 cache_encoded=True, cache_solution=False, clear_cache=True,
                 logging_func=_print_fitnesses,
                 n_processes=0):
        """Find the optimal inputs for a given fitness function.

        Args:
            problem: An instance of Problem. The problem to solve.
            max_iterations: The number of iterations to optimize before stopping.
            cache_encoded: bool; Whether or not to cache fitness of encoded strings.
                Encoded strings are produced directly by the optimizer.
                If an encoded string is found in cache, it will not be decoded.
            cache_solution: bool; Whether or not to cache fitness of decoded solutions.
                Decoded solution is provided by problems decode function.
                If problem does not provide a hash solution function,
                Various naive hashing methods will be attempted, including:
                    tuple, tuple(sorted(dict.items)), str.
            clear_cache: bool; Whether or not to reset cache after optimization.
                Disable if you want to run optimize multiple times on the same problem.
            logging_func: func/None; Function taking:
                iteration, population, solutions, fitnesses, best_solution, best_fitness
                Called after every iteration.
                Use for custom logging, or set to None to disable logging.
                Note that best_solution and best_fitness are the best of all iterations so far.
            n_processes: int; Number of processes to use for multiprocessing.
                If <= 0, do not use multiprocessing.

        Returns:
            object; The best solution, after decoding.
        """
        if not isinstance(problem, Problem):
            raise TypeError('problem must be an instance of Problem class')

        # Prepare pool for multiprocessing
        if n_processes > 0:
            try:
                pool = multiprocessing.Pool(processes=n_processes)
            except NameError:
                raise ImportError(
                    'pickle, dill, or multiprocessing library is not available.'
                )
        else:
            pool = None

        # Set first, incase optimizer uses _max_iterations in initialization
        self.__max_iterations = max_iterations

        # Initialize algorithm
        self._reset()

        best_solution = {'solution': None, 'fitness': None}
        population = self.initial_population()
        try:
            # Begin optimization loop
            for self.iteration in range(1, self._max_iterations + 1):
                # Evaluate potential solutions
                solutions, fitnesses, finished = self._get_fitnesses(
                    problem,
                    population,
                    cache_encoded=cache_encoded,
                    cache_solution=cache_solution,
                    pool=pool)

                # If the best fitness from this iteration is better than
                # the global best
                best_index, best_fitness = max(
                    enumerate(fitnesses), key=operator.itemgetter(1))
                if best_fitness > best_solution['fitness']:
                    # Store the new best solution
                    best_solution['fitness'] = best_fitness
                    best_solution['solution'] = solutions[best_index]

                if logging_func:
                    logging_func(self.iteration, population, solutions,
                                 fitnesses, best_solution['solution'],
                                 best_solution['fitness'])

                if finished:
                    self.solution_found = True
                    break

                # Continue optimizing
                population = self.next_population(population, fitnesses)

            # Store best internally, before returning
            self.best_solution = best_solution['solution']
            self.best_fitness = best_solution['fitness']

        finally:
            # Clear caches
            if clear_cache:
                # Clear caches from memory
                self.__encoded_cache = {}
                self.__solution_cache = {}

                # Reset decoded cache key
                self._get_solution_key = self._get_solution_key_type

            # Clean up multiprocesses
            try:
                pool.terminate()  # Kill outstanding work
                pool.close()  # Close child processes
            except AttributeError:
                # No pool
                assert pool is None

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

    def _get_fitnesses(self,
                       problem,
                       population,
                       cache_encoded=True,
                       cache_solution=False,
                       pool=None):
        """Get the fitness for every solution in a population.

        Args:
            problem: Problem; The problem that defines fitness.
            population: list; List of potential solutions.
            pool: None/multiprocessing.Pool; Pool of processes for parallel
                decoding and evaluation.
        """
        fitnesses = [None] * len(population)

        #############################
        # Decoding
        #############################
        if cache_encoded:
            encoded_keys = map(tuple, population)

            # Get all fitnesses from encoded_solution cache
            to_decode_indices = []
            for i, encoded_key in enumerate(encoded_keys):
                try:
                    fitnesses[i] = self.__encoded_cache[encoded_key]
                    # Note that this fitness will never be better than the current best
                    # because we have already evaluted it,
                    # Therefore, we do not need to worry about decoding the solution
                except KeyError:  # Cache miss
                    to_decode_indices.append(i)
        else:
            encoded_keys = None
            to_decode_indices = range(len(population))

        # Decode all that need to be decoded, and combine back into list the same length
        # as population
        if encoded_keys is None:
            to_decode_keys = None
        else:
            to_decode_keys = [encoded_keys[i] for i in to_decode_indices]

        solutions = [None] * len(population)
        for i, solution in zip(to_decode_indices,
                               self._pmap(
                                   problem.decode_solution,
                                   [population[i] for i in to_decode_indices],
                                   to_decode_keys,
                                   pool)):
            solutions[i] = solution

        #############################
        # Evaluating
        #############################
        if cache_solution:
            try:
                # Try to make solutions hashable
                # Use user provided hash function if available
                if problem.hash_solution:
                    hash_solution_func = problem.hash_solution
                else:
                    # Otherwise, default to built in "smart" hash function
                    hash_solution_func = self._get_solution_key
                solution_keys = [
                    hash_solution_func(solution)
                    # None corresponds to encoded_solutions found in cache
                    if solution is not None else None for solution in solutions
                ]

                # Get all fitnesses from solution cache
                to_eval_indices = []
                for i, solution_key in enumerate(solution_keys):
                    if solution_key is not None:  # Otherwise, fitness already found in encoded cache
                        try:
                            fitnesses[i] = self.__solution_cache[solution_key]
                        except KeyError:  # Cache miss
                            to_eval_indices.append(i)

            except UnhashableError:  # Cannot hash solution
                solution_keys = None
                to_eval_indices = to_decode_indices[:]
        else:
            solution_keys = None
            to_eval_indices = to_decode_indices[:]

        # Evaluate all that need to be evaluated, and combine back into fitnesses list
        if solution_keys is None:
            if encoded_keys is None:
                # No way to detect duplicates
                to_eval_keys = None
            else:
                # Cannot use decoded keys, default to encoded keys
                to_eval_keys = [encoded_keys[i] for i in to_eval_indices]
        else:
            to_eval_keys = [solution_keys[i] for i in to_eval_indices]

        finished = False
        eval_bookkeeping = {}
        for i, fitness_finished in zip(to_eval_indices,
                                       self._pmap(
                                           problem.get_fitness,
                                           [solutions[i] for i in to_eval_indices],
                                           to_eval_keys,
                                           pool,
                                           bookkeeping_dict=eval_bookkeeping)):
            # Unpack fitness_finished tuple
            try:
                fitness, maybe_finished = fitness_finished
                if maybe_finished:
                    finished = True
            except TypeError:  # Not (fitness, finished) tuple
                fitness = fitness_finished

            fitnesses[i] = fitness

        #############################
        # Finishing
        #############################
        # Bookkeeping
        # keep track of how many times fitness is evaluated
        self.fitness_runs += len(eval_bookkeeping['key_indices'])  # Evaled once for each unique key

        # Add evaluated fitnesses to caches (both of them)
        if cache_encoded and encoded_keys is not None:
            for i in to_decode_indices:  # Encoded cache misses
                self.__encoded_cache[encoded_keys[i]] = fitnesses[i]
        if cache_solution and solution_keys is not None:
            for i in to_eval_indices:  # Decoded cache misses
                self.__solution_cache[solution_keys[i]] = fitnesses[i]

        # Return
        # assert None not in fitnesses  # Un-comment for debugging
        return solutions, fitnesses, finished

    def _pmap(self, func, items, keys, pool, bookkeeping_dict=None):
        """Efficiently map func over all items.

        Calls func only once for duplicate items.
            Item duplicates are detected by corresponding keys.
            Unless keys is None.

        Serial if pool is None, but still skips duplicates.
        """
        if keys is not None:  # Otherwise, cannot hash items
            # Remove duplicates first (use keys)
            # Create mapping (dict) of key to list of indices
            key_indices = _duplicates(keys).values()
        else:  # Cannot hash items
            # Assume no duplicates
            key_indices = [[i] for i in range(len(items))]

        # Use only the first of duplicate indices in decoding
        if pool is not None:
            # Parallel map
            results = pool.map(
                functools.partial(_unpickle_run, pickle.dumps(func)),
                [items[i[0]] for i in key_indices])
        else:
            results = map(func, [items[i[0]] for i in key_indices])

        # Add bookkeeping
        if bookkeeping_dict is not None:
            bookkeeping_dict['key_indices'] = key_indices

        # Combine duplicates back into list
        all_results = [None] * len(items)
        for indices, result in zip(key_indices, results):
            for j, i in enumerate(indices):
                # Avoid duplicate result objects in list,
                # in case they are used in functions with side effects
                if j > 0:
                    result = copy.deepcopy(result)
                all_results[i] = result

        return all_results

    def _get_solution_key_type(self, solution):
        # Start by just trying to hash it
        try:
            {solution: None}
            self._get_solution_key = self._get_solution_key_simple
        except:
            # Not hashable
            # Try tuple

            # Before trying tuple, check if dict
            # tuple(dict) will return a tuple of the KEYS only
            if isinstance(solution, dict):
                self._get_solution_key = self._get_solution_key_dict
            else:
                # Not dict, try tuple
                try:
                    {tuple(solution): None}
                    self._get_solution_key = self._get_solution_key_tuple
                except:
                    # Cannot convert to tuple
                    # Try str
                    try:
                        {str(solution): None}
                        self._get_solution_key = self._get_solution_key_str
                    except:
                        # Nothing works, give up
                        self._get_solution_key = self._get_solution_key_none

        # Done discovering, return key
        return self._get_solution_key(solution)

    def _get_solution_key_simple(self, solution):
        return solution

    def _get_solution_key_tuple(self, solution):
        return tuple(solution)

    def _get_solution_key_str(self, solution):
        return str(solution)

    def _get_solution_key_dict(self, solution):
        return tuple(solution.items())

    def _get_solution_key_none(self, solution):
        raise UnhashableError()

    def _set_hyperparameters(self, parameters):
        """Set internal optimization parameters."""
        for name, value in parameters.iteritems():
            try:
                getattr(self, name)
            except AttributeError:
                raise ValueError(
                    'Each parameter in parameters must be an attribute. '
                    '{} is not.'.format(name))
            setattr(self, name, value)

    def _get_hyperparameters(self):
        """Get internal optimization parameters."""
        hyperparameters = {}
        for key in self._hyperparameters:
            hyperparameters[key] = getattr(self, key)
        return hyperparameters

    def optimize_hyperparameters(self,
                                 problems,
                                 parameter_locks=None,
                                 smoothing=20,
                                 max_iterations=100,
                                 _meta_optimizer=None,
                                 _low_memory=True):
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
                    raise TypeError(
                        'problem must be Problem instance or list of Problem instances'
                    )
        elif isinstance(problems, Problem):
            problems = [problems]
        else:
            raise TypeError(
                'problem must be Problem instance or list of Problem instances'
            )

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
        decode = _make_hyperparameter_decode_func(locked_values,
                                                  meta_parameters)

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
        META_FITNESS = Problem(
            _meta_fitness_func,
            decode_function=decode,
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
        best_parameters = _meta_optimizer.optimize(
            META_FITNESS, max_iterations=max_iterations)

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
            'type': 'int',
            'min': 2,
            'max': 1026
        }


def _duplicates(list_):
    """Return dict mapping item -> indices."""
    item_indices = {}
    for i, item in enumerate(list_):
        try:
            item_indices[item].append(i)
        except KeyError:  # First time seen
            item_indices[item] = [i]
    return item_indices


def _unpickle_run(pickled_func, arg):
    """Run pickled_func on arg and return.

    Helper function for multiprocessing.
    """
    return pickle.loads(pickled_func)(arg)


#############################
# Hyperparamter optimization
#############################
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
            raise ValueError('Parameter type "{}" does not match known values'.
                             format(parameters['type']))

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
                value = helpers.binary_to_int(
                    binary,
                    offset=parameters['min'],
                    upper_bound=parameters['max'])
            elif parameters['type'] == 'float':
                value = helpers.binary_to_float(
                    binary,
                    minimum=parameters['min'],
                    maximum=parameters['max'])
            else:
                raise ValueError(
                    'Parameter type "{}" does not match known values'.format(
                        parameters['type']))

            # Store value
            hyperparameters[name] = value

        return hyperparameters

    return decode


def _meta_fitness_func(parameters,
                       _optimizer,
                       _problems,
                       _master_fitness_dict,
                       _runs=20):
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
