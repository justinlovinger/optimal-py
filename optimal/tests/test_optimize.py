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

import copy
import random
import functools
import multiprocessing

import numpy

import pytest

from optimal import optimize, common, GenAlg, Problem


def simple_function(binary):
    return (float(binary[0]) + float(binary[1]) + 0.001), (binary[0]
                                                           and binary[1])


SIMPLE_PROBLEM = Problem(simple_function)


#########################
# Problem
#########################
def test_Problem_copy():
    problem = Problem(simple_function, fitness_args=['a'])
    problem_copy = problem.copy()
    assert problem_copy is not problem
    assert problem_copy.__dict__ == problem.__dict__

    problem_copy = problem.copy(fitness_args=['a', 'b'])
    assert problem_copy._fitness_args == ['a', 'b']
    assert problem._fitness_args == ['a']


###############################
# Optimizer
###############################
def test_Optimizer_optimize_parallel():
    optimzier = GenAlg(2)
    optimzier.optimize(SIMPLE_PROBLEM, n_processes=random.randint(2, 4))
    assert optimzier.solution_found


###############################
# Optimizer._get_fitnesses
###############################
def test_Optimizer_get_fitnesses_no_finished():
    """Fitnesses should correspond to solutions."""
    # Fitness function is weighted summation of bits
    solution_size = random.randint(1, 50)
    weights = numpy.random.random(solution_size)

    def fitness_func(solution):
        return weights.dot(solution)

    # Test Optimizer._get_fitnesses
    _check_get_fitnesses(fitness_func, lambda x: x, solution_size)


def test_Optimizer_get_fitnesses_correct_with_finished():
    """Fitnesses should correspond to solutions."""
    # Fitness function is weighted summation of bits
    solution_size = random.randint(1, 50)
    weights = numpy.random.random(solution_size)

    def fitness_func(solution):
        # Return tuple, with finished as second value
        return weights.dot(solution), False

    # Test Optimizer._get_fitnesses
    _check_get_fitnesses(
        fitness_func,
        lambda x: x,
        solution_size,
        fitness_func_returns_finished=True)


def test_Optimizer_get_fitnesses_with_fitness_func_side_effects():
    """Fitness function modifying solution should not affect fitnesses.

    This could potentially be a problem when there are duplicate solutions.
    """
    # Fitness function is weighted summation of bits
    solution_size = random.randint(1, 50)
    weights = numpy.random.random(solution_size)

    def fitness_func(solution):
        for i, val in enumerate(solution):
            solution[i] *= 2
        return weights.dot(solution)

    # Test Optimizer._get_fitnesses
    problem = Problem(fitness_func)

    optimizer = optimize.Optimizer()


    # Use simple map of fitness function over solutions as oracle
    # Repeat to test cache
    for _ in range(100):
        # Create a random population, and compare values returned by _get_fitness to simple maps
        population = common.make_population(
            random.randint(1, 20), common.random_binary_solution,
            solution_size)

        solutions, fitnesses, finished = optimizer._get_fitnesses(
            problem, copy.deepcopy(population), pool=None)

        assert fitnesses == map(fitness_func, population)

        assert finished is False


def test_Optimizer_get_fitnesses_with_decoder():
    """Fitnesses should correspond to solutions."""
    # Fitness function is weighted summation of bits
    solution_size = random.randint(1, 50)
    weights = numpy.random.random(solution_size)

    def fitness_func(solution):
        return weights.dot(solution)

    decode_weights = numpy.random.random(solution_size)

    def decode_func(encoded_solution):
        return list(decode_weights * encoded_solution)

    # Test Optimizer._get_fitnesses
    _check_get_fitnesses(fitness_func, decode_func, solution_size)


def test_Optimizer_get_fitnesses_unhashable_solution():
    """Should not fail when solution cannot be hashed or cached."""
    # Fitness function is weighted summation of bits
    solution_size = random.randint(1, 50)
    weights = numpy.random.random(solution_size)

    def fitness_func(solution):
        return weights.dot(solution.list)

    class ListWrapper(object):
        def __init__(self, list_):
            self.list = list_

        def __eq__(self, other):
            return type(self) == type(other) and self.list == other.list

        def __hash__(self):
            raise NotImplementedError()

        def __str__(self):
            raise NotImplementedError()

    decode_weights = numpy.random.random(solution_size)

    def decode_func(encoded_solution):
        return ListWrapper(list(decode_weights * encoded_solution))

    optimizer = optimize.Optimizer()

    # Test Optimizer._get_fitnesses
    _check_get_fitnesses(fitness_func, decode_func, solution_size, optimizer=optimizer, cache_solution=True)

    assert optimizer._Optimizer__solution_cache == {}


def test_Optimizer_get_fitnesses_cache_encoded_True_cache_solution_True():
    """Fitnesses should correspond to solutions."""
    # Fitness function is weighted summation of bits
    solution_size = random.randint(1, 50)
    weights = numpy.random.random(solution_size)

    def fitness_func(solution):
        return weights.dot(solution)

    # Optimizer with disabled encoded cache
    optimizer = optimize.Optimizer()

    # Test Optimizer._get_fitnesses
    _check_get_fitnesses(
        fitness_func,
        lambda x: x,
        solution_size,
        optimizer=optimizer,
        cache_encoded=True,
        cache_solution=True)

    # Check caches as expected
    assert optimizer._Optimizer__encoded_cache != {}
    assert optimizer._Optimizer__solution_cache != {}


def test_Optimizer_get_fitnesses_cache_encoded_False_cache_solution_True():
    """Fitnesses should correspond to solutions."""
    # Fitness function is weighted summation of bits
    solution_size = random.randint(1, 50)
    weights = numpy.random.random(solution_size)

    def fitness_func(solution):
        return weights.dot(solution)

    # Optimizer with disabled encoded cache
    optimizer = optimize.Optimizer()

    # Test Optimizer._get_fitnesses
    _check_get_fitnesses(
        fitness_func,
        lambda x: x,
        solution_size,
        optimizer=optimizer,
        cache_encoded=False,
        cache_solution=True)

    # Check caches as expected
    assert optimizer._Optimizer__encoded_cache == {}
    assert optimizer._Optimizer__solution_cache != {}


def test_Optimizer_get_fitnesses_cache_encoded_True_cache_solution_False():
    """Fitnesses should correspond to solutions."""
    # Fitness function is weighted summation of bits
    solution_size = random.randint(1, 50)
    weights = numpy.random.random(solution_size)

    def fitness_func(solution):
        return weights.dot(solution)

    # Optimizer with disabled encoded cache
    optimizer = optimize.Optimizer()
    optimizer.cache_decoded_solution = False

    # Test Optimizer._get_fitnesses
    _check_get_fitnesses(
        fitness_func,
        lambda x: x,
        solution_size,
        optimizer=optimizer,
        cache_encoded=True,
        cache_solution=False)

    # Check caches as expected
    assert optimizer._Optimizer__encoded_cache != {}
    assert optimizer._Optimizer__solution_cache == {}


def test_Optimizer_get_fitnesses_cache_encoded_False_cache_solution_False():
    """Fitnesses should correspond to solutions."""
    # Fitness function is weighted summation of bits
    solution_size = random.randint(1, 50)
    weights = numpy.random.random(solution_size)

    def fitness_func(solution):
        return weights.dot(solution)

    # Optimizer with disabled encoded cache
    optimizer = optimize.Optimizer()
    optimizer.cache_decoded_solution = False

    # Test Optimizer._get_fitnesses
    _check_get_fitnesses(
        fitness_func,
        lambda x: x,
        solution_size,
        optimizer=optimizer,
        cache_encoded=False,
        cache_solution=False)

    # Check caches as expected
    assert optimizer._Optimizer__encoded_cache == {}
    assert optimizer._Optimizer__solution_cache == {}


def test_Optimizer_get_fitnesses_with_pool():
    """Fitnesses should correspond to solutions."""
    # Fitness function is weighted summation of bits
    solution_size = random.randint(1, 50)
    weights = numpy.random.random(solution_size)

    def fitness_func(solution):
        return weights.dot(solution)

    # Test Optimizer._get_fitnesses
    _check_get_fitnesses(
        fitness_func,
        lambda x: x,
        solution_size,
        n_processes=random.randint(2, 4))


def _check_get_fitnesses(fitness_func,
                         decode_func,
                         solution_size,
                         fitness_func_returns_finished=False,
                         optimizer=None,
                         n_processes=0,
                         **kwargs):
    """Assert that return values of Optimizer._get_fitnesses are correct."""
    problem = Problem(fitness_func, decode_function=decode_func)

    if optimizer is None:
        optimizer = optimize.Optimizer()

    if n_processes > 0:
        pool = multiprocessing.Pool(processes=n_processes)
    else:
        pool = None

    # Use simple map of fitness function over solutions as oracle
    # Repeat to test cache
    for _ in range(100):
        # Create a random population, and compare values returned by _get_fitness to simple maps
        population = common.make_population(
            random.randint(1, 20), common.random_binary_solution,
            solution_size)

        solutions, fitnesses, finished = optimizer._get_fitnesses(
            problem, copy.deepcopy(population), pool=pool, **kwargs)
        # NOTE: _get_fitnesses will return None for solutions in cache, this is expected and ok
        assert False not in [
            solution == expected
            for solution, expected in zip(solutions,
                                          map(decode_func, population))
            if solution is not None
        ]

        if fitness_func_returns_finished is False:
            assert fitnesses == map(fitness_func, map(decode_func, population))
        else:
            # Need to strip finished from fitness_func return values
            assert fitnesses == [
                fitness_finished[0]
                for fitness_finished in map(fitness_func,
                                            map(decode_func, population))
            ]

        assert finished is False


###############################
# Caching
###############################
def test_Optimizer_encoded_cache_correct():
    """Should map the correct key to fitness."""
    optimizer = optimize.Optimizer()

    def fitness_func(solution):
        return solution[0] + 0.5 * solution[1]

    problem = Problem(fitness_func)

    # Test cache
    optimizer._get_fitnesses(
        problem, [[0, 0], [0, 1], [1, 0], [1, 1]], cache_encoded=True)
    assert optimizer._Optimizer__encoded_cache == {
        (0, 0): 0,
        (0, 1): 0.5,
        (1, 0): 1.0,
        (1, 1): 1.5
    }


def test_Optimizer_solution_cache_correct():
    """Should map the correct key to fitness."""
    optimizer = optimize.Optimizer()

    def fitness_func(solution):
        return solution[0] + 0.5 * solution[1]

    def decode_func(encoded_solution):
        return (-encoded_solution[0], -encoded_solution[1])

    problem = Problem(fitness_func, decode_function=decode_func)

    # Test cache
    optimizer._get_fitnesses(
        problem, [[0, 0], [0, 1], [1, 0], [1, 1]], cache_solution=True)
    assert optimizer._Optimizer__solution_cache == {
        (0, 0): 0,
        (0, -1): -0.5,
        (-1, 0): -1.0,
        (-1, -1): -1.5
    }


def test_Optimizer_get_solution_key():
    # Hashable
    optimizer = optimize.Optimizer()
    optimizer._get_solution_key('1') == '1'

    # Dict
    # NOTE: This requires special treatment, otherwise,
    # tuple(dict) will return a tuple of the KEYS only
    optimizer = optimize.Optimizer()
    optimizer._get_solution_key({'a': '1'}) == tuple([('a', '1')])

    # Tupleable
    optimizer = optimize.Optimizer()
    optimizer._get_solution_key(['1']) == tuple(['1'])

    # Stringable
    optimizer = optimize.Optimizer()
    optimizer._get_solution_key([['1']]) == str([['1']])


def test_Optimizer_optimize_cache_encoded_False_cache_solution_True():
    """Should only cache encoded solutions if True."""
    # After calling Optimizer._get_fitnesses
    # __encoded_cache should be empty
    # __solution_cache should not
    optimizer = GenAlg(2)

    # Optimize
    optimizer.optimize(
        SIMPLE_PROBLEM,
        max_iterations=1,
        cache_encoded=False,
        cache_solution=True,
        clear_cache=False)

    # Assert caches as expected
    assert optimizer._Optimizer__encoded_cache == {}
    assert optimizer._Optimizer__solution_cache != {}


def test_Optimizer_optimize_cache_encoded_True_cache_solution_False():
    """Should only cache decoded solutions if True."""
    # After calling Optimizer._get_fitnesses
    # __encoded_cache should not be empty
    # __solution_cache should be empty
    optimizer = GenAlg(2)

    # Get fitnesses
    optimizer.optimize(
        SIMPLE_PROBLEM,
        max_iterations=1,
        cache_encoded=True,
        cache_solution=False,
        clear_cache=False)

    # Assert caches as expected
    assert optimizer._Optimizer__encoded_cache != {}
    assert optimizer._Optimizer__solution_cache == {}


####################################
# Integration
####################################
def test_Optimizer_optimize_solution_correct():
    optimizer = GenAlg(2)
    assert optimizer.optimize(SIMPLE_PROBLEM) == [1, 1]


####################################
# Hyperparameters
####################################
def test_Optimizer_get_hyperparameters():
    optimizer = optimize.StandardOptimizer(2)

    hyperparameters = optimizer._get_hyperparameters()
    assert hyperparameters != None
    assert hyperparameters['_population_size']


def test_Optimizer_set_hyperparameters_wrong_parameter():
    optimizer = optimize.StandardOptimizer(2)

    with pytest.raises(ValueError):
        optimizer._set_hyperparameters({'test': None})


def test_Optimizer_meta_optimize_parameter_locks():
    # Run meta optimize with locks
    # assert that locked parameters did not change

    # Only optimize mutation chance
    parameter_locks = [
        '_population_size', '_crossover_chance', '_selection_function',
        '_crossover_function'
    ]

    my_genalg = GenAlg(2)
    original = copy.deepcopy(my_genalg)

    # Low smoothing for faster performance
    my_genalg.optimize_hyperparameters(
        SIMPLE_PROBLEM, parameter_locks=parameter_locks, smoothing=1)

    # Check that mutation chance changed
    assert my_genalg._mutation_chance != original._mutation_chance

    # And all others stayed the same
    for parameter in parameter_locks:
        assert getattr(my_genalg, parameter) == getattr(original, parameter)
