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
import pathos

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


def test_Optimizer_get_fitnesses_disabled_encoded_cache():
    """Fitnesses should correspond to solutions."""
    # Fitness function is weighted summation of bits
    solution_size = random.randint(1, 50)
    weights = numpy.random.random(solution_size)

    def fitness_func(solution):
        return weights.dot(solution)

    # Optimizer with disabled encoded cache
    optimizer = optimize.Optimizer()
    optimizer.cache_encoded_solution = False

    # Test Optimizer._get_fitnesses
    _check_get_fitnesses(
        fitness_func, lambda x: x, solution_size, optimizer=optimizer)

    # Check caches as expected
    assert optimizer._Optimizer__encoded_cache == {}
    assert optimizer._Optimizer__decoded_cache != {}


def test_Optimizer_get_fitnesses_disabled_decoded_cache():
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
        fitness_func, lambda x: x, solution_size, optimizer=optimizer)

    # Check caches as expected
    assert optimizer._Optimizer__encoded_cache != {}
    assert optimizer._Optimizer__decoded_cache == {}


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
                         n_processes=0):
    """Assert that return values of Optimizer._get_fitnesses are correct."""
    problem = Problem(fitness_func, decode_function=decode_func)

    if optimizer is None:
        optimizer = optimize.Optimizer()

    if n_processes > 0:
        pool = pathos.multiprocessing.Pool(processes=n_processes)
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
            problem, population, pool=pool)
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
    optimizer._get_fitnesses(problem, [[0, 0], [0, 1], [1, 0], [1, 1]])
    assert optimizer._Optimizer__encoded_cache == {
        (0, 0): 0,
        (0, 1): 0.5,
        (1, 0): 1.0,
        (1, 1): 1.5
    }


def test_Optimizer_decoded_cache_correct():
    """Should map the correct key to fitness."""
    optimizer = optimize.Optimizer()

    def fitness_func(solution):
        return solution[0] + 0.5 * solution[1]

    def decode_func(encoded_solution):
        return (-encoded_solution[0], -encoded_solution[1])

    problem = Problem(fitness_func, decode_function=decode_func)

    # Test cache
    optimizer._get_fitnesses(problem, [[0, 0], [0, 1], [1, 0], [1, 1]])
    assert optimizer._Optimizer__decoded_cache == {
        (0, 0): 0,
        (0, -1): -0.5,
        (-1, 0): -1.0,
        (-1, -1): -1.5
    }


def test_Optimizer_get_decoded_key():
    # Hashable
    optimizer = optimize.Optimizer()
    optimizer._get_decoded_key('1') == '1'

    # Dict
    # NOTE: This requires special treatment, otherwise,
    # tuple(dict) will return a tuple of the KEYS only
    optimizer = optimize.Optimizer()
    optimizer._get_decoded_key({'a': '1'}) == tuple([('a', '1')])

    # Tupleable
    optimizer = optimize.Optimizer()
    optimizer._get_decoded_key(['1']) == tuple(['1'])

    # Stringable
    optimizer = optimize.Optimizer()
    optimizer._get_decoded_key([['1']]) == str([['1']])


def test_Optimizer_cache_encoded_solution_false():
    """Should only cache encoded solutions if True."""
    # After calling Optimizer._get_fitnesses
    # __encoded_cache should be empty
    # __decoded_cache should not
    optimizer = optimize.Optimizer()
    optimizer.cache_encoded_solution = False
    assert optimizer.cache_decoded_solution is True

    # Get fitnesses
    population = common.make_population(
        random.randint(1, 20), common.random_binary_solution, 2)
    optimizer._get_fitnesses(SIMPLE_PROBLEM, population)

    # Assert caches as expected
    assert optimizer._Optimizer__encoded_cache == {}
    assert optimizer._Optimizer__decoded_cache != {}


def test_Optimizer_cache_decoded_solution_false():
    """Should only cache decoded solutions if True."""
    # After calling Optimizer._get_fitnesses
    # __encoded_cache should not be empty
    # __decoded_cache should be empty
    optimizer = optimize.Optimizer()
    assert optimizer.cache_encoded_solution is True
    optimizer.cache_decoded_solution = False

    # Get fitnesses
    population = common.make_population(
        random.randint(1, 20), common.random_binary_solution, 2)
    optimizer._get_fitnesses(SIMPLE_PROBLEM, population)

    # Assert caches as expected
    assert optimizer._Optimizer__encoded_cache != {}
    assert optimizer._Optimizer__decoded_cache == {}


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
