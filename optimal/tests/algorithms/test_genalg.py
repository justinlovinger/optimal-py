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

import functools

import pytest

from optimal import Problem, GenAlg, problems, optimize, gaoperators


def very_simple_function(binary):
    return float(binary[0]) + 0.001, binary[0] == True


VERY_SIMPLE_PROBLEM = Problem(very_simple_function)


def test_genalg_chromosome_size_eq_1():
    """Regression test for chromosome_size == 1 edge case."""
    optimizer = GenAlg(1)
    optimizer.optimize(VERY_SIMPLE_PROBLEM)
    assert optimizer.solution_found


#############################
# Optimize
#############################
def test_genalg_sphere_defaults():
    _check_optimizer(GenAlg(32))


def test_genalg_sphere_tournament_no_diversity():
    _check_optimizer(
        GenAlg(
            32,
            selection_function=functools.partial(
                gaoperators.tournament_selection, diversity_weight=0.0)))


def test_genalg_sphere_tournament_with_diversity():
    _check_optimizer(
        GenAlg(
            32,
            selection_function=functools.partial(
                gaoperators.tournament_selection, diversity_weight=1.0)))


def test_genalg_sphere_roulette_selection():
    # Needs higher population size to consistently succeed
    _check_optimizer(
        GenAlg(
            32,
            population_size=40,
            selection_function=gaoperators.roulette_selection))


def test_genalg_sphere_stochastic_selection():
    # Needs higher population size to consistently succeed
    _check_optimizer(
        GenAlg(
            32,
            population_size=40,
            selection_function=gaoperators.stochastic_selection))


def _check_optimizer(optimizer):
    optimizer._logging_func = lambda x, y, z : optimize._print_fitnesses(x, y, z, frequency=100)
    optimizer.optimize(problems.sphere_binary)
    assert optimizer.solution_found


@pytest.mark.slowtest()
def test_genalg_problems():
    # Attempt to solve various problems
    # Assert that the optimizer can find the solutions
    optimizer = GenAlg(32)
    optimizer._logging_func = lambda x, y, z : optimize._print_fitnesses(x, y, z, frequency=100)
    optimizer.optimize(problems.ackley_binary)
    assert optimizer.solution_found

    # TODO: test other functions


##################################
# Metaoptimize
##################################
@pytest.mark.slowtest()
def test_metaoptimize_genalg():
    optimizer = GenAlg(32)
    optimizer._logging_func = lambda x, y, z : optimize._print_fitnesses(x, y, z, frequency=100)
    prev_hyperparameters = optimizer._get_hyperparameters()

    # Test without metaoptimize, save iterations to solution
    optimizer.optimize(problems.sphere_binary)
    iterations_to_solution = optimizer.iteration

    # Test with metaoptimize, assert that iterations to solution is lower
    optimizer.optimize_hyperparameters(
        problems.sphere_binary,
        smoothing=1,
        max_iterations=1,
        _meta_optimizer=GenAlg(None, population_size=2))
    optimizer.optimize(problems.sphere_binary)

    assert optimizer._get_hyperparameters() != prev_hyperparameters
    #assert optimizer.iteration < iterations_to_solution # Improvements are made
