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

import pytest

from optimal import GSA, GenAlg, problems, optimize


def test_gsa_sphere():
    optimizer = GSA(2, [-5.0] * 2, [5.0] * 2)
    optimizer._logging_func = lambda x, y, z : optimize._print_fitnesses(x, y, z, frequency=100)
    optimizer.optimize(problems.sphere_real, max_iterations=1000)
    assert optimizer.solution_found


@pytest.mark.slowtest()
def test_gsa_problems():
    # Attempt to solve various problems
    # Assert that the optimizer can find the solutions
    optimizer = GSA(2, [-5.0] * 2, [5.0] * 2)
    optimizer._logging_func = lambda x, y, z : optimize._print_fitnesses(x, y, z, frequency=100)
    optimizer.optimize(problems.ackley_real, max_iterations=1000)
    assert optimizer.solution_found

    # TODO: test other functions


@pytest.mark.slowtest()
def test_metaoptimize_gsa():
    optimizer = GSA(2, [-5.0] * 2, [5.0] * 2)
    optimizer._logging_func = lambda x, y, z : optimize._print_fitnesses(x, y, z, frequency=100)
    prev_hyperparameters = optimizer._get_hyperparameters()

    # Test without metaoptimize, save iterations to solution
    optimizer.optimize(problems.sphere_real)
    iterations_to_solution = optimizer.iteration

    # Test with metaoptimize, assert that iterations to solution is lower
    optimizer.optimize_hyperparameters(
        problems.sphere_real,
        smoothing=1,
        max_iterations=1,
        _meta_optimizer=GenAlg(None, population_size=2))
    optimizer.optimize(problems.sphere_real)

    assert optimizer._get_hyperparameters() != prev_hyperparameters
    #assert optimizer.iteration < iterations_to_solution # Improvements are made