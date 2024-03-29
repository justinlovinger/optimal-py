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

import random

import pytest
import numpy

from optimal.algorithms import gsa
from optimal import GSA, GenAlg, problems, optimize


def test_initial_gsa_population():
    population_size = random.randint(1, 20)
    solution_size = random.randint(1, 20)
    lower_bounds = numpy.array([random.uniform(-10, 10) for _ in range(solution_size)])
    upper_bounds = numpy.array([random.uniform(lb+1e-10, lb+10) for lb in lower_bounds])

    for lb, up in zip(lower_bounds, upper_bounds):
        assert lb < up

    initial_population = gsa._initial_gsa_population(population_size, solution_size, lower_bounds, upper_bounds)

    assert len(initial_population) == population_size
    for row in initial_population:
        assert len(row) == solution_size

        for i, v in enumerate(row):
            assert v >= lower_bounds[i] and v <= upper_bounds[i]


def test_gsa_sphere():
    optimizer = GSA(2, [-5.0] * 2, [5.0] * 2)
    optimizer.optimize(
        problems.sphere_real,
        max_iterations=1000,
        logging_func=
        lambda *args: optimize._print_fitnesses(*args, frequency=100))
    assert optimizer.solution_found


@pytest.mark.slowtest()
def test_gsa_problems():
    # Attempt to solve various problems
    # Assert that the optimizer can find the solutions
    optimizer = GSA(2, [-5.0] * 2, [5.0] * 2)
    optimizer.optimize(
        problems.ackley_real,
        max_iterations=1000,
        logging_func=
        lambda *args: optimize._print_fitnesses(*args, frequency=100))
    assert optimizer.solution_found

    # TODO: test other functions


@pytest.mark.slowtest()
def test_metaoptimize_gsa():
    optimizer = GSA(2, [-5.0] * 2, [5.0] * 2)
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
