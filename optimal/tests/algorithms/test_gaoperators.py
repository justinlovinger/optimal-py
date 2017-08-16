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

from optimal import common
from optimal import gaoperators


##########################
# Selection
##########################
def test_diversity_metric_in_range():
    """Diversity value should be in [0.0, 1.0]."""
    solution_size = random.randint(1, 20)

    solution = common.random_binary_solution(solution_size)
    population = common.make_population(
        random.randint(0, 20), common.random_binary_solution, solution_size)

    assert 0.0 <= gaoperators._diversity_metric(solution, population) <= 1.0


def test_manhattan_distance_not_same_length():
    """Both vectors must be of the same length."""
    with pytest.raises(ValueError):
        gaoperators._manhattan_distance([1, 0], [1, 0, 1])


##########################
# Crossover
##########################
def test_one_point_crossover():
    # Only one possible random point
    assert gaoperators.one_point_crossover(([0, 1], [1, 0])) == ([0, 0],
                                                                 [1, 1])

    # Two possible crossover points
    assert (gaoperators.one_point_crossover(
        ([0, 0, 0], [1, 1, 1])) in [([0, 1, 1], [1, 0, 0]), ([0, 0, 1],
                                                             [1, 1, 0])])
