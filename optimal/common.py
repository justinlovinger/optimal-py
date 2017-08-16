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
"""Helpful functions for most metaheuristics."""

import random


def random_binary_solution(solution_size):
    """Make a list of random 0s and 1s."""
    return [random.randint(0, 1) for _ in range(solution_size)]


def random_real_solution(solution_size, lower_bounds, upper_bounds):
    """Make a list of random real numbers between lower and upper bounds."""
    return [
        random.uniform(lower_bounds[i], upper_bounds[i])
        for i in range(solution_size)
    ]


def make_population(population_size, solution_generator, *args, **kwargs):
    """Make a population with the supplied generator."""
    return [
        solution_generator(*args, **kwargs) for _ in range(population_size)
    ]
