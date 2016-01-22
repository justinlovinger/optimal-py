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

from optimal import crossentropy, examplefunctions, optimize

@pytest.mark.parametrize('solution,pdf,expected', [
        ([1, 1, 1], [1.0, 1.0, 1.0], 1.0),
        ([1, 1, 1], [0.0, 0.0, 0.0], 0.0),
        ([0, 0, 0], [1.0, 1.0, 1.0], 0.0),
        ([0, 0, 0], [0.5, 0.5, 0.5], 0.5),
        ([1, 1, 1], [0.5, 0.5, 0.5], 0.5),
        ])
def test_chance(solution, pdf, expected):
    assert crossentropy.chance(solution, pdf) == expected

@pytest.mark.parametrize('values,q,expected', [
        ([0.0, 0.5, 1.0], 1, 0.5),
        ([0.0, 0.5, 1.0], 0, 1.0),
        ([0.0, 0.5, 1.0], 2, 0.0),
        ([1.0, 0.5, 0.0], 0, 1.0),
        ])
def test_quantile_cutoff(values, q, expected):
    assert crossentropy.get_quantile_cutoff(values, q) == expected

@pytest.mark.parametrize('num_values,q,expected', [
        (10, 1.0, 0),
        (10, 0.0, 9),
        (10, 0.5, 4)
        ])
def test_get_quantile_offset(num_values, q, expected):
    assert crossentropy.get_quantile_offset(num_values, q) == expected

def test_pdf_value():
    assert 0

def test_best_pdf():
    assert 0

def test_crossentropy_problems():
    # Attempt to solve various problems
    # Assert that the optimizer can find the solutions
    optimizer = crossentropy.CrossEntropy(examplefunctions.ackley, 32, max_iterations=1000,
                                          decode_func=examplefunctions.ackley_binary)
    optimizer._logging_func = lambda x, y, z : optimize._print_fitnesses(x, y, z, frequency=100)
    optimizer.optimize()
    assert optimizer.solution_found

    # TODO: test other functions

def test_metaoptimize_crossentropy():
    assert 0