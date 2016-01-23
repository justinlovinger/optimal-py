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

import copy

import pytest

from optimal import optimize
from optimal.genalg import GenAlg

def simple_function(binary):
    finished = binary[0] and binary[1]
    return float(binary[0])+float(binary[1])+0.001, finished

def test_get_hyperparameters():
    optimizer = optimize.StandardOptimizer(simple_function, 2)

    hyperparameters = optimizer._get_hyperparameters()
    assert hyperparameters != None
    assert hyperparameters['_population_size']

def test_set_hyperparameters_wrong_parameter():
    optimizer = optimize.StandardOptimizer(simple_function, 2)

    with pytest.raises(ValueError):
        optimizer._set_hyperparameters({'test': None})

def test_meta_optimize_parameter_locks():
    # Run meta optimize with locks
    # assert that locked parameters did not change

    # Only optimize mutation chance
    parameter_locks=['_population_size', '_crossover_chance', '_selection_function', '_crossover_function']

    my_genalg = GenAlg(simple_function, 2)
    original = copy.deepcopy(my_genalg)

    # Low smoothing for faster performance
    my_genalg.optimize_hyperparameters(parameter_locks=parameter_locks, smoothing=1)

    # Check that mutation chance changed
    assert my_genalg._mutation_chance != original._mutation_chance

    # And all others stayed the same
    for parameter in parameter_locks:
        assert getattr(my_genalg, parameter) == getattr(original, parameter)