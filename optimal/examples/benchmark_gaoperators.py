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
"""An example of benchmarking with optimal.

For this example, we'll compare the one point and uniform crossover operators

For reference only.
"""

# Add library to path
import sys, os
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import copy
import pprint

from optimal import GenAlg
from optimal import gaoperators
from optimal import problems
from optimal import benchmark

PROBLEMS = [
    problems.ackley_binary, problems.levis_binary, problems.eggholder_binary,
    problems.table_binary, problems.shaffer_binary, problems.cross_binary
]


def benchmark_multi(optimizer):
    """Benchmark an optimizer configuration on multiple functions."""
    # Get our benchmark stats
    all_stats = benchmark.compare(optimizer, PROBLEMS, runs=100)
    return benchmark.aggregate(all_stats)


# Create the genetic algorithm configurations to compare
# In reality, we would also want to optimize other hyperparameters
ga_onepoint = GenAlg(32, crossover_function=gaoperators.one_point_crossover)
ga_uniform = GenAlg(32, crossover_function=gaoperators.uniform_crossover)

# Run a benchmark for multiple problems, for robust testing
onepoint_stats = benchmark_multi(ga_onepoint)
uniform_stats = benchmark_multi(ga_uniform)

print
print 'One Point'
pprint.pprint(onepoint_stats)
print
print 'Uniform'
pprint.pprint(uniform_stats)

# We can obtain an easier comparison by performing another aggregate step
aggregate_stats = benchmark.aggregate({
    'One Point': onepoint_stats,
    'Uniform': uniform_stats
})
print
print 'Both'
pprint.pprint(aggregate_stats)
