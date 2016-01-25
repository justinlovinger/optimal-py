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

# For this example, we'll just compare the one point and uniform crossover operators

# Perform a little hack to make sure the optimal library is visible from this script
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0,parentdir) 

# Normal imports
import copy
import pprint

import genalg
import gaoperators
import examplefunctions
import benchmark


functions = [
    {'func': examplefunctions.levis_function, 'decode': examplefunctions.levi_binary},
    {'func': examplefunctions.eggholder_function, 'decode': examplefunctions.egg_binary},
    {'func': examplefunctions.table_function, 'decode': examplefunctions.table_binary},
    {'func': examplefunctions.shaffer_function, 'decode': examplefunctions.shaffer_binary},
    {'func': examplefunctions.cross_function, 'decode': examplefunctions.cross_binary},
    ]


def benchmark_multi(optimizer):
    """Benchmark an optimizer configuration on multiple functions."""
    
    # First, create optimizer for each function, for use in compare
    optimizers = []
    for f in functions:
        # Set optimizer function
        # we can easily do this since all functions require the same solution size
        optimizer._fitness_function = f['func']
        optimizer._additional_parameters['decode_func'] = f['decode']
        
        # Make a copy, or we'll only have one optimizer repeated in our list
        optimizers.append(copy.deepcopy(optimizer))
        
    # Get our benchmark stats
    all_stats = benchmark.compare(optimizers, 100)
    return benchmark.aggregate(all_stats)
    

# Create the genetic algorithm configurations to compare
# In reality, we would also want to optimize other hyperparameters
ga_onepoint = genalg.GenAlg(None, 32, 
                        crossover_function=gaoperators.one_point_crossover)
ga_uniform = genalg.GenAlg(None, 32, 
                    crossover_function=gaoperators.uniform_crossover)

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
aggregate_stats = benchmark.aggregate({'One Point': onepoint_stats, 
                                       'Uniform': uniform_stats})
print
print 'Both'
pprint.pprint(aggregate_stats)