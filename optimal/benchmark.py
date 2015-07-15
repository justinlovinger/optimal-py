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

import math
import copy
import numbers

# stats:
# stats['runs'] -> [{'stat_name' -> stat}, ]
# stats['mean'] -> mean(stats['runs'])
# stats['sd'] -> standard_deviation(stats['runs'])

def mean_of_runs(stats, key='runs'):
    """Obtain the mean of stats.

    Args:
        stats: dict; A set of stats, structured as above.
        key: str; Optional key to determine where list of runs is found in stats
    """

    num_runs = len(stats[key])
    first = stats[key][0]

    mean = {}
    for stat_key in first:
        # Skip non numberic attributes
        if isinstance(first[stat_key], numbers.Number):
            mean[stat_key] = sum(run[stat_key] for run in stats[key])/float(num_runs)

    return mean

def sd_of_runs(stats, mean, key='runs'):
    """Obtain the standard deviation of stats.

    Args:
        stats: dict; A set of stats, structured as above.
        mean: dict; Mean for each key in stats.
        key: str; Optional key to determine where list of runs is found in stats
    """

    num_runs = len(stats[key])
    first = stats[key][0]

    sd = {}
    for stat_key in first:
        # Skip non numberic attributes
        if isinstance(first[stat_key], numbers.Number):
            sd[stat_key] = math.sqrt(sum((run[stat_key]-mean[stat_key])**2 for run in stats[key])/float(num_runs))
    
    return sd

def add_mean_sd_to_stats(stats, key='runs'):
    mean = mean_of_runs(stats, key)
    sd = sd_of_runs(stats, mean, key) 

    stats['mean'] = mean
    stats['sd'] = sd

def benchmark(optimizer, runs=20):
    """Run an optimizer through a problem multiple times."""
    stats = {'runs': []}

    # Disable logging, to avoid spamming the user
    logging = optimizer.logging
    optimizer.logging = False

    # Determine effectiveness of metaheuristic over many runs
    # The stochastic nature of metaheuristics make this necessary
    # for an accurate evaluation
    for i in range(runs):
        optimizer.optimize()

        # Convert bool to number for mean and standard deviation calculations
        if optimizer.solution_found:
            finished_num = 1.0
        else:
            finished_num = 0.0

        stats_ = {'fitness': optimizer.best_fitness, 
                  'evaluation_runs': optimizer.evaluation_runs,
                  'solution_found': finished_num}
        stats['runs'].append(stats_)

        # Little progress 'bar'
        print '.',

    # Mean gives a good overall idea of the metaheuristics effectiveness
    # Standard deviation (SD) shows consistency of performance
    add_mean_sd_to_stats(stats)

    # Bring back the users logging option for their optimizer
    optimizer.logging = logging

    return stats

def compare(optimizers, runs=20):
    """Compare a set of optimizers.

    Args:
        optimizers: list; A list of optimizers to compare.
        runs: int; How many times to run each optimizer (smoothness)
    """
    stats = {}
    key_counts = {}
    for optimizer in optimizers:
        # For nice human readable dictionaries, extract useful names from optimizer
        class_name = optimizer.__class__.__name__
        fitness_func_name = optimizer.fitness_function.__name__
        key_name = '{} {}'.format(class_name, fitness_func_name)
        
        # Keep track of how many optimizers of each class / fitness func
        # for better keys in stats dict
        try:
            key_counts[key_name] += 1
        except KeyError:
            key_counts[key_name] = 1

        # Foo 1, Foo 2, Bar 1, etc.
        key = '{} {}'.format(key_name, key_counts[key_name])

        print key + ': ',

        # Finally, get the actual stats
        stats[key] = benchmark(optimizer, runs)

        print

    return stats

def aggregate(all_stats):
    """Combine stats for multiple optimizers to obtain one mean and sd.

    Useful for combining stats for the same optimizer class and multiple problems.
    """
    aggregate_stats = {'means': [], 'sds': []}
    for optimizer_key in all_stats:
        # runs is the mean, for add_mean_sd function
        mean_stats = copy.deepcopy(all_stats[optimizer_key]['mean'])
        mean_stats['name'] = optimizer_key
        aggregate_stats['means'].append(mean_stats)

        # also keep track of standard deviations
        sd_stats = copy.deepcopy(all_stats[optimizer_key]['sd'])
        sd_stats['name'] = optimizer_key
        aggregate_stats['sds'].append(sd_stats)


    add_mean_sd_to_stats(aggregate_stats, 'means')

    return aggregate_stats

if __name__ == '__main__':
    import pprint
    import examplefunctions
    from genalg import GenAlg
    from gsa import GSA

    my_genalg = GenAlg(examplefunctions.ackley, 32, 
                       decode_func=examplefunctions.ackley_binary)
    my_gsa = GSA(examplefunctions.ackley, 2, [-5.0]*2, [5.0]*2, 
                 decode_func=examplefunctions.decode_real)

    stats = compare([my_genalg, my_gsa])
    pprint.pprint(stats)

    aggregate_stats = aggregate(stats)
    pprint.pprint(aggregate_stats)