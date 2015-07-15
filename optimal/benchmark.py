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

# stats:
# stats['runs'] -> [{'stat_name' -> stat}, ]
# stats['mean'] -> mean(stats['runs'])
# stats['sd'] -> standard_deviation(stats['runs'])

def mean_of_runs(stats):
    num_runs = len(stats['runs'])

    mean = {}
    for key in stats['runs'][0]:
        mean[key] = sum(run[key] for run in stats['runs'])/float(num_runs)

    return mean

def sd_of_runs(stats, mean):
    num_runs = len(stats['runs'])

    sd = {}
    for key in stats['runs'][0]:
        sd[key] = math.sqrt(sum((run[key]-mean[key])**2 for run in stats['runs'])/float(num_runs))
    
    return sd

def add_mean_sd_to_stats(stats):
    mean = mean_of_runs(stats)
    sd = sd_of_runs(stats, mean) 

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
    stats = {}
    class_counts = {}
    for optimizer in optimizers:
        class_name = optimizer.__class__.__name__
        
        # Keep track of how many optimizers of each class
        # for better keys in stats dict
        try:
            class_counts[class_name] += 1
        except KeyError:
            class_counts[class_name] = 1

        # Foo 1, Foo 2, Bar 1, etc.
        key = '{} {}'.format(class_name, class_counts[class_name])

        print key + ': ',

        # Finally, get the actual stats
        stats[key] = benchmark(optimizer, runs)

        print

    return stats


if __name__ == '__main__':
    import pprint
    import examplefunctions
    from genalg import GenAlg
    from gsa import GSA

    my_genalg = GenAlg(examplefunctions.ackley, 32, 
                       decode_func=examplefunctions.ackley_binary)
    my_gsa = GSA(examplefunctions.ackley, 2, [-5.0]*2, [5.0]*2, 
                 decode_func=examplefunctions.ackley_real)

    stats = compare([my_genalg, my_gsa])
    pprint.pprint(stats)