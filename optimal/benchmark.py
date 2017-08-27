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
"""Functions for benchmarking and testing metaheuristics."""

import math
import copy
import numbers
import collections

# stats:
# stats['runs'] -> [{'stat_name' -> stat}, ]
# stats['mean'] -> mean(stats['runs'])
# stats['standard_deviation'] -> standard_deviation(stats['runs'])


def _mean_of_runs(stats, key='runs'):
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
            mean[stat_key] = sum(run[stat_key]
                                 for run in stats[key]) / float(num_runs)

    return mean


def _sd_of_runs(stats, mean, key='runs'):
    """Obtain the standard deviation of stats.

    Args:
        stats: dict; A set of stats, structured as above.
        mean: dict; Mean for each key in stats.
        key: str; Optional key to determine where list of runs is found in stats
    """

    num_runs = len(stats[key])
    first = stats[key][0]

    standard_deviation = {}
    for stat_key in first:
        # Skip non numberic attributes
        if isinstance(first[stat_key], numbers.Number):
            standard_deviation[stat_key] = math.sqrt(
                sum((run[stat_key] - mean[stat_key])**2
                    for run in stats[key]) / float(num_runs))

    return standard_deviation


def _add_mean_sd_to_stats(stats, key='runs'):
    mean = _mean_of_runs(stats, key)
    standard_deviation = _sd_of_runs(stats, mean, key)

    stats['mean'] = mean
    stats['standard_deviation'] = standard_deviation


def benchmark(optimizer, problem, runs=20, **kwargs):
    """Run an optimizer through a problem multiple times.

    Args:
        optimizer: Optimizer; The optimizer to benchmark.
        problem: Problem; The problem to benchmark on.
        runs: int > 0; Number of times that optimize is called on problem.

    Returns:
        dict; A dictionary of various statistics.
    """
    stats = {'runs': []}

    # Disable logging, to avoid spamming the user
    # TODO: Maybe we shouldn't disable by default?
    kwargs = copy.copy(kwargs)
    kwargs['logging_func'] = None

    # Determine effectiveness of metaheuristic over many runs
    # The stochastic nature of metaheuristics make this necessary
    # for an accurate evaluation
    for _ in range(runs):
        optimizer.optimize(problem, **kwargs)

        # Convert bool to number for mean and standard deviation calculations
        if optimizer.solution_found:
            finished_num = 1.0
        else:
            finished_num = 0.0

        stats_ = {
            'fitness': optimizer.best_fitness,
            'fitness_runs': optimizer.fitness_runs,
            'solution_found': finished_num
        }
        stats['runs'].append(stats_)

        # Little progress 'bar'
        print '.',

    # Mean gives a good overall idea of the metaheuristics effectiveness
    # Standard deviation (SD) shows consistency of performance
    _add_mean_sd_to_stats(stats)

    return stats


def compare(optimizers, problems, runs=20, all_kwargs={}):
    """Compare a set of optimizers.

    Args:
        optimizers: list/Optimizer; Either a list of optimizers to compare,
            or a single optimizer to test on each problem.
        problems: list/Problem; Either a problem instance or a list of problem instances,
            one for each optimizer.
        all_kwargs: dict/list<dict>; Either the Optimizer.optimize keyword arguments
            for all optimizers, or a list of keyword arguments, one for each optimizer.
        runs: int; How many times to run each optimizer (smoothness)

    Returns:
        dict; mapping optimizer identifier to stats.
    """
    if not (isinstance(optimizers, collections.Iterable)
            or isinstance(problems, collections.Iterable)):
        raise TypeError('optimizers or problems must be iterable')

    # If optimizers is not a list, repeat into list for each problem
    if not isinstance(optimizers, collections.Iterable):
        optimizers = [copy.deepcopy(optimizers) for _ in range(len(problems))]

    # If problems is not a list, repeat into list for each optimizer
    if not isinstance(problems, collections.Iterable):
        problems = [copy.deepcopy(problems) for _ in range(len(optimizers))]

    # If max_iterations is an integer, repeat it into a list
    if isinstance(all_kwargs, dict):
        all_kwargs = [all_kwargs] * len(optimizers)
    elif not isinstance(problems, collections.Iterable):
        raise TypeError('all_kwargs must be dict of list of dict')

    stats = {}
    key_counts = {}
    for optimizer, problem, kwargs in zip(optimizers, problems, all_kwargs):
        # For nice human readable dictionaries, extract useful names from
        # optimizer
        class_name = optimizer.__class__.__name__
        fitness_func_name = problem._fitness_function.__name__
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
        stats[key] = benchmark(optimizer, problem, runs=runs, **kwargs)

        print

    return stats


def aggregate(all_stats):
    """Combine stats for multiple optimizers to obtain one mean and sd.

    Useful for combining stats for the same optimizer class and multiple problems.

    Args:
        all_stats: dict; output from compare.
    """
    aggregate_stats = {'means': [], 'standard_deviations': []}
    for optimizer_key in all_stats:
        # runs is the mean, for add_mean_sd function
        mean_stats = copy.deepcopy(all_stats[optimizer_key]['mean'])
        mean_stats['name'] = optimizer_key
        aggregate_stats['means'].append(mean_stats)

        # also keep track of standard deviations
        sd_stats = copy.deepcopy(
            all_stats[optimizer_key]['standard_deviation'])
        sd_stats['name'] = optimizer_key
        aggregate_stats['standard_deviations'].append(sd_stats)

    _add_mean_sd_to_stats(aggregate_stats, 'means')

    return aggregate_stats
