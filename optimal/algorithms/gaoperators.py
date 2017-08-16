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
"""Genetic Algorithm operators for mutation, crossover, and selection."""

import random
import operator

#################################
# Selection
#################################
# TODO: Support diversity factor for all selection algorithms


def tournament_selection(population,
                         fitnesses,
                         num_competitors=2,
                         diversity_weight=0.0):
    """Create a list of parents with tournament selection.

    Args:
        population: A list of solutions.
        fitnesses: A list of fitness values corresponding to solutions in population.
        num_competitors: Number of solutions to compare every round.
            Best solution among competitors is selected.
        diversity_weight: Weight of diversity metric.
            Determines how frequently diversity is used to select tournament winners.
            Note that fitness is given a weight of 1.0.
            diversity_weight == 1.0 gives equal weight to diversity and fitness.
    """
    # Optimization if diversity factor is disabled
    if diversity_weight <= 0.0:
        fitness_pop = zip(fitnesses,
                          population)  # Zip for easy fitness comparison

        # Get num_competitors random chromosomes, then add best to result,
        # by taking max fitness and getting chromosome from tuple.
        # Repeat until full.
        return [
            max(random.sample(fitness_pop, num_competitors))[1]
            for _ in range(len(population))
        ]
    else:
        indices = range(len(population))

        # Select tournament winners by either max fitness or diversity.
        # The metric to check is randomly selected, weighted by diversity_weight.
        # diversity_metric is calculated between the given solution,
        # and the list of all currently selected solutions.
        selected_solutions = []
        # Select as many solutions are there are in population
        for _ in range(len(population)):
            competitor_indices = random.sample(indices, num_competitors)

            # Select by either fitness or diversity,
            # Selected by weighted random selection
            # NOTE: We assume fitness has a weight of 1.0
            if random.uniform(0.0, 1.0) < (1.0 / (1.0 + diversity_weight)):
                # Fitness
                selected_solutions.append(
                    max(
                        zip([fitnesses[i] for i in competitor_indices],
                            [population[i] for i in competitor_indices]))[-1])
            else:
                # Diversity
                # Break ties by fitness
                selected_solutions.append(
                    max(
                        zip([
                            _diversity_metric(population[i], selected_solutions
                                              ) for i in competitor_indices
                        ], [fitnesses[i] for i in competitor_indices],
                            [population[i] for i in competitor_indices]))[-1])

        return selected_solutions


def stochastic_selection(population, fitnesses):
    """Create a list of parents with stochastic universal sampling."""
    pop_size = len(population)
    probabilities = _fitnesses_to_probabilities(fitnesses)

    # Create selection list (for stochastic universal sampling)
    selection_list = []
    selection_spacing = 1.0 / pop_size
    selection_start = random.uniform(0.0, selection_spacing)
    for i in range(pop_size):
        selection_list.append(selection_start + selection_spacing * i)

    # Select intermediate population according to selection list
    intermediate_population = []
    for selection in selection_list:
        for (i, probability) in enumerate(probabilities):
            if probability >= selection:
                intermediate_population.append(population[i])
                break
    random.shuffle(intermediate_population)

    return intermediate_population


def roulette_selection(population, fitnesses):
    """Create a list of parents with roulette selection."""
    probabilities = _fitnesses_to_probabilities(fitnesses)

    intermediate_population = []
    for _ in range(len(population)):
        # Choose a random individual
        selection = random.uniform(0.0, 1.0)
        # Iterate over probabilities list
        for i, probability in enumerate(probabilities):
            if probability >= selection:  # First probability that is greater
                intermediate_population.append(population[i])
                break

    return intermediate_population


def _rescale(vector):
    """Scale values in vector to the range [0, 1].

    Args:
        vector: A list of real values.
    """
    # Subtract min, making smallest value 0
    min_val = min(vector)
    vector = [v - min_val for v in vector]

    # Divide by max, making largest value 1
    max_val = float(max(vector))
    try:
        return [v / max_val for v in vector]
    except ZeroDivisionError:  # All values are the same
        return [1.0] * len(vector)


def _diversity_metric(solution, population):
    """Return diversity value for solution compared to given population.

    Metric is sum of distance between solution and each solution in population,
    normalized to [0.0, 1.0].
    """
    # Edge case for empty population
    # If there are no other solutions, the given solution has maximum diversity
    if population == []:
        return 1.0

    return (
        sum([_manhattan_distance(solution, other) for other in population])
        # Normalize (assuming each value in solution is in range [0.0, 1.0])
        # NOTE: len(solution) is maximum manhattan distance
        / (len(population) * len(solution)))


def _manhattan_distance(vec_a, vec_b):
    """Return manhattan distance between two lists of numbers."""
    if len(vec_a) != len(vec_b):
        raise ValueError('len(vec_a) must equal len(vec_b)')
    return sum(map(lambda a, b: abs(a - b), vec_a, vec_b))


def _fitnesses_to_probabilities(fitnesses):
    """Return a list of probabilities proportional to fitnesses."""
    # Do not allow negative fitness values
    min_fitness = min(fitnesses)
    if min_fitness < 0.0:
        # Make smallest fitness value 0
        fitnesses = map(lambda f: f - min_fitness, fitnesses)

    fitness_sum = sum(fitnesses)

    # Generate probabilities
    # Creates a list of increasing values.
    # The greater the gap between two values, the greater the probability.
    # Ex. [0.1, 0.23, 0.56, 1.0]
    prob_sum = 0.0
    probabilities = []
    for fitness in fitnesses:
        if fitness < 0:
            raise ValueError(
                "Fitness cannot be negative, fitness = {}.".format(fitness))
        prob_sum += (fitness / fitness_sum)
        probabilities.append(prob_sum)
    probabilities[-1] += 0.0001  # to compensate for rounding errors

    return probabilities


##############################
# Crossover
##############################
def one_point_crossover(parents):
    """Perform one point crossover on two parent chromosomes.

    Select a random position in the chromosome.
    Take genes to the left from one parent and the rest from the other parent.
    Ex. p1 = xxxxx, p2 = yyyyy, position = 2 (starting at 0), child = xxyyy
    """
    # The point that the chromosomes will be crossed at (see Ex. above)
    crossover_point = random.randint(1, len(parents[0]) - 1)

    return (_one_parent_crossover(parents[0], parents[1], crossover_point),
            _one_parent_crossover(parents[1], parents[0], crossover_point))


def _one_parent_crossover(parent_1, parent_2, crossover_point):
    return parent_1[:crossover_point] + parent_2[crossover_point:]


def uniform_crossover(parents):
    """Perform uniform crossover on two parent chromosomes.

    Randomly take genes from one parent or the other.
    Ex. p1 = xxxxx, p2 = yyyyy, child = xyxxy
    """
    chromosome_length = len(parents[0])

    children = [[], []]

    for i in range(chromosome_length):
        selected_parent = random.randint(0, 1)

        # Take from the selected parent, and add it to child 1
        # Take from the other parent, and add it to child 2
        children[0].append(parents[selected_parent][i])
        children[1].append(parents[1 - selected_parent][i])

    return children


#############################
# Mutation
#############################
def random_flip_mutate(population, mutation_chance):
    """Mutate every chromosome in a population, list is modified in place.

    Mutation occurs by randomly flipping bits (genes).
    """
    for chromosome in population:  # For every chromosome in the population
        for i in range(len(chromosome)):  # For every bit in the chromosome
            # If mutation takes place
            if random.uniform(0.0, 1.0) <= mutation_chance:
                chromosome[i] = 1 - chromosome[i]  # flip the bit
