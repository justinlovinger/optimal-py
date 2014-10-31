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

"""Genetic Algorithm operators for mutation, crossover, and selection."""

import random

def stochastic_selection(population, probabilities):
    """Create a list of parents with stochastic universal sampling."""

    pop_size = len(population)

    #create selection list (for stochastic universal sampling)
    selection_list = []
    selection_spacing = 1.0/(pop_size)
    selection_start = random.uniform(0.0, selection_spacing)
    for i in range(pop_size):
        selection_list.append(selection_start+selection_spacing*i)

    #select intermediate population according to selection list
    intermediate_population = []
    for selection in selection_list:
        for (i, probability) in enumerate(probabilities):
            if probability >= selection:
                intermediate_population.append(population[i])
                break
    random.shuffle(intermediate_population)

    return intermediate_population

def roulette_selection(population, probabilities):
    """Create a list of parents with roulette selection."""

    pop_size = len(population)

    intermediate_population = []
    for j in range(pop_size):
        selection = random.uniform(0.0, 1.0) #choose a random selection
        for (i, probability) in enumerate(probabilities): #iterate over probabilities list
            if probability >= selection: #first probability that is greater
                intermediate_population.append(population[i])
                break

    return intermediate_population


def one_point_crossover(parents):
    """Perform one point crossover on two parent chromosomes.

    Select a random position in the chromosome. Take genes to the left from one parent and the rest from the other parent.
    Ex. p1 = xxxxx, p2 = yyyyy, position = 2 (starting at 0), child = xxyyy
    """

    chromosome_length = len(parents[0])
    crossover_point = random.randint(1, chromosome_length-1) #the point that the chomosomes will be crossed at

    children = [[], []] #the two children that will be created

    for i in range(chromosome_length): #for every bit in the chromosome
        if i < crossover_point: #if the pointer is less than the crossover point
            children[0].append(parents[0][i]) #take from same parents, add to the children
            children[1].append(parents[1][i])
        else:
            children[1].append(parents[0][i]) #take from other parents, add to the children
            children[0].append(parents[1][i])

    return children

def uniform_crossover(parents):
    """Perform uniform crossover on two parent chromosomes.
    
    Randomly take genes from one parent or the other.
    Ex. p1 = xxxxx, p2 = yyyyy, child = xyxxy
    """
    chromosome_length = len(parents[0])

    children = [[], []]

    for i in range(chromosome_length): #for every bit in the chromosome
        selected_parent = random.randint(0, 1)

        children[0].append(parents[selected_parent][i]) #take from the selected parent, and add it to child 1
        children[1].append(parents[1-selected_parent][i]) #take from the other parent, and add it to child 2

    return children

def mutate(population, mutation_chance):
    """Mutate every chromosome in a population, list is modified in place.
    
    Muation occurs by randomly flipping bits (genes).
    """
    for i in range(len(population)): #for every chromosome in the population
        for j in range(len(population[i])): #for every bit in the chomosome
            mutation = random.uniform(0.0, 1.0)
            if mutation <= mutation_chance: #if mutation takes place
                population[i][j] = 1-population[i][j] #flip the bit