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
"""Gravitational search algorithm"""

import random
import math

import numpy

from optimal import optimize, common

EPSILON = 1e-10


# TODO: Optimize to use numpy array operations wherever possible
class GSA(optimize.StandardOptimizer):
    """Gravitational Search Algorithm

    Perform gravitational search algorithm optimization with a given fitness function.
    """

    def __init__(self,
                 solution_size,
                 lower_bounds,
                 upper_bounds,
                 population_size=20,
                 grav_initial=1.0,
                 grav_reduction_rate=0.5):
        """Create an object that optimizes a given fitness function with GSA.

        Args:
            solution_size: The number of real values in each solution.
            lower_bounds: list, each value is a lower bound for the corresponding
                          component of the solution.
            upper_bounds: list, each value is a upper bound for the corresponding
                          component of the solution.
            population_size: The number of potential solutions in every generation
            grav_initial: Initial value for grav parameter (0 - 1)
            grav_reduction_rate: Rate that grav parameter decreases over time (0 - 1)
        """
        super(GSA, self).__init__(solution_size, population_size)

        # set parameters for users problem
        self._lower_bounds = numpy.array(lower_bounds)
        self._upper_bounds = numpy.array(upper_bounds)

        # GSA variables
        self._grav_initial = grav_initial  # G_i in GSA paper
        self._grav_reduction_rate = grav_reduction_rate
        self._velocity_matrix = None
        self.initialize()

        # Hyperparameter definitions
        self._hyperparameters['_grav_initial'] = {
            'type': 'float',
            'min': 0.0,
            'max': 1.0
        }
        self._hyperparameters['_grav_reduction_rate'] = {
            'type': 'float',
            'min': 0.0,
            'max': 1.0
        }

    def initialize(self):
        # Initialize GSA variables
        self._velocity_matrix = numpy.zeros((self._population_size,
                                             self._solution_size))

    def initial_population(self):
        return _initial_gsa_population(self._population_size,
                                       self._solution_size, self._lower_bounds,
                                       self._upper_bounds)

    def next_population(self, population, fitnesses):
        new_pop, self._velocity_matrix = _new_population_gsa(
            population, fitnesses, self._velocity_matrix, self._lower_bounds,
            self._upper_bounds, self._grav_initial, self._grav_reduction_rate,
            self.iteration, self._max_iterations)
        return new_pop


def _initial_gsa_population(population_size, solution_size, lower_bounds,
                            upper_bounds):
    """Create a random initial population of floating point values.

    Args:
        population_size: an integer representing the number of solutions in the population.
        problem_size: the number of values in each solution.
        lower_bounds: array; each value is a lower bound for the corresponding
                      part of the solution.
        upper_bounds: array; each value is a upper bound for the corresponding
                      part of the solution.

    Returns:
        list; A list of random solutions.
    """
    if len(lower_bounds) != solution_size or len(upper_bounds) != solution_size:
        raise ValueError(
            "Lower and upper bounds much have a length equal to the problem size."
        )


    # population_size rows
    # solution_size columns
    # Each column in range of corresponding lower and upper bounds
    return numpy.random.uniform(lower_bounds, upper_bounds, (population_size,
                                                             solution_size))


def _new_population_gsa(population, fitnesses, velocity_matrix, lower_bounds,
                        upper_bounds, grav_initial, grav_reduction_rate,
                        iteration, max_iterations):
    """Generate a new population as given by GSA algorithm.

    In GSA paper, grav_initial is G_0
    """
    # Make sure population is a numpy array
    if not isinstance(population, numpy.ndarray):
        population = numpy.array(population)

    # Update the gravitational constant, and the best and worst of the population
    # Calculate the mass and acceleration for each solution
    # Update the velocity and position of each solution
    population_size = population.shape[0]
    solution_size = population.shape[1]

    # In GSA paper, grav is G
    grav = _next_grav(grav_initial, grav_reduction_rate, iteration,
                          max_iterations)
    mass_vector = _get_masses(fitnesses)

    # Get the force on each solution
    # Only the best K solutions apply force
    # K linearly decreases to 1
    num_best = int(population_size - (population_size - 1) *
                   (iteration / float(max_iterations)))

    force_matrix = _get_force_matrix(grav, population, mass_vector, num_best)


    # Get the acceleration of each solution
    # By dividing each force vector by corresponding mass
    acceleration_matrix = force_matrix / mass_vector.reshape(force_matrix.shape[0], 1)

    # Update the velocity of each solution
    # The GSA algorithm specifies that the new velocity for each dimension
    # is a sum of a random fraction of its current velocity in that dimension,
    # and its acceleration in that dimension
    new_velocity_matrix = numpy.random.random(
        velocity_matrix.shape) * velocity_matrix + acceleration_matrix

    # Create the new population
    new_population = numpy.clip(
        # Move each position by its velocity vector
        population + new_velocity_matrix,
        # Clip to constrain to bounds
        lower_bounds, upper_bounds)

    return new_population, new_velocity_matrix


def _next_grav(grav_initial, grav_reduction_rate, iteration, max_iterations):
    """Calculate G as given by GSA algorithm.

    In GSA paper, grav is G
    """
    return grav_initial * math.exp(
        -grav_reduction_rate * iteration / float(max_iterations))


def _get_masses(fitnesses):
    """Convert fitnesses into masses, as given by GSA algorithm."""
    # Make sure fitnesses is a numpy array
    if not isinstance(fitnesses, numpy.ndarray):
        fitnesses = numpy.array(fitnesses)

    # Obtain constants
    best_fitness = numpy.max(fitnesses)
    worst_fitness = numpy.min(fitnesses)
    fitness_range = best_fitness - worst_fitness

    # Calculate raw masses for each solution
    # By scaling each fitness to a positive value
    masses = (fitnesses - worst_fitness) / (fitness_range + EPSILON) + EPSILON

    # Normalize to a sum of 1 to obtain final mass for each solution
    masses /= numpy.sum(masses)

    return masses


def _get_force_matrix(grav, position_matrix, mass_vector, num_best):
    """Gives the force of solution j on solution i.

    num_rows(position_matrix) == num_elements(mass_vector)
    Each element in mass_vector corresponds to a row in position_matrix.

    args:
        grav: The gravitational constant. (G)
        position_matrix: Each row is a parameter vector,
            a.k.a. the position of a body body.
        masses: Each element is the mass of corresponding
            parameter vector (row) in position_matrix
        num_best: How many bodies to apply their force
            to each other body

    returns:
        numpy.array; Matrix of total force on each body.
            Each row is a force vector corresponding
            to corresponding row / body in position_matrix.
    """
    # TODO: Refactor to avoid calculating per position vector

    # Get index of num_best highest masses (corresponds to rows in population)
    k_best_indices = numpy.argpartition(mass_vector, -num_best)[-num_best:]

    # The GSA algorithm specifies that the total force in each dimension
    # is a random sum of the individual forces in that dimension.
    force_matrix = []
    for mass, position_vector in zip(mass_vector, position_matrix):
        # NOTE: We can ignore position_vector being in k_best because
        # difference will just be 0 vector
        diff_matrix = position_matrix[k_best_indices] - position_vector
        force_matrix.append(
            # Scale result by gravity constant
            grav *
            # Add force vector applied to this body by each other body
            numpy.sum(
                # Multiply each force scalar by a random number
                # in range [0, 1)
                numpy.random.random(diff_matrix.shape) *
                # Multiply each position difference vector by
                # product of corresponding masses
                ((mass_vector[k_best_indices] * mass) / (
                    # divided by distance
                    numpy.linalg.norm(diff_matrix, ord=2) + EPSILON
                )).reshape(diff_matrix.shape[0], 1) *
                # All multiplied by matrix of position difference vectors,
                # giving direction of force vectors
                diff_matrix,
                # Sum along each force vector
                axis=0))
    force_matrix = numpy.array(force_matrix)

    return force_matrix
