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

import random
import math

import numpy

from optimal import optimize

epsilon = 0.0000001

class GSA(optimize.StandardOptimizer):
    """Gravitational Search Algorithm
    
    Peform graviational search algorithm optimization with a given fitness function."""

    def __init__(self, fitness_function, solution_size, lower_bounds, upper_bounds, 
                 population_size=20, max_iterations=100, 
                 G_initial=1.0, G_reduction_rate=0.5,
                 **kwargs):
        """Create an object that optimizes a given fitness function with GSA.

        Args:
            fitness_function: A function representing the problem to solve, must return a fitness value.
            solution_size: The number of real values in each solution.
            lower_bounds: list, each value is a lower bound for the corrosponding part of the solution.
            upper_bounds: list, each value is a upper bound for the corrosponding part of the solution.
            population_size: The number of potential solutions in every generation
            max_iterations: The number of iterations to optimize before stopping
        """
        super(GSA, self).__init__(fitness_function, solution_size, population_size, 
                                  max_iterations, **kwargs)

        #set paramaters for users problem
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

        # GSA variables
        self._G_initial = G_initial
        self._G_reduction_rate = G_reduction_rate
        self._velocities = None
        self.initialize()

    def initialize(self):
        # Intialize GSA variables
        self._velocities = [[0.0]*self._solution_size]*self._population_size

    def create_initial_population(self):
        return _initial_population_gsa(self._population_size, self._solution_size,
                                         self._lower_bounds, self._upper_bounds)

    def new_population(self, population, fitnesses):
        new_pop, new_velocities = _new_population_gsa(population, fitnesses, self._velocities,
                                                 self._lower_bounds, self._upper_bounds,
                                                 self._G_initial, self._G_reduction_rate,
                                                 self.iteration, self._max_iterations)
        self._velocities = new_velocities
        return new_pop

def _initial_population_gsa(population_size, solution_size, 
                              lower_bounds, upper_bounds):
    """Create a random initial population of floating point values.

    Args:
        population_size: an integer representing the number of solutions in the population.
        problem_size: the number of values in each solution.
        lower_bounds: a list, each value is a lower bound for the corresponding part of the solution.
        upper_bounds: a list, each value is a upper bound for the corresponding part of the solution.

    Returns:
        list; A list of random solutions.
    """
    if len(lower_bounds) != solution_size or len(upper_bounds) != solution_size:
        raise ValueError("Lower and upper bounds much have a length equal to the problem size.")

    return optimize.make_population(population_size, optimize.random_solution_real,
                                    solution_size, lower_bounds, upper_bounds)

def _new_population_gsa(population, fitnesses, velocities,
                   lower_bounds, upper_bounds, 
                   G_initial, G_reduction_rate, iteration, max_iterations):
    # Update the gravitational constant, and the best and worst of the population
    # Calulate the mass and acceleration for each solution
    # Update the velocity and position of each solution
    population_size = len(population)
    solution_size = len(population[0])

    G = _G_gsa(G_initial, G_reduction_rate, iteration, max_iterations)
    masses = _get_masses(fitnesses)
        
    # Create bundled solution with position and mass for the K best calculation
    # Also store index to later check if two solutions are the same
    # Sorted by solution fitness (mass)
    solutions = [{'pos': pos, 'mass': mass, 'index': i} for i, (pos, mass) in enumerate(zip(population, masses))]
    solutions.sort(key = lambda x: x['mass'], reverse=True)

    # Get the force on each solution
    # Only the best K solutions apply force
    # K linearly decreases to 1
    K = int(population_size-(population_size-1)*(iteration/float(max_iterations)))
    forces = []
    for i in range(population_size):
        force_vectors = []
        for j in range(K):
            # If it is not the same solution
            if i != solutions[j]['index']:
                force_vectors.append(_gsa_force(G, masses[i], solutions[j]['mass'], 
                                                population[i], solutions[j]['pos']))
        forces.append(_gsa_total_force(force_vectors, solution_size))

    # Get the accelearation of each solution
    accelerations = []
    for i in range(population_size):
        accelerations.append(_gsa_acceleration(forces[i], masses[i]))

    # Update the velocity of each solution
    new_velocities = []
    for i in range(population_size):
        new_velocities.append(_gsa_update_velocity(velocities[i], accelerations[i]))

    # Create the new population
    new_population = []
    for i in range(population_size):
        new_position = _gsa_update_position(population[i], new_velocities[i])
        # Constrain to bounds
        new_position = list(numpy.clip(new_position, lower_bounds, upper_bounds))
        #for j in range(solution_size):
        #    if new_position[j] < lower_bounds[j]:
        #        new_position[j] = lower_bounds[j]
        #    if new_position[j] > upper_bounds[j]:
        #        new_position[j] = upper_bounds[j]

        new_population.append(new_position)

    return new_population, new_velocities

def _G_physics(G_initial, t, G_reduction_rate):
    return G_initial*(1.0/t)**G_reduction_rate

def _G_gsa(G_initial, G_reduction_rate, iteration, max_iterations):
    return G_initial*math.exp(-G_reduction_rate*iteration/float(max_iterations))

def _get_masses(fitnesses):
    # Obtain constants
    best_fitness = max(fitnesses)
    worst_fitness = min(fitnesses)
    fitness_range = best_fitness-worst_fitness

    # Calculate raw masses for each solution
    m_vec = []
    for fitness in fitnesses:
        # Epsilon is added to prevent divide by zero errors
        m_vec.append((fitness-worst_fitness)/(fitness_range+epsilon)+epsilon)

    # Normalize to obtain final mass for each solution
    total_m = sum(m_vec)
    M_vec = []
    for m in m_vec:
        M_vec.append(m/total_m)

    return M_vec

def _gsa_force(G, M_i, M_j, x_i, x_j):
    """Gives the force of solution j on solution i.
    
    args:
        G: The gravitational constant.
        M_i: The mass of solution i (derived from fitness).
        M_j: The mass of solution j (derived from fitness).
        x_i: The position of solution i.
        x_j: The position of solution j.

    returns:
        numpy.array; The force vector of solution j on solution i.
    """

    position_diff = numpy.subtract(x_j, x_i)
    distance = numpy.linalg.norm(position_diff)

    # The first 3 terms give the magnitude of the force
    # The last term is a vector that provides the direction
    # Epsilon prevents divide by zero errors
    return G*(M_i*M_j)/(distance+epsilon)*position_diff

def _gsa_total_force(force_vectors, vector_length):
    """Return a randomly weighted sum of the force vectors.
    
    args:
        force_vectors: A list of force vectors on solution i.

    returns:
        numpy.array; The total force on solution i.
    """
    if len(force_vectors) == 0:
        return [0.0]*vector_length
    # The GSA algorithm specifies that the total force in each dimension 
    # is a random sum of the individual forces in that dimension.
    # For this reason we sum the dimensions individually instead of simply using vec_a+vec_b
    total_force = [0.0]*vector_length
    for force_vec in force_vectors:
        for d in range(vector_length):
            total_force[d] += random.uniform(0.0, 1.0)*force_vec[d]
    return total_force

def _gsa_acceleration(total_force, M_i):
    return numpy.divide(total_force, M_i)

def _gsa_update_velocity(v_i, a_i):
    """Stochastically update the velocity of solution i."""

    # The GSA algorithm specifies that the new velocity for each dimension
    # is a sum of a random fraction of its current velocity in that dimension, 
    # and its acceleration in that dimension
    # For this reason we sum the dimensions individually instead of simply using vec_a+vec_b
    v = []
    for d in range(len(v_i)):
        v.append(random.uniform(0.0, 1.0)*v_i[d]+a_i[d])
    return v

def _gsa_update_position(x_i, v_i):
    return numpy.add(x_i, v_i)
