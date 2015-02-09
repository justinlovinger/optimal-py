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

import optimize
import random
import numpy

class GSA(optimize.Optimizer):
    """Peform genetic algorithm optimization with a given fitness function."""

    def __init__(self, fitness_function, solution_size, lower_bounds, upper_bounds, 
                 population_size=20, max_iterations=100,
                 **kwargs):
        """Create an object that performs genetic algorithm optimization with a given fitness function.

        Args:
            fitness_function: A function representing the problem to solve, must return a fitness value.
            solution_size: The number of values in each solution.
            lower_bounds: a list, each value is a lower bound for the corrosponding part of the solution.
            upper_bounds: a list, each value is a upper bound for the corrosponding part of the solution.
            population_size: The number of chromosomes in every generation
            max_iterations: The number of iterations to optimize before stopping
        """
        optimize.Optimizer.__init__(self, fitness_function, population_size, 
                                    max_iterations, **kwargs)

        #set paramaters for users problem
        self.solution_size = solution_size

        # GSA function parameters
        self.initial_pop_args = [self.solution_size]
        self.new_pop_args = []

        # GSA variables
        self.G_initial = 1.0
        self.G = self.G_initial
        self.velocities = [0.0]*solution_size

    def create_initial_population(self, population_size, problem_size, lower_bounds, upper_bounds):
        """Create a random initial population of floating point values.

        Args:
            population_size: an integer representing the number of solutions in the population.
            problem_size: the number of values in each solution.
            lower_bounds: a list, each value is a lower bound for the corrosponding part of the solution.
            upper_bounds: a list, each value is a upper bound for the corrosponding part of the solution.

        Returns:
            list; A list of random solutions.
        """
        if len(lower_bound) != problem_size or len(upper_bound) != problem_size:
            raise ValueError("Lower and upper bounds much have a length equal to the problem size.")

        # Intialize GSA variables
        self.G = self.G_initial
        self.velocities = [0.0]*solution_size

        # Create random population
        population = []

        for i in range(population_size): #for every chromosome
            solution = []
            for j in range(problem_size): #for every bit in the chromosome
                solution.append(random.uniform(lower_bounds[j], upper_bound[j])) #randomly add a 0 or a 1
            population.append(solution) #add the chromosome to the population

        return population

    def new_population(self, population, fitnesses):
        # Update the gravitational constant, and the best and worst of the population
        # Calulate the mass and acceleration for each solution
        # Update the velocity and position of each solution
        
        masses = get_masses(fitnesses)
        
        # Get the force on each solution
        forces = []
        for i in range(self.population_size):
            force_vectors = []
            for j in range(self.population_size):
                if i != j:
                    force_vectors.append(gsa_force(self.G, masses[i], masses[j], 
                                                   population[i], population[j]))
            forces.append(gsa_total_force(force_vectors))

        # Get the accelearation of each solution
        accelerations = []
        for i in range(self.population_size):
            accelerations.append(gsa_acceleration(forces[i], masses[i]))

        # Update the velocity of each solution
        for i in range(self.population_size):
            self.velocities[i] = gsa_update_velocity(self.velocities[i], accelerations[i])

        # Create the new population
        new_population = []
        for i in range(self.population_size):
            new_population.append(gsa_update_position(population[i], self.velocities[i]))

        return new_population


epsilon = 0.0000001
def gsa_force(G, M_i, M_j, x_i, x_j):
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

def gsa_total_force(force_vectors):
    """Return a randomly weighted sum of the force vectors.
    
    args:
        force_vectors: A list of force vectors on solution i.

    returns:
        numpy.array; The total force on solution i.
    """

    # The GSA algorithm specifies that the total force in each dimension 
    # is a random sum of the individual forces in that dimension.
    # For this reason we sum the dimensions individually instead of simply using vec_a+vec_b
    total_force = [0.0]*len(force_vectors[0])
    for force_vec in force_vectors:
        for d in range(force_vec):
            total_force[d] += random.uniform(0.0, 1.0)*force_vec[d]
    return numpy.array(total_force)

def gsa_acceleration(total_force, M_i):
    return total_force/M_i

def gsa_update_velocity(v_i, a_i):
    """Stochastically update the velocity of solution i."""

    # The GSA algorithm specifies that velocity is randomly weighted for each dimension.
    # For this reason we sum the dimensions individually instead of simply using vec_a+vec_b
    v = []
    for d in range(v_i):
        v.append(random.uniform(0.0, 1.0)*v_i[d]+a_i[d])
    return v

def gsa_update_position(x_i, v_i):
    return numpy.add(x_i+v_i)

def get_masses(fitnesses):
    # Obtain constants
    best_fitness = max(fitnesses)
    worst_fitness = min(fitnesses)
    fitness_range = best_fitness-worst_fitness

    # Calculate raw masses for each solution
    m_vec = []
    for fitness in fitnesses:
        m_vec.append((fitness-worst_fitness)/fitness_range)

    # Normalize to obtain final mass for each solution
    total_m = sum(m_vec)
    M_vec = []
    for m in m_vec:
        M_vec.append(m/total_m)

    return M_vec