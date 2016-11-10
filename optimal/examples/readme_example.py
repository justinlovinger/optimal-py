import math

from optimal import GenAlg
from optimal import Problem
from optimal import helpers

# The genetic algorithm uses binary solutions.
# A decode function is useful for converting the binary solution to real numbers
def decode_ackley(binary):
    # Helpful functions from helpers are used to convert binary to float
    # x1 and x2 range from -5.0 to 5.0
    x1 = helpers.binary_to_float(binary[0:16], -5.0, 5.0)
    x2 = helpers.binary_to_float(binary[16:32], -5.0, 5.0)
    return x1, x2

# ackley is our fitness function
# This is how a user defines to goal of their problem
def ackley_fitness(solution):
    x1, x2 = solution

    # Ackley's function
    # A common mathematical optimization problem
    output = -20 * math.exp(-0.2 * math.sqrt(0.5 * (x1**2 + x2**2))) - math.exp(
        0.5 * (math.cos(2 * math.pi * x1) + math.cos(2 * math.pi * x2))) + 20 + math.e

    # You can prematurely stop the metaheuristic by returning True
    # as the second return value
    # Here, we consider the problem solved if the output is <= 0.01
    finished = output <= 0.01

    # Because this function is trying to minimize the output,
    # a smaller output has a greater fitness
    fitness = 1 / output

    # First return argument must be a real number
    # The higher the number, the better the solution
    # Second return argument is a boolean, and optional
    return fitness, finished

# Define a problem instance to optimize
# We can optionally include a decode function
# The optimizer will pass the decoded solution into your fitness function
# Additional fitness function and decode function parameters can also be added
ackley = Problem(ackley_fitness, decode_function=decode_ackley)

# Create a genetic algorithm with a chromosome size of 32,
# and use it to solve our problem
my_genalg = GenAlg(32)
best_solution = my_genalg.optimize(ackley)

print best_solution
