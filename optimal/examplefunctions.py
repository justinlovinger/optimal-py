import helpers
import math

def ackley_binary(binary):
    # Helpful functions from helpers are used to convert binary to floats
    x1 = helpers.binary_to_float(binary[0:16], -5, 5)
    x2 = helpers.binary_to_float(binary[16:32], -5, 5)
    return x1, x2

def ackley_real(values):
    return values

# The first argument must always be a potential solution.
# Additional arguments can optionally come after
# The optimizer will feed additional arguements to this function
def ackley(solution, decode_func): 
    #Turn our chromosome of bits into floating point values
    x1, x2 = decode_func(solution)

    # Ackley's function
    # A common mathematical optimization problem
    output = -20*math.exp(-0.2*math.sqrt(0.5*(x1**2+x2**2)))-math.exp(0.5*(math.cos(2*math.pi*x1)+math.cos(2*math.pi*x2)))+20+math.e

    # You can prematurely stop the genetic algorithm by returning True 
    # as the second return value
    # Here, we consider the problem solved if the output is <= 0.01
    if output <= 0.01:
        finished = True
    else:
        finished = False

    # Because this function is trying to minimize the output, 
    # a smaller output has a greater fitness
    fitness = 1/output

    return fitness, finished