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

import helpers
import math
import functools

def decode_binary(binary, min_, max_):
    # Helpful functions from helpers are used to convert binary to floats
    x1 = helpers.binary_to_float(binary[0:16], min_, max_)
    x2 = helpers.binary_to_float(binary[16:32], min_, max_)
    return x1, x2

def decode_real(values):
    return values

###########################
# Ackley
##########################
ACKLEY_MIN = -5.0
ACKLEY_MAX = 5.0

ackley_binary = functools.partial(decode_binary, min_=ACKLEY_MIN, max_=ACKLEY_MAX)

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
    finished = output <= 0.01

    # Because this function is trying to minimize the output, 
    # a smaller output has a greater fitness
    fitness = 1/output

    return fitness, finished

######################
# Levi
######################
LEVI_MIN = -5.0
LEVI_MAX = 5.0

levi_binary = functools.partial(decode_binary, min_=LEVI_MIN, max_=LEVI_MAX)

def levis_function(solution, decode_func):
    x1, x2 = decode_func(solution)

    output = math.sin(3*math.pi*x1)**2+(x1-1)**2*(1+math.sin(3*math.pi*math.pi*x2)**2)+(x2-1)**2*(1+math.sin(2*math.pi*x2)**2)
    finished = output <= 0.02

    return 1/output, finished

######################
# Eggholder
######################
EGG_MIN = 256.0
EGG_MAX = 512.0

egg_binary = functools.partial(decode_binary, min_=EGG_MIN, max_=EGG_MAX)

def eggholder_function(solution, decode_func):
    x, y = decode_func(solution)

    output = -(y+47)*math.sin(math.sqrt(math.fabs(y+x/2+47)))-x*math.sin(math.sqrt(math.fabs(x-(y+47))))
    finished = output < -934.0 # solution == -959.6407

    return 1/(output+959.6407), finished

######################
# Holder's Table
######################
TABLE_MIN = -10.0
TABLE_MAX = 10.0

table_binary = functools.partial(decode_binary, min_=TABLE_MIN, max_=TABLE_MAX)

def table_function(solution, decode_func):
    x, y = decode_func(solution)

    output = -math.fabs(math.sin(x)*math.cos(y)*math.exp(math.fabs(1-(math.sqrt(x*x+y*y))/math.pi)))
    finished = output < -19.200 # solution == -19.2085

    return 1/(output+19.2085), finished

######################
# Shaffer N2
######################
SHAFFER_MIN = -25.0
SHAFFER_MAX = 25.0

shaffer_binary = functools.partial(decode_binary, min_=SHAFFER_MIN, max_=SHAFFER_MAX)

def shaffer_function(solution, decode_func):
    x, y = decode_func(solution)

    output = 0.5+(math.sin(x*x-y*y)**2-0.5)/(1+0.001*(x*x+y*y))**2
    finished = output < 0.01

    return 1/output, finished

######################
# Cross Tray
######################
CROSS_MIN = -5.0
CROSS_MAX = 5.0

cross_binary = functools.partial(decode_binary, min_=CROSS_MIN, max_=CROSS_MAX)

def cross_function(solution, decode_func):
    x, y = decode_func(solution)

    output = -0.0001*(math.fabs(math.sin(x)*math.sin(y)*math.exp(math.fabs(100-math.sqrt(x*x+y*y)/math.pi)))+1)**0.1
    finished = output < -2.062 # solution == -2.06261

    return 1/(output+2.06261), finished