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
"""Simple fitness functions for demonstration and internal testing.

Don't rely on the functionality from this module, as it is subject to change.
"""

import math
import functools

from optimal import Problem, helpers


def decode_binary(binary, min_, max_):
    # Helpful functions from helpers are used to convert binary to floats
    x1 = helpers.binary_to_float(binary[0:16], min_, max_)
    x2 = helpers.binary_to_float(binary[16:32], min_, max_)
    return x1, x2


###########################
# Sphere
##########################
def sphere_function(solution):
    x1, x2 = solution

    output = x1**2 + x2**2

    return 1.0 - output, output <= 0.01


sphere_binary = Problem(sphere_function,
                        functools.partial(decode_binary, min_=-5.0, max_=5.0))
sphere_real = Problem(sphere_function)


###########################
# Ackley
##########################
def ackley_function(solution):
    # Turn our solution of bits into floating point values
    x1, x2 = solution

    # Ackley's function
    # A common mathematical optimization problem
    output = -20 * math.exp(-0.2 * math.sqrt(0.5 * (
        x1**2 + x2**2))) - math.exp(0.5 * (math.cos(
            2 * math.pi * x1) + math.cos(2 * math.pi * x2))) + 20 + math.e

    # You can prematurely stop the optimizer by returning True
    # as the second return value
    # Here, we consider the problem solved if the output is <= 0.01
    finished = output <= 0.01

    # Because this function is trying to minimize the output,
    # a smaller output has a greater fitness
    return 1.0 - output, finished


ackley_binary = Problem(ackley_function,
                        functools.partial(decode_binary, min_=-5.0, max_=5.0))
ackley_real = Problem(ackley_function)


######################
# Levi
######################
def levis_function(solution):
    x1, x2 = solution

    output = math.sin(3 * math.pi * x1)**2 + (x1 - 1)**2 * \
        (1 + math.sin(3 * math.pi * math.pi * x2)**2) + (x2 - 1)**2 * \
        (1 + math.sin(2 * math.pi * x2)**2)

    return 1.0 - output, output <= 0.02


levis_binary = Problem(levis_function,
                       functools.partial(decode_binary, min_=-5.0, max_=5.0))
levis_real = Problem(levis_function)


######################
# Eggholder
######################
def eggholder_function(solution):
    x, y = solution

    output = -(y + 47) * math.sin(math.sqrt(math.fabs(y + x / 2 + 47))) \
        - x * math.sin(math.sqrt(math.fabs(x - (y + 47))))

    return 1.0 - (output + 959.6407), output < -934.0  # solution == -959.6407


eggholder_binary = Problem(eggholder_function,
                           functools.partial(
                               decode_binary, min_=256.0, max_=512.0))
eggholder_real = Problem(eggholder_function)


######################
# Holder's Table
######################
def table_function(solution):
    x, y = solution

    output = -math.fabs(
        math.sin(x) * math.cos(y) *
        math.exp(math.fabs(1 - (math.sqrt(x * x + y * y)) / math.pi)))

    return 1.0 - (output + 19.2085), output < -19.200  # solution == -19.2085


table_binary = Problem(table_function,
                       functools.partial(decode_binary, min_=-10.0, max_=10.0))
table_real = Problem(table_function)


######################
# Shaffer N2
######################
def shaffer_function(solution):
    x, y = solution

    output = 0.5 + (math.sin(x * x - y * y)**2 - 0.5) / \
        (1 + 0.001 * (x * x + y * y))**2

    return 1.0 - output, output < 0.01


shaffer_binary = Problem(shaffer_function,
                         functools.partial(
                             decode_binary, min_=-25.0, max_=25.0))
shaffer_real = Problem(shaffer_function)


######################
# Cross Tray
######################
def cross_function(solution):
    x, y = solution

    output = -0.0001 * (math.fabs(
        math.sin(x) * math.sin(y) * math.exp(
            math.fabs(100 - math.sqrt(x * x + y * y) / math.pi))) + 1)**0.1

    return 1.0 - (output + 2.06261), output < -2.062  # solution == -2.06261


cross_binary = Problem(cross_function,
                       functools.partial(decode_binary, min_=-5.0, max_=5.0))
cross_real = Problem(cross_function)
