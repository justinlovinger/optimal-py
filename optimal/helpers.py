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
"""Useful functions for working with metaheuristics."""

import math


def binary_to_int(binary_list, offset=0, upper_bound=None):
    """Takes a list of binary values, returns a integer representation.

   The maximum value is determined by the number of bits in binary_list,
   and the offset.

    Args:
        Offset: The lowest value that can be return (if binary list is all 0s).

    Returns:
        int; Integer value of the binary input.
    """

    # Convert the binary to an integer
    # First convert the list of binary values into a string
    binary_string = ''.join([str(bit) for bit in binary_list])
    integer = int(binary_string,
                  2)  # Convert the base 2 string into an integer

    value = integer + offset

    # Trim if necessary
    if (upper_bound is not None) and value > upper_bound:
        # Bounce back. Ex. w/ upper_bound=2: [0, 1, 2, 2, 1, 0]
        return upper_bound - ((value - offset) % (upper_bound - offset + 1))

    return value


def binary_to_float(binary_list, minimum, maximum):
    """Takes a list of binary values, returns a float representation.

    Args:
        minimum: The lowest value that can be return (if binary list is all 0s).
        maximum: The highest value that can be returned (if binary list is all 1s).

    Returns:
        float; A floating point number.
    """
    # get the max value
    max_binary = 2**len(binary_list) - 1

    # convert the binary to an integer
    integer = binary_to_int(binary_list, 0)

    # convert the integer to a floating point
    floating_point = float(integer) / max_binary

    # scale the floating point from min to max
    scaled_floating_point = floating_point * maximum
    scaled_floating_point -= floating_point * minimum
    scaled_floating_point += minimum

    return scaled_floating_point


def avg(values):
    """Return the average of a set of values."""
    return sum(values) / len(values)


def binary_size(num_values):
    """Return the min number of bits to represet num_values."""
    return int(math.ceil(math.log(num_values, 2)))
