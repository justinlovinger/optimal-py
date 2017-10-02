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


def binary_size(num_values):
    """Return the min number of bits to represet num_values."""
    return int(math.ceil(math.log(num_values, 2)))


def binary_to_float(binary_list, lower_bound, upper_bound):
    """Return a floating point number between lower and upper bounds, from binary.

    Args:
        binary_list: list<int>; List of 0s and 1s.
            The number of bits in this list determine the number of possible
            values between lower and upper bound.
            Increase the size of binary_list for more precise floating points.
        lower_bound: Minimum value for output, inclusive.
            A binary list of 0s will have this value.
        upper_bound: Maximum value for output, inclusive.
            A binary list of 1s will have this value.

    Returns:
        float; A floating point number.
    """
    # Edge case for empty binary_list
    if binary_list == []:
        # With 0 bits, only one value can be represented,
        # and we default to lower_bound
        return lower_bound


    # A little bit of math gets us a floating point
    # number between upper and lower bound
    # We look at the relative position of
    # the integer corresponding to our binary list
    # between the upper and lower bound,
    # and offset that by lower bound
    return ((
        # Range between lower and upper bound
        float(upper_bound - lower_bound)
        # Divided by the maximum possible integer
        / (2**len(binary_list) - 1)
        # Times the integer represented by the given binary
        * binary_to_int(binary_list))
            # Plus the lower bound
            + lower_bound)


def binary_to_int(binary_list, lower_bound=0, upper_bound=None):
    """Return the base 10 integer corresponding to a binary list.

   The maximum value is determined by the number of bits in binary_list,
   and upper_bound. The greater allowed by the two.

    Args:
        binary_list: list<int>; List of 0s and 1s.
        lower_bound: Minimum value for output, inclusive.
            A binary list of 0s will have this value.
        upper_bound: Maximum value for output, inclusive.
            If greater than this bound, we "bounce back".
            Ex. w/ upper_bound = 2: [0, 1, 2, 2, 1, 0]
            Ex.
                raw_integer = 11, upper_bound = 10, return = 10
                raw_integer = 12, upper_bound = 10, return = 9

    Returns:
        int; Integer value of the binary input.
    """
    # Edge case for empty binary_list
    if binary_list == []:
        # With 0 bits, only one value can be represented,
        # and we default to lower_bound
        return lower_bound
    else:
        # The builtin int construction can take a base argument,
        # but it requires a string,
        # so we convert our binary list to a string
        integer = int(''.join([str(bit) for bit in binary_list]), 2)

    # Trim if over upper_bound
    if (upper_bound is not None) and integer + lower_bound > upper_bound:
        # Bounce back. Ex. w/ upper_bound = 2: [0, 1, 2, 2, 1, 0]
        return upper_bound - (integer % (upper_bound - lower_bound + 1))
    else:
        # Not over upper_bound
        return integer + lower_bound


def avg(values):
    """Return the average of a set of values."""
    return sum(values) / len(values)
