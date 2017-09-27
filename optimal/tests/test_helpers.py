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

from optimal import helpers


##########################
# binary_size
##########################
def test_binary_size():
    assert helpers.binary_size(1) == 0
    assert helpers.binary_size(2) == 1
    assert helpers.binary_size(3) == 2
    assert helpers.binary_size(4) == 2
    assert helpers.binary_size(5) == 3
    assert helpers.binary_size(6) == 3
    assert helpers.binary_size(7) == 3
    assert helpers.binary_size(8) == 3
    assert helpers.binary_size(9) == 4


##########################
# binary_to_float
##########################
def test_binary_to_float_empty_binary_list():
    """An empty list can represent one value, and defaults to lower_bound."""
    assert helpers.binary_to_float([], 0.0, 1.0) == 0.0


def test_binary_to_float_zero_one_bounds():
    assert helpers.binary_to_float([0], 0.0, 1.0) == 0.0
    assert helpers.binary_to_float([1], 0.0, 1.0) == 1.0

    assert helpers.binary_to_float([0, 0], 0.0, 1.0) == 0.0
    assert _approx_equal(
        helpers.binary_to_float([0, 1], 0.0, 1.0), 0.3333333333)
    assert _approx_equal(
        helpers.binary_to_float([1, 0], 0.0, 1.0), 0.6666666666)
    assert helpers.binary_to_float([1, 1], 0.0, 1.0) == 1.0

    assert helpers.binary_to_float([0, 0, 0], 0.0, 1.0) == 0.0
    assert _approx_equal(
        helpers.binary_to_float([0, 0, 1], 0.0, 1.0), 0.14285714285714285)
    assert _approx_equal(
        helpers.binary_to_float([0, 1, 0], 0.0, 1.0), 0.2857142857142857)
    assert _approx_equal(
        helpers.binary_to_float([0, 1, 1], 0.0, 1.0), 0.42857142857142855)
    assert _approx_equal(
        helpers.binary_to_float([1, 0, 0], 0.0, 1.0), 0.5714285714285714)
    assert _approx_equal(
        helpers.binary_to_float([1, 0, 1], 0.0, 1.0), 0.7142857142857142)
    assert _approx_equal(
        helpers.binary_to_float([1, 1, 0], 0.0, 1.0), 0.8571428571428571)
    assert helpers.binary_to_float([1, 1, 1], 0.0, 1.0) == 1.0


def test_binary_to_float_neg_lower_bound():
    assert helpers.binary_to_float([0], -1.0, 1.0) == -1.0
    assert helpers.binary_to_float([1], -1.0, 1.0) == 1.0

    assert helpers.binary_to_float([0, 0], -1.0, 1.0) == -1.0
    assert _approx_equal(helpers.binary_to_float([0, 1], -1.0, 1.0), -0.3333333333)
    assert _approx_equal(helpers.binary_to_float([1, 0], -1.0, 1.0), 0.3333333333)
    assert helpers.binary_to_float([1, 1], -1.0, 1.0) == 1.0


def test_binary_to_float_neg_lower_and_upper_bound():
    assert helpers.binary_to_float([0], -2.0, -1.0) == -2.0
    assert helpers.binary_to_float([1], -2.0, -1.0) == -1.0

    assert helpers.binary_to_float([0, 0], -2.0, -1.0) == -2.0
    assert _approx_equal(helpers.binary_to_float([0, 1], -2.0, -1.0), -1.6666666666)
    assert _approx_equal(helpers.binary_to_float([1, 0], -2.0, -1.0), -1.3333333333)
    assert helpers.binary_to_float([1, 1], -2.0, -1.0) == -1.0


def _approx_equal(a, b, tol=1e-8):
    return abs(a - b) <= tol


##########################
# binary_to_int
##########################
def test_binary_to_int_empty_binary_list():
    """An empty list can represent one value, and defaults to lower_bound."""
    assert helpers.binary_to_int([]) == 0


def test_binary_to_int():
    assert helpers.binary_to_int([0]) == 0
    assert helpers.binary_to_int([0, 0]) == 0
    assert helpers.binary_to_int([0, 0, 0]) == 0

    assert helpers.binary_to_int([1]) == 1
    assert helpers.binary_to_int([0, 1]) == 1
    assert helpers.binary_to_int([0, 0, 1]) == 1

    assert helpers.binary_to_int([1, 0]) == 2
    assert helpers.binary_to_int([0, 1, 0]) == 2

    assert helpers.binary_to_int([1, 1]) == 3
    assert helpers.binary_to_int([0, 1, 1]) == 3

    assert helpers.binary_to_int([1, 0, 0]) == 4

    assert helpers.binary_to_int([1, 0, 1]) == 5

    assert helpers.binary_to_int([1, 1, 0]) == 6

    assert helpers.binary_to_int([1, 1, 1]) == 7


def test_binary_to_int_upper_bound():
    assert helpers.binary_to_int([0, 0], upper_bound=2) == 0
    assert helpers.binary_to_int([0, 1], upper_bound=2) == 1
    assert helpers.binary_to_int([1, 0], upper_bound=2) == 2
    assert helpers.binary_to_int([1, 1], upper_bound=2) == 2
    assert helpers.binary_to_int([1, 0, 0], upper_bound=2) == 1
    assert helpers.binary_to_int([1, 0, 1], upper_bound=2) == 0


def test_binary_to_int_lower_bound_and_upper_bound():
    assert helpers.binary_to_int([0, 0], lower_bound=-2, upper_bound=0) == -2
    assert helpers.binary_to_int([0, 1], lower_bound=-2, upper_bound=0) == -1
    assert helpers.binary_to_int([1, 0], lower_bound=-2, upper_bound=0) == 0
    assert helpers.binary_to_int([1, 1], lower_bound=-2, upper_bound=0) == 0
    assert helpers.binary_to_int([1, 0, 0], lower_bound=-2, upper_bound=0) == -1
    assert helpers.binary_to_int([1, 0, 1], lower_bound=-2, upper_bound=0) == -2

    assert helpers.binary_to_int([0, 0], lower_bound=2, upper_bound=4) == 2
    assert helpers.binary_to_int([0, 1], lower_bound=2, upper_bound=4) == 3
    assert helpers.binary_to_int([1, 0], lower_bound=2, upper_bound=4) == 4
    assert helpers.binary_to_int([1, 1], lower_bound=2, upper_bound=4) == 4
    assert helpers.binary_to_int([1, 0, 0], lower_bound=2, upper_bound=4) == 3
    assert helpers.binary_to_int([1, 0, 1], lower_bound=2, upper_bound=4) == 2
