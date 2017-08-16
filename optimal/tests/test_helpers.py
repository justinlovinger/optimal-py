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


def test_binary_to_int_upper_bound():
    assert helpers.binary_to_int([0, 0], upper_bound=2) == 0
    assert helpers.binary_to_int([0, 1], upper_bound=2) == 1
    assert helpers.binary_to_int([1, 0], upper_bound=2) == 2
    assert helpers.binary_to_int([1, 1], upper_bound=2) == 2
    assert helpers.binary_to_int([1, 0, 0], upper_bound=2) == 1
    assert helpers.binary_to_int([1, 0, 1], upper_bound=2) == 0


def test_binary_to_int_upper_bound_and_offset():
    assert helpers.binary_to_int([0, 0], offset=-2, upper_bound=0) == -2
    assert helpers.binary_to_int([0, 1], offset=-2, upper_bound=0) == -1
    assert helpers.binary_to_int([1, 0], offset=-2, upper_bound=0) == 0
    assert helpers.binary_to_int([1, 1], offset=-2, upper_bound=0) == 0
    assert helpers.binary_to_int([1, 0, 0], offset=-2, upper_bound=0) == -1
    assert helpers.binary_to_int([1, 0, 1], offset=-2, upper_bound=0) == -2

    assert helpers.binary_to_int([0, 0], offset=2, upper_bound=4) == 2
    assert helpers.binary_to_int([0, 1], offset=2, upper_bound=4) == 3
    assert helpers.binary_to_int([1, 0], offset=2, upper_bound=4) == 4
    assert helpers.binary_to_int([1, 1], offset=2, upper_bound=4) == 4
    assert helpers.binary_to_int([1, 0, 0], offset=2, upper_bound=4) == 3
    assert helpers.binary_to_int([1, 0, 1], offset=2, upper_bound=4) == 2
