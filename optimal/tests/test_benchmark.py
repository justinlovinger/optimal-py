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

from optimal import optimize, problems, benchmark


class BadOptimizer(optimize.StandardOptimizer):
    def initial_population(self):
        return [[0] * self._solution_size
                for _ in range(self._population_size)]

    def next_population(self, *args):
        return self.initial_population()


def test_compare_no_kwargs():
    optimizers = [BadOptimizer(32) for _ in range(3)]
    benchmark.compare(optimizers, problems.sphere_binary)


def test_compare_kwargs():
    optimizers = [BadOptimizer(32) for _ in range(3)]
    benchmark.compare(
        optimizers, problems.sphere_binary, all_kwargs={'max_iterations': 10})
    assert [optimizer.iteration for optimizer in optimizers] == [10, 10, 10]


def test_compare_kwargs_list():
    optimizers = [BadOptimizer(32) for _ in range(3)]
    benchmark.compare(
        optimizers,
        problems.sphere_binary,
        all_kwargs=[{
            'max_iterations': 1
        }, {
            'max_iterations': 10
        }, {
            'max_iterations': 100
        }])
    assert [optimizer.iteration for optimizer in optimizers] == [1, 10, 100]
