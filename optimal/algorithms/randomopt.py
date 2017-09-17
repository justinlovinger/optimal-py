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
"""Random optimizers.

Useful for benchmarking and testing optimizers.
An optimizer should do better than random.

These optimizers should not be used for real optimization.
"""


from optimal import optimize, common


class _RandomOptimizer(optimize.StandardOptimizer):
    """Optimizer generating random solutions."""

    def __init__(self,
                 solution_size,
                 population_size=20):
        """Create an object that optimizes a given fitness function with GenAlg.

        Args:
            solution_size: The number of bits in every solution.
            population_size: The number of solutions in every iteration.
        """
        super(_RandomOptimizer, self).__init__(solution_size, population_size)

    def _generate_solution(self):
        """Return a single random solution."""
        raise NotImplementedError()

    def initial_population(self):
        """Make the initial population before each optimization run.

        Returns:
            list; a list of solutions.
        """
        return common.make_population(self._population_size,
                                      self._generate_solution)

    def next_population(self, population, fitnesses):
        """Make a new population after each optimization iteration.

        Args:
            population: The population current population of solutions.
            fitnesses: The fitness associated with each solution in the population
        Returns:
            list; a list of solutions.
        """
        return common.make_population(self._population_size,
                                      self._generate_solution)


class RandomBinary(_RandomOptimizer):
    """Optimizer generating random bit strings."""

    def _generate_solution(self):
        """Return a single random solution."""
        return common.random_binary_solution(self._solution_size)


class RandomReal(_RandomOptimizer):
    """Optimizer generating random lists of real numbers."""

    def __init__(self,
                 solution_size,
                 lower_bounds,
                 upper_bounds,
                 population_size=20):
        """Create an object that optimizes a given fitness function with GenAlg.

        Args:
            solution_size: The number of bits in every solution.
            lower_bounds: list, each value is a lower bound for the corresponding
                          component of the solution.
            upper_bounds: list, each value is a upper bound for the corresponding
                          component of the solution.
            population_size: The number of solutions in every iteration.
        """
        super(RandomReal, self).__init__(solution_size, population_size)

        # Set parameters for users problem
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

    def _generate_solution(self):
        """Return a single random solution."""
        return common.random_real_solution(
            self._solution_size, self._lower_bounds, self._upper_bounds)
