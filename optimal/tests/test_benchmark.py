from optimal import optimize, problems, benchmark

class BadOptimizer(optimize.StandardOptimizer):
    def initial_population(self):
        return [[0]*self._solution_size for _ in range(self._population_size)]

    def next_population(self, *args):
        return self.initial_population()


def test_compare_iterations_integer():
    optimizers = [BadOptimizer(32) for _ in range(3)]
    benchmark.compare(optimizers, problems.sphere_binary, all_max_iterations=10)
    assert [optimizer.iteration for optimizer in optimizers] == [10, 10, 10]


def test_compare_iterations_list():
    optimizers = [BadOptimizer(32) for _ in range(3)]
    benchmark.compare(optimizers, problems.sphere_binary, all_max_iterations=[1, 10, 100])
    assert [optimizer.iteration for optimizer in optimizers] == [1, 10, 100]