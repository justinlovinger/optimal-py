from optimal import (problems, helpers, Problem, RandomBinary, RandomReal,
                     ExhaustiveBinary)
from optimal.algorithms import baseline


#########################
# Random Optimizers
#########################
def test_RandomBinary_sphere():
    optimizer = RandomBinary(32)
    optimizer.optimize(problems.sphere_binary, max_iterations=1000)
    assert optimizer.solution_found


def test_RandomReal_sphere():
    optimizer = RandomReal(2, [-1.0, -1.0], [1.0, 1.0])
    optimizer.optimize(problems.sphere_real, max_iterations=1000)
    assert optimizer.solution_found


########################
# ExhaustiveOptimizer
########################
def test_int_to_binary():
    assert baseline._int_to_binary(0) == [0]
    assert baseline._int_to_binary(1) == [1]
    assert baseline._int_to_binary(2) == [1, 0]
    assert baseline._int_to_binary(3) == [1, 1]


def test_int_to_binary_size_too_small():
    assert baseline._int_to_binary(0, size=3) == [0, 0, 0]
    assert baseline._int_to_binary(1, size=3) == [0, 0, 1]
    assert baseline._int_to_binary(2, size=3) == [0, 1, 0]
    assert baseline._int_to_binary(3, size=3) == [0, 1, 1]


def test_int_to_binary_size_too_large():
    assert baseline._int_to_binary(2, size=1) == [0]
    assert baseline._int_to_binary(3, size=1) == [1]


def test_ExhaustiveBinary_sphere():
    # Simplify problem a little
    def decode_small_binary(binary, min_, max_):
        x1 = helpers.binary_to_float(binary[0:6], min_, max_)
        x2 = helpers.binary_to_float(binary[6:12], min_, max_)
        return x1, x2

    optimizer = ExhaustiveBinary(12)
    optimizer.optimize(
        Problem(
            problems.sphere_function,
            decode_function=lambda x: decode_small_binary(x, -1, 1)),
        max_iterations=1000)
    assert optimizer.solution_found
