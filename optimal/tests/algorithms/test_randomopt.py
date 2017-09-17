from optimal import problems, RandomBinary, RandomReal


def test_RandomBinary_sphere():
    optimizer = RandomBinary(32)
    optimizer.optimize(problems.sphere_binary, max_iterations=1000)
    assert optimizer.solution_found


def test_RandomReal_sphere():
    optimizer = RandomReal(2, [-1.0, -1.0], [1.0, 1.0])
    optimizer.optimize(problems.sphere_real, max_iterations=1000)
    assert optimizer.solution_found
