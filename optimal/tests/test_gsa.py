from optimal import gsa, examplefunctions, optimize

def test_gsa_problems():
    # Attempt to solve various problems
    # Assert that the optimizer can find the solutions
    optimizer = gsa.GSA(examplefunctions.ackley, 2, [-5.0]*2, [5.0]*2, max_iterations=1000,
                        decode_func=examplefunctions.decode_real)
    optimizer._logging_func = lambda x, y, z : optimize._print_fitnesses(x, y, z, frequency=100)
    optimizer.optimize()
    assert optimizer.solution_found

    # TODO: test other functions

def test_metaoptimize_gsa():
    assert 0