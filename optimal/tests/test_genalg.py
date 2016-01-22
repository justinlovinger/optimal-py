from optimal import genalg, examplefunctions, optimize

def test_genalg_problems():
    # Attempt to solve various problems
    # Assert that the optimizer can find the solutions
    optimizer = genalg.GenAlg(examplefunctions.ackley, 32, 
                              decode_func=examplefunctions.ackley_binary)
    optimizer._logging_func = lambda x, y, z : optimize._print_fitnesses(x, y, z, frequency=100)
    optimizer.optimize()
    assert optimizer.solution_found

    # TODO: test other functions

def test_metaoptimize_genalg():
    optimizer = genalg.GenAlg(examplefunctions.ackley, 32, 
                              decode_func=examplefunctions.ackley_binary)
    optimizer._logging_func = lambda x, y, z : optimize._print_fitnesses(x, y, z, frequency=100)
    
    # Test without metaoptimize, save iterations to solution
    optimizer.optimize()
    iterations_to_solution = optimizer.iteration

    # Test with metaoptimize, assert that iterations to solution is lower
    optimizer.optimize_hyperparameters(max_iterations=10 ,smoothing=5)
    optimizer.optimize()
    assert optimizer.iteration < iterations_to_solution