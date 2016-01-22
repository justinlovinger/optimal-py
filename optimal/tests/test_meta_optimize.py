# We'll use genalg for these tests
# Individual algorithms will have their own (less comprehensive) meta_optimize tests

from optimal.genalg import GenAlg
import copy

def simple_function(binary):
    finished = binary[0] and binary[1]
    return float(binary[0])+float(binary[1])+0.001, finished

def test_meta_optimize_parameter_locks():
    # Run meta optimize with locks
    # assert that locked parameters did not change

    # Only optimize mutation chance
    parameter_locks=['population_size', 'crossover_chance', 'selection_function', 'crossover_function']

    my_genalg = GenAlg(simple_function, 2)
    original = copy.deepcopy(my_genalg)

    # Low smoothing for faster performance
    my_genalg.optimize_hyperparameters(parameter_locks=parameter_locks, smoothing=1)

    # Check that mutation chance changed
    assert my_genalg.mutation_chance != original.mutation_chance

    # And all others stayed the same
    for parameter in parameter_locks:
        assert getattr(my_genalg, parameter) == getattr(original, parameter)