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
