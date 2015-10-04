import pytest

import crossentropy

@pytest.mark.parametrize('solution,pdf,expected', [
        ([1, 1, 1], [1.0, 1.0, 1.0], 1.0),
        ([1, 1, 1], [0.0, 0.0, 0.0], 0.0),
        ([0, 0, 0], [1.0, 1.0, 1.0], 0.0),
        ([0, 0, 0], [0.5, 0.5, 0.5], 0.5),
        ([1, 1, 1], [0.5, 0.5, 0.5], 0.5),
        ])
def test_chance(solution, pdf, expected):
    assert crossentropy.chance(solution, pdf) == expected

@pytest.mark.parametrize('values,q,expected', [
        ([0.0, 0.5, 1.0], 1, 0.5),
        ([0.0, 0.5, 1.0], 0, 1.0),
        ([0.0, 0.5, 1.0], 2, 0.0),
        ([1.0, 0.5, 0.0], 0, 1.0),
        ])
def test_quantile_cutoff(values, q, expected):
    assert crossentropy.get_quantile_cutoff(values, q) == expected

@pytest.mark.parametrize('num_values,q,expected', [
        (10, 1.0, 0),
        (10, 0.0, 9),
        (10, 0.5, 4)
        ])
def test_get_quantile_offset(num_values, q, expected):
    assert crossentropy.get_quantile_offset(num_values, q) == expected

def test_pdf_value():
    assert 0

def test_best_pdf():
    assert 0