import numpy as np

from wiki_pageviews.metrics import smape, wape


def test_smape_zero_convention_and_symmetry():
    assert smape(np.array([0.0]), np.array([0.0])) == 0.0
    assert smape(np.array([1.0, 2.0]), np.array([2.0, 1.0])) == smape(
        np.array([2.0, 1.0]), np.array([1.0, 2.0])
    )


def test_wape_perfect_forecast():
    assert wape(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0
