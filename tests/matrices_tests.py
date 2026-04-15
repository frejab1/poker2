# matrices_tests.py
import sys
import os
import pandas as pd

# Allows Python to find methods in src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from constants import TRANSITION_MATRICES, EMISSION_MATRICES, STAT_DIST
import numpy as np

def test_transition_matrices():
    """
    Tests that each transition matrix has valid probability distributions.
    """
    for name, matrix in TRANSITION_MATRICES.items():
        assert np.all(matrix >= 0), f"{name}: negative entries found"
        assert np.allclose(matrix.sum(axis=1), 1), f"{name}: rows do not sum to one"
        print(f"Transition {name}: passed")

def test_emission_matrices():
    """
    Tests that each emission matrix has valid probability distributions.
    """
    for name, matrix in EMISSION_MATRICES.items():
        assert np.all(matrix >= 0), f"{name}: negative entries found"
        assert np.allclose(matrix.sum(axis=1), 1), f"{name}: rows do not sum to one"
        print(f"Emission {name}: passed")


def test_initial_distribution():
    """Tests that the initial distribution is valid and sums to one."""
    assert np.all(STAT_DIST >= 0), "Negative probabilities found"
    assert np.isclose(STAT_DIST.sum(), 1), f"Distribution sums to {STAT_DIST.sum()}, expected 1"
    print("Initial distribution tests passed.")


test_transition_matrices()
test_emission_matrices()
test_initial_distribution()