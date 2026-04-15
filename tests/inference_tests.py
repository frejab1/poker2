# test_forward_algorithm.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from constants import STRATEGY_TYPES
from forward_algorithm import forward_algo
from strategy_estimator import strategy_posterior
import numpy as np

# Observational sequences, strategy, and underlying hand strengths
test_cases = [
    ([4, 4, 4, 4] * 3, 'Maniac', [2, 3, 2, 1, 3, 4, 4, 4, 2, 2, 3, 2]), # Always raises
    ([2, 2, 2, 2] * 3, 'Loose_Passive', [2, 3, 2, 1, 3, 4, 4, 4, 2, 2, 3, 2]),  # Always calls
    ([4, 4, 4, 3, 2, 2, 2, 1], 'Loose_Aggressive', [4, 4, 4, 3, 1, 1, 1, 0,]),   # Calls unless the hand is strong, then bets or raises
    ([0, 0, 0, 0, 1, 1, 1, 1], 'Tight_Passive', [0, 0, 0, 0, 2, 2, 2, 2]),  # Folds unless the hand is strong, then calls
    ([0, 0, 0, 0, 4, 4, 4, 3], 'Tight_Aggressive', [0, 0, 0, 0, 4, 4, 4, 3]),   # Folds unless the hand is strong, then bets or raises
]

def test_forward():
    """
    Tests that the Forward Algorithm returns valid log likelihoods and correctly
    identifies strategy type on known action sequences.
    """
    print("Testing forward_algo:")
    
    for sequence, expected, _ in test_cases:
        log_likelihoods = forward_algo(sequence)

        # All log likelihoods should be <= 0 (can be zero for perfect fit)
        assert all(ll <= 0 for ll in log_likelihoods), \
            f"{expected}: positive log likelihood found: {log_likelihoods}"

        # Log likelihoods should differ between strategy types
        assert len(set(np.round(log_likelihoods, 10))) > 1, \
            f"{expected}: all log likelihoods are identical"

        # Correct strategy should have strictly highest likelihood
        predicted = STRATEGY_TYPES[np.argmax(log_likelihoods)]
        assert predicted == expected, \
            f"Expected {expected}, got {predicted}"

        print(f"  {expected} sequence: passed")


def test_bayes():
    """
    Tests that the Bayes' inference returns valid probabilities and correctly
    identifies strategy type on known action sequences.
    """
    print("\nTesting strategy_posterior:")
    
    for sequence, expected, hand_strengths in test_cases:
        probabilities_dict = strategy_posterior(sequence, hand_strengths)
        
        # Convert dict to list in STRATEGY_TYPES order for easier testing
        probabilities = [probabilities_dict[s] for s in STRATEGY_TYPES]

        # All probabilities should be >= 0
        assert all(prob >= 0 for prob in probabilities), \
            f"{expected}: negative probability found: {probabilities}"

        # Probabilities should sum to approximately 1
        assert abs(sum(probabilities) - 1.0) < 1e-6, \
            f"{expected}: probabilities sum to {sum(probabilities)}, expected 1.0"

        # Probabilities should differ between strategy types
        assert len(set(np.round(probabilities, 10))) > 1, \
            f"{expected}: all probabilities are identical"

        # Correct strategy should have strictly highest probability
        predicted = max(probabilities_dict, key=probabilities_dict.get)
        assert predicted == expected, \
            f"Expected {expected}, got {predicted}. Probabilities: {probabilities_dict}"

        print(f"  {expected} sequence: passed")


test_forward()
test_bayes()
print("\nAll inference tests passed")