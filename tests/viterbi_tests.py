# viterbi_tests.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from constants import STRATEGY_TYPES
from viterbi import viterbi_algo
import numpy as np

def test_viterbi_algorithm():
    """
    Tests that the Viterbi algorithm returns valid hand strength sequences and probabilities.
    Verifies that hand strengths are in valid range (0-4) and probabilities are non-negative.
    """
    test_cases = [
        ([4, 4, 4, 4], 'Maniac'),           # Always raises
        ([2, 2, 2, 2], 'Loose_Passive'),    # Always calls
        ([0, 0, 0, 0], 'Tight_Passive'),    # Always folds
        ([4, 4, 2, 2], 'Loose_Aggressive'), # Mix of raises and calls
        ([2, 2, 0, 0], 'Tight_Passive'),    # Mix of calls and folds
    ]

    for sequence, strategy in test_cases:
        hand_strengths, probability = viterbi_algo(strategy, sequence)
        
        # All hand strengths should be between 0 and 4
        assert all(0 <= hs <= 4 for hs in hand_strengths), \
            f"{strategy}: invalid hand strength found: {hand_strengths}"
        
        # Probabilities should be non-negative
        assert probability >= 0, \
            f"{strategy}: negative probability found: {probability}"
        
        # Should return one hand strength per action
        assert len(hand_strengths) == len(sequence), \
            f"{strategy}: expected {len(sequence)} strengths, got {len(hand_strengths)}"
        
        # Probability should be > 0 for valid sequences
        assert probability > 0, \
            f"{strategy}: zero probability for valid sequence"
        
        print(f"{strategy} with {sequence}: {hand_strengths} (prob={probability:.6f}) passed")

    
    print("\nAll Viterbi tests passed")

test_viterbi_algorithm()