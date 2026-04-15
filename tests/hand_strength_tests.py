# hand_strength_tests.py
import sys
import os
import pandas as pd
import numpy as np

# Allows Python to find methods in src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from collections import Counter
from generate_data import *
import random

def evaluate_estimator_accuracy(filename, n_board, n_hands=100, n_trials=100):
    """
    Evaluate hand strength estimator accuracy for a given street.
    
    Args:
        filename (str): Path to output TSV file
        n_board (int): Number of community cards (0=preflop, 3=flop, 4=turn, 5=river)
        n_hands (int): Number of random hands to test
        n_trials (int): Number of Monte Carlo trials per hand
    
    Creates:
        TSV file with columns: Hand, VeryWeak, Weak, Moderate, Strong, VeryStrong
        Each row contains proportions of how often the hand was classified in each category.
    """
    with open(filename, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')

        # Header
        writer.writerow(['Hand', 'VeryWeak', 'Weak', 'Moderate', 'Strong', 'VeryStrong'])

        for _ in range(n_hands):
            # Sample random hand and board
            total_cards = random.sample(DECK, 2 + n_board)
            hand = total_cards[:2]
            board = total_cards[2:]

            results = []
            for _ in range(n_trials):
                results.append(hand_strength(hand, board))

            counts = Counter(results)

            # Convert to proportions in required order
            row = [
                f"{hand[0]},{hand[1]}",
                (counts[0] / n_trials),  # Very Weak
                (counts[1] / n_trials),  # Weak
                (counts[2] / n_trials),  # Moderate
                (counts[3] / n_trials),  # Strong
                (counts[4] / n_trials)   # Very Strong
            ]

            writer.writerow(row)

#evaluate_estimator_accuracy("./tests/data/preflop_hand_strength_accuracy.tsv", 0)
#evaluate_estimator_accuracy('./tests/data/flop_hand_strength_accuracy.tsv', 3)
#evaluate_estimator_accuracy('./tests/data/turn_hand_strength_accuracy.tsv', 4)
#evaluate_estimator_accuracy('./tests/data/river_hand_strength_accuracy.tsv', 5)

# Print the consistency rate for determining hand strength at each street
streets = {
    'Preflop': './tests/data/preflop_hand_strength_accuracy.tsv',
    'Flop':    './tests/data/flop_hand_strength_accuracy.tsv',
    'Turn':    './tests/data/turn_hand_strength_accuracy.tsv',
    'River':   './tests/data/river_hand_strength_accuracy.tsv'
}

cols = ['VeryWeak', 'Weak', 'Moderate', 'Strong', 'VeryStrong']

for street, filepath in streets.items():
    p = pd.read_csv(filepath, sep='\t')[cols]
    print(f"{street} consistency rate = {(p.max(axis=1) > 0.8).mean():.4f}")



def test_correctness():
    """
    Tests the hand strength evaluator on two known extreme hands at the river,
    where results are deterministic. A royal flush should return Very Strong (4)
    and a seven-high should return Very Weak (0).
    """
    # Royal flush should always return Very Strong (4)
    royal_flush_hand = ['Ah', 'Kh']
    royal_flush_board = ['Qh', 'Jh', 'Th', '2c', '3d']

    # 7 high (worst possible hand) should always return Very Weak (0)
    worst_hand = ['2c', '7d']
    worst_board = ['3h', '4s', '8d', '9c', 'Jh']

    # Check whether the correct hand strengths are assigned
    strong = hand_strength(royal_flush_hand, royal_flush_board)
    weak = hand_strength(worst_hand, worst_board)
    
    assert strong == 4, f"Expected Very Strong (4), got {strong}"
    assert weak == 0, f"Expected Very Weak (0), got {weak}"
    print("Correctness tests passed.")

test_correctness()



