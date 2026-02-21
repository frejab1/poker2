# forward_algorithm.py
import numpy as np
import pandas as pd
from constants import *


def forward_algo(observations):
    """
    Run forward algorithm for HMM with street-dependent transitions.
    
    Args:
        observations (list): Sequence of actions (0-4) across multiple hands (string representation)
    
    Returns:
        list: Probability of observations under each strategy in STRATEGY_TYPES
    """
    # Initialise Q matrix & probabilities array
    Q = OBS_LIKELIHOOD
    probabilities = []

    # For each strategy get the probability of the sequence of observations
    for strategy in STRATEGY_TYPES:
        alpha = None

        # Run the forward algorithm
        for i in range(len(observations)):
            # Get the current street
            street_index = i % len(BETTING_ROUNDS)
            street = BETTING_ROUNDS[street_index]

            action = ACTIONS[observations[i]]
            q_matrix = Q[strategy][action]

            # Initialise alpha
            if alpha is None:
                alpha = np.matmul(STAT_DIST, q_matrix)
            
            # Reset to the initial distribution at the start of each new hand
            elif street_index == 0:
                alpha = np.matmul(alpha, np.diag(STAT_DIST))
                alpha = np.matmul(alpha, q_matrix)

            # Use corresponding transition matrix for street
            else:
                P = TRANSITION_MATRICES[BETTING_ROUNDS[street_index - 1]]
                alpha = np.matmul(alpha, P)
                alpha = np.matmul(alpha, q_matrix)

        probabilities.append(np.sum(alpha) if alpha is not None else 0)
            
    return probabilities


def print_forward_algo_results():
    """
    Run forward algorithm on player data and print strategy predictions.
    
    Reads from 'data/player_data.tsv', which contains:
        - true_strategy: Actual strategy used
        - actions: List of actions (as string representation)
        - true_states: List of hand strengths (as string representation)
    
    Prints:
        DataFrame with true vs predicted strategies
        Overall accuracy percentage
    """
    # Load the test data
    df = pd.read_csv('data/player_data.tsv', sep='\t')

    # Convert lists of actions and hand strengths to Python lists
    df['actions'] = df['actions'].apply(eval)  
    df['true_states'] = df['true_states'].apply(eval)

    # Get a strategy prediction for each entry of data
    predictions = []
    for i in range(len(df)):
        obs = df['actions'].iloc[i]    
        probs = forward_algo(obs)
        predictions.append(STRATEGY_TYPES[np.argmax(probs)])

    # Print the predictions
    df['predicted_strategy'] = predictions
    print(df[['true_strategy', 'predicted_strategy']])
    accuracy = (df['true_strategy'] == df['predicted_strategy']).mean()
    print(f"Accuracy: {accuracy:.2%}")
