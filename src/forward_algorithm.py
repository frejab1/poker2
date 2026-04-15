# forward_algorithm.py
import numpy as np
import pandas as pd
from constants import *

def log_sum_exp(log_values):
    """
    Stable computation of log(sum(exp(x))). 
    Used to avoid numerical underflow/overflow when summing probabilities in log space by rescaling them.

    Args:
        log_values (list): A list of log-probabilities.
    """
    max_log = np.max(log_values)
    return max_log + np.log(np.sum(np.exp(log_values - max_log)))


def forward_algo_single_hand(obs, strategy):
    """
    Run the Forward Algorithm for a HMM on a single poker hand with street-dependent transitions.
    
    Args:
    obs (list[int]): Sequence of actions for a single poker hand (length 4)
        strategy (str): Player strategy type
    
    Returns:
        float: Log probability of the observation sequence under the given strategy
    """
    # Initialise Q matrix & epsilon for numerical stability
    Q = OBS_LIKELIHOOD
    eps = 1e-12

    # Initialise forward probabilities in log space (avoids numerical underflow)
    action = ACTIONS[obs[0]]
    log_alpha = (np.log(STAT_DIST + eps) + np.log(Q[strategy][ACTIONS[obs[0]]] + eps))

    # Iterate through each remaining street in the hand
    for t in range(1, len(obs)):
            # Get the current action
            action = ACTIONS[obs[t]]

            # Convert to normal space for matrix multiplication
            alpha = np.exp(log_alpha)

            # Apply state transition if a valid transition matrix exists for this round
            if t - 1 < len(BETTING_ROUNDS) and BETTING_ROUNDS[t - 1] in TRANSITION_MATRICES:
                P = TRANSITION_MATRICES[BETTING_ROUNDS[t - 1]]
                alpha = alpha @ P

            # Apply the emission
            alpha = alpha * np.diag(Q[strategy][action])

            # Convert back to log space
            log_alpha = np.log(alpha + eps)

    # Rescale the probabilities to further avoid underflow
    return log_sum_exp(log_alpha) 


def forward_algo(observations):
    """
    Run the Forward Algorithm for each strategy type and compute the average log-likelihood across hands.
    Returns a score for each strategy.

    Args:
        observations (list[int]): Full action sequence across multiple hands

    Returns:
        list[float]: Log probability of the sequence under each strategy
    """
    log_probabilities = []
    n_hands = len(observations) // 4

    # Iterate through each strategy type
    for strategy in STRATEGY_TYPES:

        # Initialise log probabilties for all hands
        hand_log_probs = []

        # Iterate through each hand in the observation sequence
        for h in range(n_hands):
            start = h * 4
            end = start + 4
            hand_obs = observations[start:end]

            # Record the probabilities of each strategy
            hand_log_probs.append(
                forward_algo_single_hand(hand_obs, strategy)
            )

        # Store the average log-likelihood across the independent hands
        log_probabilities.append(np.mean(hand_log_probs))

    return log_probabilities


def print_forward_algo_results(filename):
    """
    Run the Forward Algorithm on a given file of player data, save the results to a TSV file, and print strategy predictions.
    
    Args:
        filename (str): Path to player data file
    """
    # Load the test data
    df = pd.read_csv(filename, sep='\t')

    # Convert lists of actions and hand strengths to Python lists
    df['actions'] = df['actions'].apply(eval)  
    df['true_states'] = df['true_states'].apply(eval)

    # Get a strategy prediction for each entry of data
    predictions = []
    for i in range(len(df)):
        obs = df['actions'].iloc[i]    

        # Use the log probabilities to prevent underflow
        log_probs = forward_algo(obs)
        predictions.append(STRATEGY_TYPES[np.argmax(log_probs)])

    # Write results to TSV
    df['forward_pred_strategy'] = predictions
    results_df = df[['true_strategy', 'forward_pred_strategy']]
    results_df.to_csv('./data/forward_algo_results.tsv', sep='\t', index=False)

    # Append predicted strategies to player data
    df.to_csv('./data/player_data.tsv', sep='\t', index=False)

    # Print the predictions
    print(df[['true_strategy', 'forward_pred_strategy']])
    accuracy = (df['true_strategy'] == df['forward_pred_strategy']).mean()
    print(f"Accuracy: {accuracy:.2%}")