# calc_transitions.py
import pandas as pd
import numpy as np
from constants import STRENGTHS

def count_transitions_by_street(tsv_file):
    """
    Count transitions between hand strength states for each street.
    
    Args:
        tsv_file (str): Path to TSV file with columns preflop, flop, turn, river
    
    Returns:
        dict: Count matrices for each street transition
    """
    df = pd.read_csv(tsv_file, sep='\t')
    n_states = len(STRENGTHS)
    
    # Initialise matrices for each transition
    street_pairs = {
        'preflop_to_flop': np.zeros((n_states, n_states)),
        'flop_to_turn': np.zeros((n_states, n_states)),
        'turn_to_river': np.zeros((n_states, n_states))
    }
    
    # Count transitions between hand strengths for each row of data
    for _, row in df.iterrows():
        street_pairs['preflop_to_flop'][int(row['preflop']), int(row['flop'])] += 1
        street_pairs['flop_to_turn'][int(row['flop']), int(row['turn'])] += 1
        street_pairs['turn_to_river'][int(row['turn']), int(row['river'])] += 1
        
    return street_pairs


def normalise_matrix(count_matrix):
    """
    Normalise a count matrix to probabilities, so that the rows sum to 1.
    
    Args: 
        count_matrix (array): 5x5 array of transition counts where rows represent current hand strength & columns represent next strength
    
    Returns:
        normalised_array: 5x5 probability matrix where each row sums to 1, rows with zero total count remain as zeros
    """
    n_states = len(STRENGTHS)
    normalised_array = np.zeros((n_states, n_states))
    for i in range(n_states):
        row_sum = np.sum(count_matrix[i])
        if row_sum > 0:
            normalised_array[i] = count_matrix[i] / row_sum
        else:
            normalised_array[i] = np.zeros(n_states)
    return normalised_array


def analyse_transitions(tsv_file):
    """
    Load hand strength data, count transitions, and compute probability matrices.
    
    Args:
        tsv_file (str): Path to TSV file with columns preflop, flop, turn, river, each column contains hand strength categories (0-4)
    
    Returns:
        dict: Dictionary with normalised transition matrices:
            - 'preflop_to_flop': P(flop | preflop)
            - 'flop_to_turn': P(turn | flop)
            - 'turn_to_river': P(river | turn)
    
    Prints:
        Each transition matrix with 2 decimal precision
    """
    street_counts = count_transitions_by_street(tsv_file)

    # Normalise all matrices
    transition_matrices = {
        street: normalise_matrix(counts) 
        for street, counts in street_counts.items()
    }
    
    # Print the results
    np.set_printoptions(precision=4, suppress=True)
    for street, matrix in transition_matrices.items():
        print(f"\n{street.upper()} transition matrix:")
        print(matrix)
    
    return transition_matrices
