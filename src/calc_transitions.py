# calc_transitions.py
import pandas as pd
import numpy as np

def count_transitions_by_street(tsv_file):
    """Count transitions between hand strength states for each street"""
    df = pd.read_csv(tsv_file, sep='\t')
    
    # Initialise matrices for each transition
    street_pairs = {
        'preflop_to_flop': np.zeros((5, 5)),
        'flop_to_turn': np.zeros((5, 5)),
        'turn_to_river': np.zeros((5, 5))
    }
    
    # Count transitions between hand strengths for each row of data
    for _, row in df.iterrows():
        street_pairs['preflop_to_flop'][int(row['preflop']), int(row['flop'])] += 1
        street_pairs['flop_to_turn'][int(row['flop']), int(row['turn'])] += 1
        street_pairs['turn_to_river'][int(row['turn']), int(row['river'])] += 1
        
    return street_pairs


def normalise_matrix(count_matrix):
    """Normalise the count matrix to probabilities"""
    normalised = np.zeros((5, 5))
    for i in range(5):
        row_sum = np.sum(count_matrix[i])
        if row_sum > 0:
            normalised[i] = count_matrix[i] / row_sum
        else:
            normalised[i] = np.zeros(5)
    return normalised


def analyse_transitions(tsv_file):
    """Load the data, count transitions, and normalise matrices"""
    street_counts = count_transitions_by_street(tsv_file)

    # Normalise all matrices
    transition_matrices = {
        street: normalise_matrix(counts) 
        for street, counts in street_counts.items()
    }
    
    # Print the results
    np.set_printoptions(precision=2, suppress=True)
    for street, matrix in transition_matrices.items():
        print(f"\n{street.upper()} transition matrix:")
        print(matrix)
    
    return transition_matrices
