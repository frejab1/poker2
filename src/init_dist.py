# init_dist.py
import pandas as pd
import numpy as np

def calculate_initial_distribution(tsv_file):
    """
    Calculate the initial distribution of hand strengths from the
    'Preflop' column of a hand_transition_data TSV file.
    Returns a probability vector over hand strength categories 0-4.
    """
    # Load the TSV
    df = pd.read_csv(tsv_file, sep='\t')
    
    # Count occurrences of each hand strength category
    counts = df['preflop'].value_counts().sort_index()
    
    # Make sure all 5 categories 0-4 are present
    counts = counts.reindex(range(5), fill_value=0)
    
    # Normalise to get probabilities
    total = counts.sum()
    pi = counts / total
    
    return pi




