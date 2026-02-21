# init_dist.py
import pandas as pd
import numpy as np
from constants import STRENGTHS

def calculate_initial_distribution(tsv_file):
    """
    Calculate initial distribution of hand strengths from preflop data.
    
    Args:
        tsv_file (str): Path to TSV file with 'preflop' column containing hand strength categories (0-4)
    
    Returns:
        pi (array): Probability vector of length 5, where element i is P(initial hand strength = i) & the probabilities sum to 1.
    """
    # Load the TSV file
    df = pd.read_csv(tsv_file, sep='\t')
    n_states = len(STRENGTHS)
    
    # Count occurrences of each hand strength category
    counts = df['preflop'].value_counts().sort_index()
    
    # Make sure all categories (0-4) are present
    counts = counts.reindex(range(n_states), fill_value=0)
    
    # Normalise to get initial probabilities
    total = counts.sum()
    pi = counts / total
    
    return pi




