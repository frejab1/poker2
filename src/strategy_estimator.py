import numpy as np
import pandas as pd
from constants import *

def strategy_posterior(actions, states, prior=None):
    """
    Calculate P(strategy | actions, states) for each strategy.
    
    Args:
        actions (list): List of actions (0=Fold, 1=Check, 2=Call, 3=Bet, 4=Raise)
        states (list): List of hand strengths (0-4) corresponding to each action
        prior (dict, optional): Prior probabilities for each strategy. If None, uses uniform prior.
    
    Returns:
        dict: Posterior probabilities for each strategy in STRATEGY_TYPES
    """
    if prior is None:
        prior = {s: 1/len(STRATEGY_TYPES) for s in STRATEGY_TYPES}
    
    # Likelihood P(actions | strategy, states)
    likelihood = {}
    for s in STRATEGY_TYPES:
        prob = 1.0
        # Create (action, hand strength) tuples
        for a, h in zip(actions, states):
            prob *= EMISSION_MATRICES[s][h][a]          
        likelihood[s] = prob
    
    # Calculate the posterior
    unnormalised = {s: likelihood[s] * prior[s] for s in STRATEGY_TYPES}
    total = sum(unnormalised.values())
    
    return {s: unnormalised[s]/total for s in STRATEGY_TYPES} if total > 0 else {s: 0 for s in STRATEGY_TYPES}

def print_strategy_estimation_results():
    """
    Load player data and estimate strategies for each player.
    
    Reads from 'data/player_data.tsv', which contains columns:
        - true_strategy: Actual strategy used
        - actions: List of actions (string representation)
        - true_states: List of hand strengths (string representation)
    
    Prints:
        DataFrame with true vs predicted strategies
        Overall accuracy percentage
    """
    # Load TSV data
    df = pd.read_csv('data/player_data.tsv', sep='\t')
    df['actions'] = df['actions'].apply(eval)
    df['true_states'] = df['true_states'].apply(eval)
    predictions = []
    
    for i in range(len(df)):
        # Uniform prior for each player
        post = strategy_posterior(df['actions'].iloc[i], df['true_states'].iloc[i], prior=None)
        predictions.append(max(post, key=post.get))

    # Results
    df['predicted'] = predictions
    print(df[['true_strategy', 'predicted']])
    print(f"Accuracy: {(df['true_strategy'] == df['predicted']).mean():.2%}")