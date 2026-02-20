import numpy as np
import pandas as pd
from constants import *

def strategy_posterior(actions, states, prior=None):
    """P(strategy | actions, states) for each strategy."""
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
    
    # Posterior ∝ likelihood × prior
    unnormalised = {s: likelihood[s] * prior[s] for s in STRATEGY_TYPES}
    total = sum(unnormalised.values())
    
    return {s: unnormalised[s]/total for s in STRATEGY_TYPES} if total > 0 else {s: 0 for s in STRATEGY_TYPES}

def print_strategy_estimation_results():
    # Load data
    df = pd.read_csv('data/player_data.tsv', sep='\t')
    df['actions'] = df['actions'].apply(eval)
    df['true_states'] = df['true_states'].apply(eval)

    # Sequential updating across hands
    prior = None
    predictions = []
    
    for i in range(len(df)):
        post = strategy_posterior(df['actions'].iloc[i], df['true_states'].iloc[i], prior)
        predictions.append(max(post, key=post.get))
        prior = post

    # Results
    df['predicted'] = predictions
    print(df[['true_strategy', 'predicted']])
    print(f"Accuracy: {(df['true_strategy'] == df['predicted']).mean():.2%}")