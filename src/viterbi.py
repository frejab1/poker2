# viterbi.py
import numpy as np
import pandas as pd
from constants import *

def viterbi_algo(strategy, obs):
    """
    Run Viterbi algorithm to find most likely hand strength sequence.
    
    Args:
        strategy (str): Player strategy from STRATEGY_TYPES
        obs (list): Sequence of observed actions (0-4)
    
    Returns:
        tuple: (best_path, best_prob)
            best_path (list): Most likely hand strength sequence (0-4)
            best_prob (float): Probability of this path
    """
    n_states = len(STAT_DIST)
    n_rounds = len(obs)
    emission_probs = EMISSION_MATRICES[strategy]
    
    # Initialise the trellis
    delta = np.zeros((n_rounds, n_states))
    parent_states = np.zeros((n_rounds, n_states), dtype=int)
    
    # Initialise arrays for trellis probabilities and parent states
    for state in range(n_states):
        delta[0, state] = STAT_DIST[state] * emission_probs[state, obs[0]]
        parent_states[0, state] = 0
    
    # Recurse for each state in each round
    for t in range(1, n_rounds):
        for j in range(n_states):

            # Find which previous hand strength makes the current one most likely
            probs = []
            for i in range(n_states):
                # Probability = (best path to prev state) * (transition prob) * (action prob)
                transition_prob = TRANSITION_MATRICES[BETTING_ROUNDS[t-1]][i, j]
                prob = delta[t-1, i] * transition_prob * emission_probs[j, obs[t]]
                probs.append(prob)
            
            # Store the best probability and the state it came from
            delta[t, j] = max(probs)
            parent_states[t, j] = np.argmax(probs)
    

    # Find the best overall path 
    best_path = np.zeros(n_rounds, dtype=int)
    # Get position of best state in the final round
    best_path[-1] = np.argmax(delta[-1, :])
    # Get probability of best state in the final round 
    best_prob = np.max(delta[-1, :])
    

    # Backtrack to reconstruct the best path
    for t in range(n_rounds-2, -1, -1):
        best_path[t] = parent_states[t+1, best_path[t+1]]
    
    return best_path.tolist(), best_prob


def print_viterbi_results():
    """
    Run Viterbi algorithm on player data to infer hand strength sequences.
    
    Reads from 'data/player_data.tsv', which contains:
        - true_strategy: Actual strategy used
        - actions: List of actions across multiple hands (as string representation)
        - true_states: List of hand strengths across multiple hands (as string representation)
    
    The actions and states are split into chunks of 4 (one hand each), and Viterbi is run separately on each hand. 
    Only the final state of each hand is considered as in real applications one would only want to know their opponents' current hand strength.

    Prints:
        Exact accuracy percentage for River state prediction
        ±1 margin accuracy percentage (prediction within 1 of true value)
    """
    # Load data
    df = pd.read_csv('data/player_data.tsv', sep='\t')
    df['actions'] = df['actions'].apply(eval)
    df['true_states'] = df['true_states'].apply(eval)
    
    exact_correct = 0
    margin_correct = 0
    total = 0
    
    for i in range(len(df)):
        strategy = df['true_strategy'].iloc[i]
        actions = df['actions'].iloc[i]
        true_states = df['true_states'].iloc[i]
        
        # Split into hands of 4 rounds each
        n_hands = int(len(actions) // 4)
        
        for hand in range(n_hands):
            start = hand * 4
            end = start + 4
            
            hand_actions = actions[start:end]
            hand_true_states = true_states[start:end]
            
            # Run Viterbi on this single hand
            pred_states, _ = viterbi_algo(strategy, hand_actions)
            
            # Check last element (River)
            pred = pred_states[-1]
            true = hand_true_states[-1]
            
            # Exact match
            if pred == true:
                exact_correct += 1
            
            # Within ±1
            if abs(pred - true) <= 1:
                margin_correct += 1
            
            total += 1
    
    # Calculate accuracies
    exact_accuracy = exact_correct / total if total > 0 else 0
    margin_accuracy = margin_correct / total if total > 0 else 0
    
    print(f"Viterbi River state prediction accuracy (exact): {exact_accuracy:.2%}")
    print(f"Viterbi River state prediction accuracy (±1 category): {margin_accuracy:.2%}")
    
    return exact_accuracy, margin_accuracy