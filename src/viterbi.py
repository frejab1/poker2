# viterbi.py
import numpy as np
import pandas as pd
from constants import *

def viterbi_algo(strategy, obs):
    """
    Run the Viterbi algorithm to the find most likely hand strength sequence for a string of observations, 
    given inferred strategy type.
    This uses a trellis framework to move between states and keep track of parent nodes.
    
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
    

    # Initialise an array to store the best overall path 
    best_path = np.zeros(n_rounds, dtype=int)
    # Get position of best state in the final round
    best_path[-1] = np.argmax(delta[-1, :])
    # Get probability of best state in the final round 
    best_prob = np.max(delta[-1, :])
    

    # Backtrack to reconstruct the best path
    for t in range(n_rounds-2, -1, -1):
        best_path[t] = parent_states[t+1, best_path[t+1]]
    
    return best_path.tolist(), best_prob


def print_viterbi_results(filepath, method):
    """
    Run the Viterbi algorithm on player data to infer hand strength sequences, using a given filepath and strategy inference method.
    The actions and states are split into chunks of 4 (one hand each), and Viterbi is run separately on each hand. 

    Args:
        filepath (str): Path to player data file
        method (str): the strategy inference method to be used, either "Forward Algorithm" of "Bayes' Inference"

    Prints:
        Accuracy percentage for state prediction.
    """
    # Load data
    df = pd.read_csv(filepath, sep='\t')
    df['actions'] = df['actions'].apply(eval)
    df['true_states'] = df['true_states'].apply(eval)

    # Use the correct data for the corresponding inference method
    if method == "Forward Algorithm":
        pred_df = pd.read_csv('data/forward_algo_results.tsv', sep='\t')
        column_name = "forward_pred_strategy"
    elif method == "Bayes' Inference":
        pred_df = pd.read_csv('data/bayes_inference_results.tsv', sep='\t')
        column_name = "bayes_pred_strategy"
    
    # Initialise counts and an array to store predictions
    exact_correct = 0
    total = 0
    pred_states_all = []
        
    # Iterate through all the player data from the given filepath
    for i in range(len(df)):
        strategy = pred_df[column_name].iloc[i]
        actions = df['actions'].iloc[i]
        true_states = df['true_states'].iloc[i]
        
        # Split into hands of 4 rounds each
        n_hands = int(len(actions) // 4)
        hand_preds = []
        for hand in range(n_hands):
            start = hand * 4
            end = start + 4
            
            hand_actions = actions[start:end]
            hand_true_states = true_states[start:end]
            
            # Run Viterbi on this single hand
            pred_states, _ = viterbi_algo(strategy, hand_actions)
            hand_preds.extend(pred_states)  
            # Check accuracy for each street individually
            for t in range(4):
                if pred_states[t] == hand_true_states[t]:
                    exact_correct += 1
                total += 1
        pred_states_all.append(hand_preds)

    # Calculate accuracies
    exact_accuracy = exact_correct / total if total > 0 else 0
    
    print(f"Viterbi River state prediction accuracy with {method} : {exact_accuracy:.2%}")

    # Append predicted states to player data
    df[f"{column_name}_strength"] = pred_states_all
    df.to_csv(filepath, sep='\t', index=False)
    
    return exact_accuracy