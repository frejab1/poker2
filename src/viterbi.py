# viterbi.py
import numpy as np
from constants import *

def viterbi_algo(strategy, obs):
    # Fixed initial distribution
    initial_probs = np.array([0.205922, 0.311032, 0.279692, 0.156896, 0.046458])
    n_states = len(initial_probs)
    n_rounds = len(obs)

    # Map numbers to action names
    observations = [ACTIONS[o] for o in obs]

    # Emission probabilities for the current strategy
    emission_probs = OBS_LIKELIHOOD[strategy]
    
    # Initialise the trellis
    delta = np.zeros((n_rounds, n_states))
    parent_states = np.zeros((n_rounds, n_states), dtype=int)
    
    # Initialise arrays for trellis probabilities and parent states
    for state in range(n_states):
        delta[0, state] = initial_probs[state] * emission_probs[observations[0]][state, state]
        parent_states[0, state] = 0
    
    # Recurse for each state in each round
    for t in range(1, n_rounds):
        for j in range(n_states):

            # Find which previous hand strength makes the current one most likely
            probs = []
            for i in range(n_states):
                # Probability = (best path to prev state) * (transition prob) * (action prob)
                transition_prob = TRANSITION_MATRICES[BETTING_ROUNDS[t-1]][i, j]
                prob = delta[t-1, i] * transition_prob * emission_probs[observations[t]][j, j]
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
