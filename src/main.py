# main.py
from generate_data import *
from calc_transitions import analyse_transitions
from init_dist import calculate_initial_distribution
from forward_algorithm import print_forward_algo_results
from viterbi import viterbi_algo
from constants import *
from strategy_estimator import *

# Generate data about how hands transition between strengths
#hand_transition_data(10000)


# Calculate all 3 street transition matrices from the hand transition data
#analyse_transitions("./data/hand_transition_data.tsv")

#### Update transition matrices #####################
#### Update to actually use each one ###################


# Obtain the initial distribution of hands for the Preflop
#### Update to actually use when needed ###################
#pi = calculate_initial_distribution('data/hand_transition_data.tsv')
#print("Stationary distribution:", [round(v, 6) for v in pi])



# Generate simulated poker player data for strategy analysis
player_data(10, 5)


# Run the Forward Algorithm on a player's recent action sequence to estimate their strategy
#### Make so it takes player data and actions are not hardcoded ###############################
print_forward_algo_results()


#print_strategy_estimation_results()


# Run the Viterbi Algorithm to get a player's most likely hand strength sequence for this hand
#### Make so it takes player strategy and actions are not hardcoded ###############################
####  Check what emission matrix viterbi uses, diag or otherwise #################
best_path, best_prob = viterbi_algo("Tight_Aggressive", [0, 0, 1, 0])

print(f"Most likely hand strength sequence: {best_path}")
#print(f"Probability: {best_prob:.6f}")
#print(f"Interpretation: Preflop={best_path[0]}, Flop={best_path[1]}, Turn={best_path[2]}, River={best_path[3]}")
