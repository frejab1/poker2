# main.py
from generate_data import *
from calc_transitions import analyse_transitions
from init_dist import calculate_initial_distribution
from forward_algorithm import print_forward_algo_results
from viterbi import print_viterbi_results
from constants import *
from strategy_estimator import print_strategy_estimation_results

def main():
    """Run the complete poker HMM analysis pipeline."""

    print("=" * 60)
    print("POKER HMM ANALYSIS")
    print("=" * 60)

    # Generate data about how hands transition between strengths
    # Use this to obtain the hand strength transition matrices for each street & the initial distribution of hand strengths
    # Doesn't need to be repeated as the results are stored in the constants file as EMISSION_MATRICES (dict) and STAT_DIST (array)
    #hand_transition_data(1000000)
    #analyse_transitions("./data/hand_transition_data.tsv")
    #pi = calculate_initial_distribution('data/hand_transition_data.tsv')
    #print("Stationary distribution:", [round(v, 6) for v in pi])

    # Generate simulated poker player data
    print("\n[1/4] Generating player data...")
    player_data(10, 5)
    print("Player data saved to ./data/player_data.tsv")

    # Forward algorithm
    print("\n[2/4] Running Forward Algorithm for strategy estimation...")
    print_forward_algo_results()

    # Strategy Posterior
    print("\n[3/4] Running strategy posterior estimation...")
    print_strategy_estimation_results()

    # Run the Viterbi Algorithm to get a player's most likely hand strength sequence for this hand
    print("\n[4/4] Running Viterbi Algorithm for hand strength inference...")
    print_viterbi_results()

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()