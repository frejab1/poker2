import numpy as np
import random
import csv
from treys import Card, Evaluator
from constants import *
evaluator = Evaluator()

def hand_transition(current_strength, street): 
    """Transition hand strength given current hand strength and street"""
    probs = TRANSITION_MATRICES[street][current_strength, :]
    return np.random.choice(len(probs), p=probs)


def player_action(strategy_type, hand_strength):
    """Get player action based on strategy and hand strength"""
    action_probs = EMISSION_MATRICES[strategy_type][hand_strength]
    return(np.random.choice(len(action_probs), p=action_probs))


def hand_strength(hand, board):
    "Calculate hand strength of a given hand and board using the treys package"
    treys_hand = [Card.new(c) for c in hand]
    treys_board = [Card.new(c) for c in board]
    iterations = 10000
    strengths = []

    
    # Get all remaining cards in deck
    remaining_cards = [c for c in DECK if c not in set(hand + board)]

    # River
    if len(board) == 5:
        hand_eval = evaluator.evaluate(treys_board, treys_hand)
        s = 1 - (hand_eval / 7462)

    # Turn
    elif len(board) == 4:
        # Simulate many possible hands, choosing a river hand from a new deck each time

        for _ in range(iterations):

            # Draw one random river card
            river_card = random.choice(remaining_cards)
            full_board = treys_board + [Card.new(river_card)]

            # Calculate hand strength
            hand_eval = evaluator.evaluate(full_board, treys_hand)
            strengths.append(1 - (hand_eval / 7462))

        # Mean hand strength from Monte Carlo simulations
        s = np.mean(strengths)
    

    # Flop
    elif len(board) == 3:

        for _ in range(iterations):
            # Sample turn and river without replacement
            turn_card, river_card = random.sample(remaining_cards, 2)

            full_board = treys_board + [
                Card.new(turn_card),
                Card.new(river_card)
            ]

            hand_eval = evaluator.evaluate(full_board, treys_hand)
            strengths.append(1 - (hand_eval / 7462))

        s = np.mean(strengths)


    # Pre-flop 
    elif len(board) == 0:

        for _ in range(iterations):
            # Sample flop (3), turn (1), river (1)
            flop_turn_river = random.sample(remaining_cards, 5)

            full_board = [
                Card.new(flop_turn_river[0]),
                Card.new(flop_turn_river[1]),
                Card.new(flop_turn_river[2]),
                Card.new(flop_turn_river[3]),
                Card.new(flop_turn_river[4]),
            ]

            hand_eval = evaluator.evaluate(full_board, treys_hand)
            strengths.append(1 - (hand_eval / 7462))

        s = np.mean(strengths)

    # Return categorical strength
    if s > 0.8: 
        return 4
    elif s > 0.6: 
        return 3
    elif s > 0.4: 
        return 2
    elif s > 0.2: 
        return 1
    else: 
        return 0


def hand_transition_data(n_hands):
    """ Generate hand strength data across betting rounds.
        Creates TSV file with hand strength categories (0-4) for the
        preflop, flop, turn, and river.
    """

    with open('./data/hand_transition_data.tsv', 'w', newline='') as tsvfile:
        tsv_writer = csv.writer(tsvfile, delimiter='\t')
        tsv_writer.writerow(['preflop', 'flop', 'turn', 'river'])
        data = []

        for hand_num in range(n_hands):
            deck = DECK.copy()
            np.random.shuffle(deck)

            row = []

            # Evaluate hand strength after pre-flop
            hand = [deck.pop() for _ in range(2)]
            row.append(hand_strength(hand, []))

            # Evaluate hand strength after flop
            board = [deck.pop() for _ in range(3)]
            row.append(hand_strength(hand, board))

            # Evaluate hand strength after turn
            board.append(deck.pop())
            row.append(hand_strength(hand, board))

            # Evaluate hand strength after river
            board.append(deck.pop())
            row.append(hand_strength(hand, board))

            # Store hand transition data
            tsv_writer.writerow(row)
            data.append(row)

    print("done")
    return(data)


def player_data(n_players, n_hands):
    """ Generate simulated poker player data for strategy analysis.
        Simulates players with different strategies making decisions based on
        evolving hand strengths across each betting round.
        Data is saved to './data/player_data.tsv' with columns: player_num, true_strategy, actions, true_states
    """

    with open('./data/player_data.tsv', 'w', newline='') as tsvfile:
        tsv_writer = csv.writer(tsvfile, delimiter='\t')
        tsv_writer.writerow(['player_num', 'true_strategy', 'actions', 'true_states'])

        data = []
        # Generate data for each strategy type
        for strategy in STRATEGY_TYPES:
            # Create multiple players with this strategy
            for player in range(n_players):
                states = []
                actions = []

                # Simulate multiple poker hands
                for hand_num in range(n_hands): 
                    # Four betting rounds
                    for round_num in range(4):
                        if round_num == 0:
                            # Choose initial hand strength from stationary distribution
                            current_state = np.random.choice(len(STAT_DIST), p=STAT_DIST)
                            states.append(current_state)
                        else: 
                            # Use last state from previous round
                            current_state = states[-1]

                        # Choose player action based on strategy type and hand strength
                        actions.append(player_action(strategy, current_state))

                        # Transition to next hand strength
                        if round_num < 3:
                            states.append(hand_transition(current_state, BETTING_ROUNDS[round_num]))

                # Store this player's data
                data.append([player, strategy, actions, states])
                tsv_writer.writerow([player, strategy, actions, states])
    
    return data
