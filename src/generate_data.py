import numpy as np
import random
import csv
from treys import Card, Evaluator
from constants import *
evaluator = Evaluator()

def hand_transition(current_strength, street): 
    """
    Transition hand strength from current round to next.
    
    Args:
        current_strength (int): Current hand strength category (0-4)
        street (str): Current betting round ('Preflop', 'Flop', 'Turn')
    
    Returns:
        int: New hand strength category (0-4)
    """
    probs = TRANSITION_MATRICES[street][current_strength, :]
    return np.random.choice(len(probs), p=probs)


def player_action(strategy_type, hand_strength):
    """
    Get player action based on strategy and current hand strength.
    
    Args:
        strategy_type (str): Player strategy from STRATEGY_TYPES
        hand_strength (int): Current hand strength category (0-4)
    
    Returns:
        int: Action taken (0=Fold, 1=Check, 2=Call, 3=Bet, 4=Raise)
    """
    action_probs = EMISSION_MATRICES[strategy_type][hand_strength]
    return np.random.choice(len(action_probs), p=action_probs)


def hand_strength(hand, board, iterations=10000):
    """
    Estimate hand strength using Monte Carlo simulation and treys package.
    
    Args:
        hand (list): Two-card hand (eg ['Ac', 'Kh'])
        board (list): Community cards (length 0, 3, 4, or 5)
        iterations (int): Number of Monte Carlo simulations
    
    Returns:
        int: Categorical hand strength (0=Very Weak to 4=Very Strong)
    """
    treys_hand = [Card.new(c) for c in hand]
    treys_board = [Card.new(c) for c in board]
    strengths = []
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

            # Calculate hand strength
            hand_eval = evaluator.evaluate(full_board, treys_hand)
            strengths.append(1 - (hand_eval / 7462))

        # Mean hand strength from Monte Carlo simulations
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

            # Calculate hand strength
            hand_eval = evaluator.evaluate(full_board, treys_hand)
            strengths.append(1 - (hand_eval / 7462))

        # Mean hand strength from Monte Carlo simulations
        s = np.mean(strengths)

    # Return categorical strength
    if s > STRENGTH_THRESHOLDS[4]:
        return 4
    elif s > STRENGTH_THRESHOLDS[3]:
        return 3
    elif s > STRENGTH_THRESHOLDS[2]:
        return 2
    elif s > STRENGTH_THRESHOLDS[1]:
        return 1
    else:
        return 0


def hand_transition_data(n_hands):
    """
    Generate hand strength data across betting rounds.
    
    Args:
        n_hands (int): Number of hands to simulate
    
    Returns:
        list: List of [preflop, flop, turn, river] strength categories
    
    Creates:
        ./data/hand_transition_data.tsv: TSV file with strength progression
    """

    # Write the data to a TSV file 
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

    return data


def player_data(n_players, n_hands):
    """
    Generate simulated poker player data for strategy analysis.
    
    Args:
        n_players (int): Number of players per strategy type
        n_hands (int): Number of hands per player
    
    Returns:
        list: List of [player_num, strategy, actions, states] simulations
    
    Creates:
        ./data/player_data.tsv: TSV file with player data
    """

    # Write the data to a TSV file 
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
