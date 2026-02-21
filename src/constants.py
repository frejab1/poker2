# constants.py
import numpy as np

# Probability of hand strength changing from one round to the next
# Rows: current strength, Cols: next strength (0=Very Weak to 4=Very Strong)
TRANSITION_MATRICES = {
    # Preflop -> Flop
    'Preflop': np.array([
        [1.00, 0.00, 0.00, 0.00, 0.00], 
        [0.00, 0.56, 0.39, 0.04, 0.01], 
        [0.00, 0.43, 0.43, 0.12, 0.02], 
        [0.00, 0.00, 0.50, 0.37, 0.13],
        [0.00, 0.00, 0.00, 0.00, 1.00]
    ]),  

    # Flop -> Turn
    'Flop': np.array([
        [1.00, 0.00, 0.00, 0.00, 0.00], 
        [0.10, 0.65, 0.23, 0.02, 0.00], 
        [0.00, 0.27, 0.51, 0.20, 0.02], 
        [0.00, 0.00, 0.32, 0.60, 0.08],
        [0.00, 0.00, 0.00, 0.36, 0.64]
    ]), 

    # Turn -> River
    'Turn': np.array([
        [0.64, 0.23, 0.13, 0.00, 0.00], 
        [0.42, 0.31, 0.17, 0.09, 0.01], 
        [0.02, 0.28, 0.44, 0.23, 0.03], 
        [0.00, 0.00, 0.27, 0.62, 0.11],
        [0.00, 0.00, 0.00, 0.33, 0.67]
    ])  
}

ACTIONS = ['Fold', 'Check', 'Call', 'Bet', 'Raise']
ACTION_MAP = {0: 'Fold', 1: 'Check', 2: 'Call', 3: 'Bet', 4: 'Raise'}

STRENGTHS = ['Very Weak', 'Weak', 'Medium', 'Strong', 'Very Strong']
STRATEGY_TYPES = ['Tight_Aggressive', 'Loose_Aggressive', 'Tight_Passive', 'Loose_Passive', 'Maniac']
BETTING_ROUNDS = ['Preflop', 'Flop', 'Turn', 'River']


# Probability of each action given hand strength for each player type
# Rows: hand strength (0=Very Weak to 4=Very Strong)
# Cols: action (0=Fold, 1=Check, 2=Call, 3=Bet, 4=Raise)
EMISSION_MATRICES =  {
    'Tight_Aggressive': np.array([
        # Selective with hands played but aggressive with strong hands
        [0.85, 0.10, 0.05, 0.00, 0.00],
        [0.50, 0.30, 0.15, 0.05, 0.00],
        [0.10, 0.40, 0.30, 0.15, 0.05],
        [0.00, 0.10, 0.25, 0.45, 0.20],
        [0.00, 0.02, 0.08, 0.40, 0.50]
    ]),

    'Loose_Aggressive': np.array([
        # Plays many hands but aggressive with strong hands
        [0.30, 0.25, 0.20, 0.15, 0.10],
        [0.15, 0.25, 0.25, 0.20, 0.15],
        [0.05, 0.15, 0.25, 0.35, 0.20],
        [0.00, 0.05, 0.15, 0.40, 0.40],
        [0.00, 0.02, 0.08, 0.30, 0.60]
    ]),

    'Tight_Passive': np.array([
        # Selective with hands played but calls even with strong hands
        [0.90, 0.08, 0.02, 0.00, 0.00],  
        [0.70, 0.20, 0.10, 0.00, 0.00],
        [0.40, 0.40, 0.20, 0.00, 0.00],
        [0.10, 0.50, 0.40, 0.00, 0.00],
        [0.00, 0.40, 0.60, 0.00, 0.00]
    ]),

    'Loose_Passive': np.array([
        # Plays many hands but calls too often 
        [0.50, 0.30, 0.15, 0.05, 0.00],
        [0.30, 0.35, 0.30, 0.05, 0.00],
        [0.10, 0.30, 0.55, 0.05, 0.00],
        [0.02, 0.20, 0.70, 0.06, 0.02],
        [0.00, 0.10, 0.80, 0.08, 0.02]
    ]),

    'Maniac': np.array([
        # Very aggressive, raises even with very weak hands
        [0.00, 0.10, 0.20, 0.40, 0.30],
        [0.00, 0.05, 0.15, 0.45, 0.35],
        [0.00, 0.02, 0.10, 0.50, 0.38],
        [0.00, 0.00, 0.05, 0.45, 0.50],
        [0.00, 0.00, 0.02, 0.38, 0.60]
    ])
}

# Observation likelihood matrices for the forward algorithm
OBS_LIKELIHOOD = {}
for strategy, matrix in EMISSION_MATRICES.items():
    OBS_LIKELIHOOD[strategy] = {
        'Fold': np.diag(matrix[:, 0]),
        'Check': np.diag(matrix[:, 1]),
        'Call': np.diag(matrix[:, 2]),
        'Bet': np.diag(matrix[:, 3]),
        'Raise': np.diag(matrix[:, 4])
    }

# Card deck
SUITS = ['c', 'h', 'd', 's']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
DECK = [r + s for s in SUITS for r in RANKS]
RANK_TO_IDX = {'A':0, 'K':1, 'Q':2, 'J':3, 'T':4, '9':5, '8':6, '7':7, '6':8, '5':9, '4':10, '3':11, '2':12}

# Hand strength thresholds
STRENGTH_THRESHOLDS = [0.0, 0.2, 0.4, 0.6, 0.8]

# Derived stationary distribution
STAT_DIST = np.array([0.2059, 0.3110, 0.2797, 0.1569, 0.0465])