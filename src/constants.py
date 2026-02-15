# constants.py
import numpy as np

# Round dependent transition matrix for hand strength
TRANSITION_MATRICES = {
    # Preflop -> Flop
    'Preflop': np.array([
        [1.00, 0.00, 0.00, 0.00, 0.00], 
        [0.00, 0.56, 0.39, 0.04, 0.01], 
        [0.00, 0.43, 0.43, 0.12, 0.02], 
        [0.00, 0.00, 0.50, 0.37, 0.13],
        [0.00, 0.00, 0.00, 0.00, 1.00]
    ]) ,  

    # Flop -> Turn
    'Flop': np.array([
        [1.00, 0.00, 0.00, 0.00, 0.00], 
        [0.10, 0.65, 0.23, 0.02, 0.00], 
        [0.00, 0.27, 0.51, 0.20, 0.02], 
        [0.00, 0.00, 0.32, 0.60, 0.08],
        [0.00, 0.00, 0.00, 0.36, 0.64]
    ]) , 

    # Turn -> River
    'Turn': np.array([
        [0.64, 0.23, 0.13, 0.00, 0.00], 
        [0.42, 0.31, 0.17, 0.09, 0.01], 
        [0.02, 0.28, 0.44, 0.23, 0.03], 
        [0.00, 0.00, 0.27, 0.62, 0.11],
        [0.00, 0.00, 0.00, 0.33, 0.67]
    ])  
}

# Action names
ACTIONS = ['Fold', 'Check', 'Call', 'Bet', 'Raise']
ACTION_MAP = {0: 'Fold', 1: 'Check', 2: 'Call', 3: 'Bet', 4: 'Raise'}

# Hand strength labels
STRENGTHS = ['Very Weak', 'Weak', 'Medium', 'Strong', 'Very Strong']

# Strategy types and observation matrices
STRATEGY_TYPES = ['Tight_Aggressive', 'Loose_Aggressive', 'Tight_Passive', 'Loose_Passive', 'Maniac']
EMISSION_MATRICES =  {

    'Tight_Aggressive': np.array([
        # Selective with hands played but aggressive with strong hands
        [0.85, 0.10, 0.04, 0.01, 0.00],
        [0.50, 0.30, 0.15, 0.04, 0.01],
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
        [0.40, 0.30, 0.20, 0.08, 0.02],
        [0.20, 0.30, 0.35, 0.10, 0.05],
        [0.05, 0.20, 0.50, 0.20, 0.05],
        [0.00, 0.10, 0.60, 0.25, 0.05],
        [0.00, 0.05, 0.70, 0.20, 0.05]
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

# Generate the observation likelihood matrices from EMISSION_MATRICES
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

# Pre-flop win percentage table
PREFLOP_WIN_PCT = [
    [85, 68, 67, 66, 66, 64, 63, 63, 62, 62, 61, 60, 59],
    [66, 83, 64, 64, 63, 61, 60, 59, 58, 58, 57, 56, 55],
    [65, 62, 80, 61, 61, 59, 58, 56, 55, 55, 54, 53, 52],
    [65, 62, 59, 78, 59, 57, 56, 54, 53, 52, 51, 50, 50],
    [64, 61, 59, 57, 75, 56, 54, 53, 51, 49, 49, 48, 47],
    [62, 59, 57, 55, 53, 72, 53, 51, 50, 48, 46, 46, 45],
    [61, 58, 55, 53, 52, 50, 69, 50, 49, 47, 45, 43, 43],
    [60, 57, 54, 52, 50, 48, 47, 67, 48, 46, 45, 43, 41],
    [59, 56, 53, 50, 48, 47, 46, 45, 64, 46, 44, 42, 40],
    [60, 55, 52, 49, 47, 45, 44, 43, 43, 61, 44, 43, 41],
    [59, 54, 51, 48, 46, 43, 42, 41, 41, 41, 58, 42, 40],
    [58, 54, 50, 48, 45, 43, 40, 39, 39, 39, 38, 55, 39],
    [57, 53, 49, 47, 44, 42, 40, 37, 37, 37, 36, 35, 51]
]

# Stationary distribution
STAT_DIST = np.array([0.205922, 0.311032, 0.279692, 0.156896, 0.046458])

# Hand strength thresholds
STRENGTH_THRESHOLDS = [0.0, 0.2, 0.4, 0.6, 0.8]

# Card rank to index
RANK_TO_IDX = {'A':0, 'K':1, 'Q':2, 'J':3, 'T':4, '9':5, '8':6, '7':7, '6':8, '5':9, '4':10, '3':11, '2':12}

# Betting rounds
BETTING_ROUNDS = ['Preflop', 'Flop', 'Turn', 'River']
