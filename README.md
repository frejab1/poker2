# Hidden Markov Models for Poker Analysis

## Overview

This project implements Hidden Markov Models (HMMs) for poker strategy analysis and hand strength inference. It simulates poker player behavior, estimates playing styles, and infers hidden hand strengths from observed actions.

## Project Structure
├── data/ # Generated data output directory
├── src/
│ ├── calc_transitions.py # Transition matrix calculation
│ ├── constants.py # HMM parameters and poker constants
│ ├── forward_algorithm.py # Forward algorithm for strategy inference
│ ├── generate_data.py # Simulated poker data generation
│ ├── init_dist.py # Initial distribution calculation
│ ├── main.py # Main execution pipeline
│ ├── plots.py # Visualisation functions
│ ├── strategy_estimator.py # Bayesian strategy posterior
│ └──  viterbi.py # Viterbi algorithm for hand strength inference
├── tests/
│ ├── hand_strength_tests.py # Hand strength estimator tests
│ ├── inference_tests.py # Combined inference tests
│ ├── matrices_tests.py # Matrix validation tests
│ └── viterbi_tests.py # Viterbi algorithm tests
├── venv/   # Virtual environment for running Python
├── README.md
└── requirements.txt # Project dependencies



## Key Features

### 1. Poker Simulation
- Monte Carlo hand strength estimation using `Treys` poker evaluation library
- Realistic hand strength transitions across betting rounds (Preflop → Flop → Turn → River)
- Five strategy types: Tight_Aggressive, Loose_Aggressive, Tight_Passive, Loose_Passive, Maniac
- Five hand strength categories: Very Weak (0) to Very Strong (4)

### 2. Strategy Inference Methods

#### Forward Algorithm (`forward_algorithm.py`)
- Computes log-likelihood of observed action sequences under each strategy
- Handles numerical stability with log-space calculations
- Returns average log-likelihood across multiple hands

#### Bayesian Inference (`strategy_estimator.py`)
- Calculates P(strategy | actions, hand_strengths) using Bayes' theorem
- Returns posterior probabilities for each strategy type

### 3. Hand Strength Inference (`viterbi.py`)
- Viterbi algorithm to find most likely hidden hand strength sequence
- Uses strategy type and observed actions as inputs
- Returns best path and its probability

### 4. Analysis & Visualisation (`plots.py`)
- Strategy prediction accuracy comparison
- Hand strength prediction heatmaps
- Accuracy over betting rounds
- Performance scaling with training data size
- Runtime comparison

## Installation

### Requirements
```bash
pip install -r requirements.txt

```
Or install individually:

```bash
pip install numpy==2.4.2 pandas==3.0.0 treys==0.1.8

```

## Usage

### Run Complete Pipeline
```bash
python main.py

```

This will:

1. Generate simulated player data
2. Run Forward Algorithm for strategy estimation
3. Run Bayesian inference for strategy estimation
4. Run Viterbi algorithm for hand strength inference
5. Generate all visualisation plots

### Generate Hand Transition Data (Optional)
Data generation for the hand strength transitions is commented out as this needs to be done just once and requires significant computational power. 

```bash
from generate_data import hand_transition_data
hand_transition_data(1000000)  # Generate 1M hands
```

### Run Individual Tests
```bash
python tests/hand_strength_tests.py
python tests/inference_tests.py
python tests/matrices_tests.py
python tests/viterbi_tests.py
```

## Constants Reference

### Strategy Types
| Strategy | Description |
|----------|-------------|
| Tight_Aggressive | Selective, aggressive with strong hands |
| Loose_Aggressive | Many hands, aggressive with strong hands |
| Tight_Passive | Selective, passive even with strong hands |
| Loose_Passive | Many hands, calls too often |
| Maniac | Very aggressive, bets/raises with any hand |

### Actions
| Code | Action |
|------|--------|
| 0    | Fold   |
| 1    | Check  |
| 2    | Call   |
| 3    | Bet    |
| 4    | Raise  |

### Hand Strengths
| Code | Category     |
|------|-------------|
| 0    | Very Weak   |
| 1    | Weak        |
| 2    | Medium      |
| 3    | Strong      |
| 4    | Very Strong |

### Betting Rounds
| Index | Round    |
|-------|----------|
| 0     | Preflop  |
| 1     | Flop     |
| 2     | Turn     |
| 3     | River    |


## Output Files

| File | Description |
|------|-------------|
| `data/accuracy_vs_rounds.png` | Accuracy vs number of training hands line plot |
| `data/bayes_inference_results.tsv` | Bayesian inference strategy predictions |
| `data/forward_algo_results.tsv` | Forward algorithm strategy predictions |
| `data/hand_strength_heatmap.png` | Hand strength prediction accuracy heatmaps |
| `data/hand_strength_streets.png` | Accuracy by betting round bar chart |
| `data/player_data.tsv` | Generated player data with predictions appended |
| `data/strategy_accuracy.png` | Strategy prediction accuracy comparison bar chart |
| `data/time_vs_rounds.png` | Runtime comparison line plot |

## Testing

### Test Files
| Test File | Description |
|-----------|-------------|
| `tests/hand_strength_tests.py` | Validates hand strength estimator on known hands |
| `tests/inference_tests.py` | Combined forward and Bayesian inference tests |
| `tests/matrices_tests.py` | Validates transition and emission matrices sum to 1 |
| `tests/viterbi_tests.py` | Tests Viterbi hand strength inference |

### Run All Tests
```bash
python tests/hand_strength_tests.py
python tests/inference_tests.py
python tests/matrices_tests.py
python tests/viterbi_tests.py
```

## Acknowledgments

- Treys library for poker hand evaluation
- NumPy, Pandas, and Matplotlib for scientific computing
