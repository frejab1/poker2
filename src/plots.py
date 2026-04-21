import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from constants import STRATEGY_TYPES, STRENGTHS, BETTING_ROUNDS
from forward_algorithm import forward_algo
from generate_data import player_data
from strategy_estimator import strategy_posterior
from matplotlib.colors import LinearSegmentedColormap

def plot_strategy_accuracy(data_path='./data/player_data.tsv', save_path='./data/strategy_accuracy.png'):
    """
    Plot strategy prediction accuracy comparing Forward Algorithm and Bayes' Theorem.
    
    Args:
        data_path (str): Path to player_data.tsv file
        save_path (str): Path to save the plot image
    """
    # Load player data
    df = pd.read_csv(data_path, sep='\t')
    
    # Calculate accuracy per strategy for each algorithm
    forward_acc = []
    bayes_acc = []
    
    for strategy in STRATEGY_TYPES:
        subset = df[df['true_strategy'] == strategy]
        forward_acc.append((subset['forward_pred_strategy'] == subset['true_strategy']).mean())
        bayes_acc.append((subset['bayes_pred_strategy'] == subset['true_strategy']).mean())
    
    x = np.arange(len(STRATEGY_TYPES))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(x - 0.1, [v * 100 for v in forward_acc], 'o', color='#6C9EBF', label='Forward Algorithm', markersize=10)
    ax.plot(x + 0.1, [v * 100 for v in bayes_acc], 'o', color='#C73E1D', label='Bayes\' Theorem', markersize=10)
 
    # Remove underscores from strategy type labels
    STRATEGY_LABELS = [s.replace('_', ' ') for s in STRATEGY_TYPES]
 
    # Labels and formatting
    ax.set_xlabel('Strategy Type', fontweight='bold', labelpad=15)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', labelpad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(STRATEGY_LABELS, ha='center')
    ax.set_yticks(np.arange(92, 101, 1))
    ax.set_ylim(91, 101.5)
    ax.legend()
    
    ax.grid(True, axis='both', linestyle='-', alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return fig, ax

def plot_accuracy_vs_rounds(save_path='./data/accuracy_vs_rounds.png'):
    """
    Plot strategy prediction accuracy vs number of rounds for Forward Algorithm and Bayes' Theorem.
    Generates fresh player data for each round count and re-runs both algorithms.
    Each point shows the % of correct strategy predictions using n rounds of data.
    Args:
        save_path (str): Path to save the plot image
    """
    fwd_acc = []
    bay_acc = []
 
    for n_rounds in range(1, 21):
        # Generate player data with n_rounds hands
        player_data(1000, n_rounds)
 
        # Load the generated data
        df = pd.read_csv('./data/player_data.tsv', sep='\t')
        df['actions'] = df['actions'].apply(eval)
        df['true_states'] = df['true_states'].apply(eval)
 
        fwd_correct = []
        bay_correct = []
 
        for i in range(len(df)):
            actions = df['actions'].iloc[i]
            states = df['true_states'].iloc[i]
            true = df['true_strategy'].iloc[i]
 
            # Forward algorithm prediction on n_rounds of data
            fwd_pred = STRATEGY_TYPES[np.argmax(forward_algo(actions))]
            fwd_correct.append(fwd_pred == true)
 
            # Bayes' prediction on n_rounds of data
            post = strategy_posterior(actions, states)
            bay_pred = max(post, key=post.get)
            bay_correct.append(bay_pred == true)
 
        # Average accuracy across all players for this round
        fwd_acc.append(np.mean(fwd_correct) * 100)
        bay_acc.append(np.mean(bay_correct) * 100)
 
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, 21), bay_acc, label='Bayes\' Theorem', color='#C73E1D', linewidth=2, marker='o')   
    ax.plot(range(1, 21), fwd_acc, label='Forward Algorithm', color='#6C9EBF', linewidth=2, marker='o')
 
    # Formatting and labels
    ax.set_xticks(range(1, 21))
    ax.set_xlabel('Number of Training Hands', fontweight='bold', labelpad=15)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', labelpad=15)
    ax.set_yticks(np.arange(60, 105, 5)) 
    ax.set_ylim(60, 105)
    ax.legend()
    ax.grid(True, linestyle='-', alpha=0.7)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return fig, ax
 



def plot_hand_strength_heatmap(data_path='./data/player_data.tsv', save_path='./data/hand_strength_heatmap.png'):
    """
    Plot heatmaps of hand strength prediction accuracy for Forward Algorithm and Bayes' Theorem.
    Each cell [i][j] shows the percentage of times true hand strength i was predicted as hand strength j.
    Args:
        data_path (str): Path to player_data.tsv file
        save_path (str): Path to save the plot image
    """
    # Load player data
    df = pd.read_csv(data_path, sep='\t')
    for col in ['true_states', 'forward_pred_strategy_strength', 'bayes_pred_strategy_strength']:
        df[col] = df[col].apply(eval)

    # Flatten each column's lists into a single 1D array across all players
    true_all = np.array(df['true_states'].tolist()).flatten()
    fwd_all = np.array(df['forward_pred_strategy_strength'].tolist()).flatten()
    bay_all = np.array(df['bayes_pred_strategy_strength'].tolist()).flatten()

    hand_strengths = [0, 1, 2, 3, 4]

    # For each true hand strength i, calculate % predicted as each hand strength j
    def build_matrix(pred_all):
        return np.array([[(pred_all[true_all == i] == j).mean() * 100 for j in hand_strengths] for i in hand_strengths])

    # Draw the heat map
    def draw_heatmap(ax, matrix, title):
        cmap = LinearSegmentedColormap.from_list('custom', ['white', 'black'])
        im = ax.imshow(matrix, vmin=0, vmax=100, cmap=cmap)

        # Add black lines between cells
        for i in range(len(hand_strengths) + 1):
            ax.axhline(i - 0.5, color='black', linewidth=1)
            ax.axvline(i - 0.5, color='black', linewidth=1)

        # Map hand strengths to readable labels
        labels = [STRENGTHS[h] for h in hand_strengths]
        ax.set_xticks(range(len(hand_strengths)))
        ax.set_yticks(range(len(hand_strengths)))
        ax.set_xticklabels(labels, fontsize=6, ha='center', rotation=0)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel('Predicted Hand Strength', fontweight='bold', labelpad=10)
        ax.set_ylabel('True Hand Strength', fontweight='bold', labelpad=-5)
        ax.set_title(title, fontweight='bold', pad=15)
        return im

    # Plot forward and bayes heatmaps in one
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plt.subplots_adjust(wspace=0.3)
    im1 = draw_heatmap(ax1, build_matrix(fwd_all), 'The Forward Algorithm')
    im2 = draw_heatmap(ax2, build_matrix(bay_all), 'Bayesian Inference')
   
    # Single colorbar on the right for both heatmaps
    fig.colorbar(im2, ax=[ax1, ax2], label='% Predicted Accurately')

    plt.savefig(save_path, dpi=150)
    plt.close()
    return fig, (ax1, ax2)

def plot_hand_strength_over_streets(data_path='./data/player_data.tsv', save_path='./data/hand_strength_streets.png'):
    """
    Plot hand strength prediction accuracy over betting streets for Forward Algorithm and Bayes' Theorem.
    Each point shows the % of correct hand strength predictions at that street across all players.
    Args:
        data_path (str): Path to player_data.tsv file
        save_path (str): Path to save the plot image
    """
    # Load the data
    df = pd.read_csv(data_path, sep='\t')
    for col in ['true_states', 'forward_pred_strategy_strength', 'bayes_pred_strategy_strength']:
        df[col] = df[col].apply(eval)
 
    # Flatten into 2D arrays: rows = players, cols = time steps
    true_all = np.array(df['true_states'].tolist())
    fwd_all = np.array(df['forward_pred_strategy_strength'].tolist())
    bay_all = np.array(df['bayes_pred_strategy_strength'].tolist())
 
    # Calculate accuracy at each street 
    n_streets = len(BETTING_ROUNDS)
    fwd_acc = []
    bay_acc = []
    for street_idx in range(n_streets):
        # Get all time steps corresponding to this street 
        street_cols = range(street_idx, true_all.shape[1], n_streets)
        fwd_acc.append((fwd_all[:, street_cols] == true_all[:, street_cols]).mean() * 100)
        bay_acc.append((bay_all[:, street_cols] == true_all[:, street_cols]).mean() * 100)
 
    x = np.arange(n_streets)
 
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x - 0.1, fwd_acc, 'o', color='#6C9EBF', label='Forward Algorithm', markersize=10)
    ax.plot(x + 0.1, bay_acc, 'o', color='#C73E1D', label='Bayes\' Theorem', markersize=10)
 
    all_vals = fwd_acc + bay_acc
    y_min = max(0, min(all_vals) - 3)
    y_max = min(100, max(all_vals) + 3)
 
    # Add labels
    ax.set_xticks(x)
    ax.set_xticklabels(BETTING_ROUNDS, ha='center', fontsize=10)
    ax.set_xlabel('Street', fontweight='bold', labelpad=15)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', labelpad=15)
    ax.set_yticks(np.arange(50, 71, 2))
    ax.set_ylim(50, 71)
    ax.legend()
    ax.grid(True, axis='both', linestyle='-', alpha=0.35)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return fig, ax

def plot_time_vs_rounds(save_path='./data/time_vs_rounds.png'):
    """
    Plot time taken to run Forward Algorithm vs Bayes' Theorem for increasing rounds.
    Args:
        save_path (str): Path to save the plot image
    """
    import time

    fwd_times = []
    bay_times = []

    for n_rounds in range(1, 21):
        # Generate player data with n_rounds hands
        player_data(100, n_rounds)
        df = pd.read_csv('./data/player_data.tsv', sep='\t')
        df['actions'] = df['actions'].apply(eval)
        df['true_states'] = df['true_states'].apply(eval)

        # Time the Forward algorithm across all players
        start = time.time()
        for i in range(len(df)):
            forward_algo(df['actions'].iloc[i])
        fwd_times.append(time.time() - start)

        # Time Bayes' inference across all players
        start = time.time()
        for i in range(len(df)):
            strategy_posterior(df['actions'].iloc[i], df['true_states'].iloc[i])
        bay_times.append(time.time() - start)

    # Plot as a line chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, 21), fwd_times, label='Forward Algorithm', color='#6C9EBF', linewidth=2, marker='o')
    ax.plot(range(1, 21), bay_times, label='Bayes\' Theorem', color='#C73E1D', linewidth=2, marker='o')

    # Add labels and formatting
    ax.set_xticks(range(1, 21))
    ax.set_xlabel('Number of Training Hands', fontweight='bold', labelpad=15)
    ax.set_ylabel('Time (s)', fontweight='bold', labelpad=15)
    ax.set_yticks(np.arange(0, 1, 0.05)) 
    ax.legend()
    ax.grid(True, linestyle='-', alpha=0.35)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return fig, ax