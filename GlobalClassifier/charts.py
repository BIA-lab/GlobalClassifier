import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_metrics(results, save_path=None):
    """
    Plots various metrics from the results of model training and evaluation and saves the figure.

    Args:
        results (dict): Dictionary containing the scores and other results from model training.
        save_path (str): Path to save the plot image. If None, saves to 'metrics_plot.png' in the current directory.
    """
    if save_path is None:
        save_path = os.path.join(os.getcwd(), 'metrics_plot.png')

    df = results['scores']
    means = df.groupby('model').mean()
    stds = df.groupby('model').std()

    folds = df['fold'].iloc[0] 

    models = means.index.str.replace('Classifier', '', regex=False)
    x = np.arange(len(models))  
    width = 0.35  

    train_times = means['time_train']
    train_times_std = stds['time_train']
    test_times = means['time_test']
    test_times_std = stds['time_test']
    train_scores = means['accuracy_train']
    train_scores_std = stds['accuracy_train']
    test_scores = means['accuracy_test']
    test_scores_std = stds['accuracy_test']
    precisions = means['precision']
    precisions_std = stds['precision']
    recalls = means['recall']
    recalls_std = stds['recall']
    f1_scores = means['f1']
    f1_scores_std = stds['f1']

    fig, axs = plt.subplots(3, 2, figsize=(16, 12)) 
    fig.tight_layout(pad=4.0, rect=[0, 0.03, 1, 0.95])  

    # Plot 1: Average train and test time
    axs[0, 0].bar(x - width / 2, train_times, width, label='Train Time', color='skyblue', yerr=train_times_std, capsize=5)
    axs[0, 0].bar(x + width / 2, test_times, width, label='Test Time', color='orange', alpha=0.7, yerr=test_times_std, capsize=5)
    axs[0, 0].set_title(f'Average Train and Test Time ({folds} folds)')
    axs[0, 0].set_ylabel('Time (s)')
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(models, fontsize=8)
    axs[0, 0].legend()

    # Plot 2: Average train and test scores
    axs[0, 1].bar(x - width / 2, train_scores, width, label='Train accuracy', color='skyblue', yerr=train_scores_std, capsize=5)
    axs[0, 1].bar(x + width / 2, test_scores, width, label='Test accuracy', color='orange', alpha=0.7, yerr=test_scores_std, capsize=5)
    axs[0, 1].set_title(f'Average Train and Test accuracy ({folds} folds)')
    axs[0, 1].set_ylabel('accuracy')
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(models, fontsize=8)
    axs[0, 1].legend()

    # Plot 3: Average precision
    axs[1, 0].bar(models, precisions, color='skyblue', yerr=precisions_std, capsize=5)
    axs[1, 0].set_title(f'Average Precision ({folds} folds)')
    axs[1, 0].set_ylabel('Precision')
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(models, fontsize=8)

    # Plot 4: Average recall
    axs[1, 1].bar(models, recalls, color='skyblue', yerr=recalls_std, capsize=5)
    axs[1, 1].set_title(f'Average Recall ({folds} folds)')
    axs[1, 1].set_ylabel('Recall')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(models, fontsize=8)

    # Plot 5: Average F1 score
    axs[2, 0].bar(models, f1_scores, color='skyblue', yerr=f1_scores_std, capsize=5)
    axs[2, 0].set_title(f'Average F1 Score ({folds} folds)')
    axs[2, 0].set_ylabel('F1 Score')
    axs[2, 0].set_xticks(x)
    axs[2, 0].set_xticklabels(models, fontsize=8)

    fig.delaxes(axs[2, 1])

    plt.savefig(save_path)
    plt.close()


def labels_per_level(results, save_path=None):
    """
    Plots the number of unique labels per level in a hierarchical classification and saves the figure.

    Args:
        results (dict): Dictionary containing the 'y_true' series with hierarchical labels.
        save_path (str): Path to save the plot image. If None, saves to 'labels_per_level.png' in the current directory.
    """
    if save_path is None:
        save_path = os.path.join(os.getcwd(), 'labels_per_level.png')

    y_true_series = results['y_true']
    
    y_true_split = y_true_series.astype(str).str.split(';')
    
    levels = y_true_split.map(len).max()
    
    expanded_df = y_true_split.apply(pd.Series)
    expanded_df.columns = [f'level_{i+1}' for i in range(levels)]
    
    unique_counts = expanded_df.nunique()
    
    plt.figure(figsize=(10, 6))
    ax = unique_counts.plot(kind='bar', color='skyblue', alpha=0.8)
    
    for i, v in enumerate(unique_counts):
        ax.text(i, v + 0.2, str(v), ha='center', va='bottom', fontsize=10)
    
    plt.title(f'Labels per Level ({levels} Levels)', fontsize=14)
    plt.xlabel('Levels', fontsize=12)
    plt.ylabel('Labels', fontsize=12)
    
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()