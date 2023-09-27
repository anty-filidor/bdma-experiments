import os
import random

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_mean_log(logs: List[pd.DataFrame]):
    base_df = logs[0].copy()
    for _df in logs[1:]:
        base_df = base_df.add(_df, fill_value=0)
    return base_df / len(logs)


def get_std_log(logs: List[pd.DataFrame]):
    mean_df = get_mean_log(logs)
    base_df = logs[0].copy() * 0
    for _df in logs:
        base_df += np.power((_df - mean_df), 2)
    return np.sqrt(base_df / len(logs))


def get_metrics(experiment_results):
    epochs_vals = []
    start_A_vals = []
    start_B_vals = []
    final_A_vals = []
    final_B_vals = []
    for i, experiment_res in enumerate(experiment_results):
        experiment_res = experiment_res._global_stats_converted['layer_1']
        epochs_vals.append(experiment_res.shape[0])
        start_A_vals.append(experiment_res.iloc[0, 0])
        start_B_vals.append(experiment_res.iloc[0, 1])
        final_A_vals.append(experiment_res.iloc[-1, 0])
        final_B_vals.append(experiment_res.iloc[-1, 1])
    epochs = np.mean(epochs_vals)
    start_A = np.mean(start_A_vals)
    start_B = np.mean(start_B_vals)
    final_A = np.mean(final_A_vals)
    final_B = np.mean(final_B_vals)
    return epochs, start_A, start_B, final_A, final_B


def visualize_results_as_heatmap(results_df, index, columns, values, title, colorbar_label=None, x_label=None,
                                 y_label=None):
    methods = results_df['method'].unique()
    fig, axes = plt.subplots(1, len(methods), figsize=(12, 6))
    fig.suptitle(title, fontsize=16)

    vmin = results_df[values].min()
    vmax = results_df[values].max()

    for i, method in enumerate(methods):
        ax = axes[i]
        df = results_df[results_df['method'] == method]
        pivot_df = df.pivot_table(index=index, columns=columns, values=values)

        # Convert the pivot table data to a NumPy array
        heatmap_data = pivot_df.to_numpy()

        # Define the x and y labels
        x_labels = pivot_df.columns
        y_labels = pivot_df.index

        # Create a heatmap using pure matplotlib
        im = ax.imshow(heatmap_data, cmap='coolwarm', interpolation='nearest', vmin=vmin, vmax=vmax)

        # Set the x and y labels
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)

        # Display the colorbar
        cbar = fig.colorbar(im, ax=ax)
        label = colorbar_label if colorbar_label else values
        cbar.set_label(label)

        for j in range(len(y_labels)):
            for k in range(len(x_labels)):
                ax.text(k, j, heatmap_data[j, k], ha='center', va='center', color='black')

        # Set labels for the x and y axes
        x_label = x_label if x_label else columns
        ax.set_xlabel(x_label)
        y_label = y_label if y_label else index
        ax.set_ylabel(y_label)
        ax.set_title(method)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    # Show the heatmap
    plt.show()
