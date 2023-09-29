import os
import random

from typing import List

import matplotlib.pyplot as plt
import networkx as nx
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
        layer = list(experiment_res._global_stats_converted.keys())[0]
        experiment_res = experiment_res._global_stats_converted[layer]
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


def visualize_results_as_heatmap(results_df, index, columns, values, title, normalize=True, colorbar_label=None,
                                 x_label=None, y_label=None):
    methods = sorted(results_df['method'].unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(12, 6))
    fig.suptitle(title, fontsize=16)

    if normalize:
        agents_sum = sum([results_df[c][0] for c in results_df.columns if c.startswith('start_')])
        for c in results_df.columns:
            if c.startswith('start_') or c.startswith('final_'):
                results_df[c] = results_df[c]/agents_sum

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
                ax.text(k, j, "{:.2f}".format(heatmap_data[j, k]), ha='center', va='center', color='black')

        # Set labels for the x and y axes
        x_label = x_label if x_label else columns
        ax.set_xlabel(x_label)
        y_label = y_label if y_label else index
        ax.set_ylabel(y_label)
        ax.set_title(method)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    # Show the heatmap
    plt.show()


def preprocess_temporal_network(network, theta, undirected=False):
    layer_name = list(network.snaps[0].layers.keys())[0]
    for snap in network.snaps:
        # 1. Drop selfloops
        snap[layer_name].remove_edges_from(nx.selfloop_edges(snap[layer_name]))

        # 2. Drop edges below theta
        drop_edges = list(filter(lambda e: e[2] <= theta, (e for e in snap[layer_name].edges.data('weight'))))
        drop_edges_ids = list(e[:2] for e in drop_edges)
        snap[layer_name].remove_edges_from(drop_edges_ids)
        if undirected:
            snap.layers[layer_name] = snap.layers[layer_name].to_undirected()
    return network


def compare_nets(n1, n2):
    n1_layer_name = list(n1.snaps[0].layers.keys())[0]
    n2_layer_name = list(n2.snaps[0].layers.keys())[0]
    n1_edges_no = []
    n2_edges_no = []
    snapshots = list(zip(n1.snaps, n2.snaps))
    for s1, s2 in snapshots:
        n1_edges_no.append(len(s1[n1_layer_name].edges))
        n2_edges_no.append(len(s2[n2_layer_name].edges))
    x = list(range(0, len(snapshots)))
    plt.plot(x, n1_edges_no)
    plt.plot(x, n2_edges_no)
    plt.show()


def create_static_network(event_data_path: str, delimiter: str, undirected=False, source='Sender', target='Recipient'):
    df_interactions = pd.read_csv(event_data_path, delimiter=delimiter)
    df_interactions = df_interactions[[source, target]]
    method = nx.Graph if undirected else nx.DiGraph
    static_graph = nx.from_pandas_edgelist(df_interactions, source=source, target=target, create_using=method)
    static_graph.remove_edges_from(nx.selfloop_edges(static_graph))
    return static_graph
