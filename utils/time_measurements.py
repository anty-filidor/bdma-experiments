import time

from dataclasses import dataclass
from typing import Any, Callable, List, Union

import matplotlib.pyplot as plt
import network_diffusion as nd
import numpy as np
import pandas as pd


@dataclass
class DataSample:
    name: str
    data: Any


def run_model(
    model: nd.models.BaseModel,
    network: Union[nd.MultilayerNetwork, nd.TemporalNetwork],
    num_epochs: int,
):
    experiment = nd.Simulator(model=model, network=network)
    experiment.perform_propagation(n_epochs=num_epochs, patience=None)


def measure_time_single_run(function: Callable, input_data: Any) -> float:
    """Call 'function' with param 'input_list' and computes execution time."""
    start = time.time()
    function(*input_data)
    end = time.time()
    return (end - start) * 1000  # mili seconds


def time_complexity(
        function_to_analyse: Callable,
        data: List[DataSample],
        repetition_num: int = 5
) -> pd.DataFrame:
    """
    Measure time of execution of given function for given data.

    :param function_to_analyse: function to compute time complexity data for
    :param data: a dataset of varying complexity to perform experiment on.
    :param repetition_num: how many times computations will be repeated for
        each sample

    :return: Dataframe with generated data
    """
    execution_times = {}

    for sample in data:
        sample_times = {
            f"run_{i}": measure_time_single_run(function_to_analyse, sample.data)
            for i in range(repetition_num)
        }
        execution_times[sample.name] = sample_times

    return pd.DataFrame(execution_times).transpose()


def plot_time_efficiency(
        data: pd.DataFrame, data_name: str, axes: plt.Axes
) -> None:
    """
    Produce plot for measurement.

    The resulting plot is a curve of mean time for each sample bounded by its
    standard deviation.

    :param data: a dataframe with measurements (opuput of "time_complexity")
    :param data_name: name of the data
    :param axes: canvas to plot curves on
    """

    x = data.index.to_numpy()
    y = data.apply(lambda row: np.mean(row), axis=1).to_numpy()
    y_std = data.apply(lambda row: np.std(row), axis=1).to_numpy()

    axes.plot(
        x, y, '-o',
        label=f"'{data_name}'",
        alpha=0.9
    )
    axes.fill_between(x, y-y_std, y+y_std, alpha=0.2)
