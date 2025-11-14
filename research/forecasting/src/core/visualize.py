# ==========================================================================================
# Creation: 14/11/2025
# Last edit: 14/11/2025
# Author: Pablo González García.
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Optional, List, Tuple
# External:
from pandas import DataFrame
import matplotlib.pyplot as plt


# ==============================
# FUNCTIONS
# ==============================

def plot_forecast(
    context:DataFrame,
    predicted:DataFrame,
    ground_truth:Optional[DataFrame] = None,
    timestamp_column:str = "Timestamp",
    target_column:str = "Target",
    prediction_col:str = "predictions",
    quantiles:Optional[List] = [0.1, 0.9],
    history_tail:int = 256,
    figsize:Tuple = (12, 3),
    file_name:str = "plot.png"
) -> None:
    """
    Plots historizal, predicted, and optionally ground truth data for time series
    forecasting.

    Args:
        context (DataFrame): Historical data including timestamp and target.
        predicted (DataFrame): Forecast output from model.
        ground_truth (Optional[DataFrame]): Future true values for comparison.
        timestamp_column (str): Name ot timestamp column.
        target_column (str): Name of target column.
        predicion_col (str): Column in predicted with central prediction.
        quantiles (Optional[List]): Columns for predicion interval.
        history_tail (int): Number of historical points to show.
        figsize (Tuple): Figure size.
        file_name (str): Name of the file to save the plot.
    """

    # Sets index.
    ts_context = context.set_index(timestamp_column)[target_column].tail(history_tail)
    ts_pred = predicted.set_index(timestamp_column)

    # Creates the plot.
    ts_context.plot(label="historical data", color="xkcd:azure", figsize=figsize)

    # Checks if ground_truth was given.
    if ground_truth is not None:
        # Sets index.
        ts_ground_truth = ground_truth.set_index(timestamp_column)[target_column]
        # Creates the plot.
        ts_ground_truth.plot(label="future data (ground truth)", color="xkcd:grass green")
    
    # Creates the plot.
    ts_pred[prediction_col].plot(label="forecast", color="xkcd:violet")
    
    if quantiles and all(q in ts_pred.columns for q in map(str, quantiles)):
        plt.fill_between(
            ts_pred.index,
            ts_pred[str(quantiles[0])],
            ts_pred[str(quantiles[1])],
            alpha=0.7,
            label="prediction interval",
            color="xkcd:light lavender",
        )
    
    # Create the graphic and shows it.
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel(target_column)
    plt.savefig(file_name, dpi=150)
    plt.close()