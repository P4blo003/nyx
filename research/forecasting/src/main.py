# ==========================================================================================
# Creation: 14/11/2025
# Last edit: 14/11/2025
# Author: Pablo González García.
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import List
# External:
import pandas as pd
from pandas import DataFrame
# Internal:
from models.base_model import BaseForecastingModel
from models.chronos_wrapper import ChronosWrapper
from core.visualize import plot_forecast


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    # Crates the wrapper.
    wrapper:BaseForecastingModel = ChronosWrapper()

    # Read a test csv.
    df = pd.read_csv(
            r"data\raw\household_electric_power_consumption.csv",
            parse_dates=["Timestamp"],
        )
    
    # Parse Timstamp.
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True)

    # Convert categorical columns to categories.
    categorical_cols = ["SM_1", "SM_2", "SM_3"]
    for col in categorical_cols:
        df[col] = df[col].astype("category")
    
    # Sets more variables.
    target_column:str = "GAP"
    timestamp_column:str = "Timestamp"
    id_column:str = "id"

    # Sets id for chronos.
    df[id_column] = "series1"

    # Features.
    features_cols:List[str] = [
        "GRP", "V",
        "GI", "SM_1",
        "SM_2", "SM_3"
    ]

    # Split.
    X = df[features_cols + [target_column, id_column, timestamp_column]]
    Y = df[target_column]

    # Create train and test dataframes.
    split_size:int = 96
    train_df:DataFrame = df.iloc[:-split_size]
    test_df:DataFrame = df.iloc[-split_size:]
    future_df:DataFrame = test_df.drop(columns=[target_column])

    # Train the model.
    wrapper.fit(x=train_df, y=train_df[target_column])

    # Predict.
    predicted_df:DataFrame = wrapper.predict(
        horizon=96,
        futures=future_df,
        id_column=id_column,
        timestamp_column=timestamp_column,
        target_column=target_column
    )

    # Visualizes the prediction.
    plot_forecast(
        context=train_df,
        predicted=predicted_df,
        ground_truth=test_df,
        timestamp_column=timestamp_column,
        target_column=target_column
    )