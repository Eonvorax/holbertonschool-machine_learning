#!/usr/bin/env python3
"""
Preprocessing the data for BTC price prediction using Keras TimeseriesGenerator.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator


def load_data(file_path):
    """
    Load CSV data into a pandas DataFrame.
    """
    return pd.read_csv(file_path)


def merge_datasets(df1: pd.DataFrame, df2: pd.DataFrame, columns):
    """
    Merge two datasets based on 'Timestamp' using an outer join
    and fill missing values by preferring values from df1 where available.
    """
    merged_df = pd.merge(df1, df2, on='Timestamp',
                         how='outer', suffixes=('_cb', '_bs'))
    for col in columns:
        if col != 'Timestamp':  # Skip 'Timestamp', as it won't have suffixes
            merged_df[col] = merged_df[f'{col}_cb'].combine_first(
                merged_df[f'{col}_bs'])
    merged_df = merged_df[['Timestamp'] + columns[1:]]  # Keep original columns
    merged_df = merged_df.dropna()  # Drop rows with missing values
    return merged_df


def rescale_data(df):
    """
    Rescale the features using MinMaxScaler to range [0, 1].
    """
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(
        df.iloc[:, 1:])  # Exclude the 'Timestamp' column
    df.iloc[:, 1:] = scaled_values
    return df, scaler


def create_time_series_data_with_generator(df, target_column='Close', past_window=1440, future_window=60, batch_size=32):
    """
    Use Keras TimeseriesGenerator to create input-output sequences for time series prediction.

    :param df: DataFrame with scaled BTC data.
    :param target_column: The target column to predict (e.g., 'Close').
    :param past_window: Number of minutes (rows) to use for past data (default: 1440 = 24 hours).
    :param future_window: Number of minutes (rows) to predict (default: 60 = 1 hour).
    :param batch_size: Number of samples per batch (default: 32).
    :return: Keras TimeseriesGenerator for training data.
    """
    data = df.drop(columns=['Timestamp']).values  # Remove 'Timestamp' column
    target = df[target_column].values  # Target variable (e.g., 'Close')

    generator = TimeseriesGenerator(
        data, target,
        length=past_window,  # Past 24 hours of data
        sampling_rate=1,
        stride=1,
        batch_size=batch_size,
        # Ensure we have enough future data
        end_index=len(data) - future_window
    )

    return generator


def remove_first_half(df):
    """
    Remove the first half of the dataset.
    """
    total_rows = len(df)
    df = df.iloc[total_rows // 2:].reset_index(drop=True)
    return df


def main():
    # Load the datasets
    df_coinbase = load_data("datasets/coinbase.csv")
    df_bitstamp = load_data("datasets/bitstamp.csv")

    # Extract the column names
    columns = df_coinbase.columns.tolist()

    # Remove the first half of the data (to avoid early NaN-heavy data)
    df_coinbase = remove_first_half(df_coinbase)
    df_bitstamp = remove_first_half(df_bitstamp)

    # Merge the datasets
    merged_df = merge_datasets(df_coinbase, df_bitstamp, columns)

    # Rescale the data
    merged_df, scaler = rescale_data(merged_df)

    # Create time series generator using Keras's TimeseriesGenerator
    generator = create_time_series_data_with_generator(merged_df)

    # Save the generator for future use (optional, if you plan to save it or use it later)
    np.savez("preprocessed_data_generator.npz",
             data=generator.data, targets=generator.targets)

    print(f"Generated {len(generator)} time-series samples using Keras TimeseriesGenerator.")


if __name__ == "__main__":
    main()
