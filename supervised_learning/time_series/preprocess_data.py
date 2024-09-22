#!/usr/bin/env python3
"""
Preprocessing the data for BTC price prediction.
"""

import pandas as pd
import numpy as np


def load_data(file_path, chunk_size=100000):
    """
    Load CSV data into a pandas DataFrame in chunks.
    """
    chunks = pd.read_csv(file_path, chunksize=chunk_size)
    df = pd.concat(chunks, ignore_index=True)
    return df


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


def create_time_series_data(df, target_column='Close', past_window=1440,
                            future_window=60):
    """
    Create input-output sequences for time series prediction.
    """
    data = df.drop(columns=['Timestamp']).values  # Remove 'Timestamp' column
    target = df[target_column].values  # Target variable (e.g., 'Close')

    X, y = [], []
    for i in range(0, len(data) - past_window - future_window, future_window):
        X.append(data[i:i + past_window])
        y.append(target[i + past_window + future_window])

    return np.array(X, dtype="float32"), np.array(y, dtype="float32")


def remove_first_half(df):
    """
    Remove the first half of the dataset.
    """
    total_rows = len(df)
    df = df.iloc[total_rows // 2:].reset_index(drop=True)
    return df


def subsample_data(df, freq=10):
    """
    Subsample the data by taking one row every `freq` minutes.
    """
    df = df.iloc[::freq].reset_index(drop=True)
    return df


def main():
    # Load the datasets
    # NOTE adjust the file paths accordingly, these are for Colab
    df_coinbase = load_data("/content/drive/MyDrive/datasets/coinbase.csv")
    df_bitstamp = load_data("/content/drive/MyDrive/datasets/bitstamp.csv")

    # Extract the column names
    columns = df_coinbase.columns.tolist()

    # Remove the first half of the data (to avoid early NaN-heavy data)
    df_coinbase = remove_first_half(df_coinbase)
    df_bitstamp = remove_first_half(df_bitstamp)

    # Merge the datasets
    merged_df = merge_datasets(df_coinbase, df_bitstamp, columns)

    # Create time series data
    X, y = create_time_series_data(merged_df)

    # Save the data in chunks to avoid memory issues
    np.savez_compressed("preprocessed_data_raw.npz", data=X, targets=y)

    print(f"Data saved: X shape {X.shape}, y shape {y.shape}")


if __name__ == "__main__":
    main()
