#!/usr/bin/env python3
"""
Preprocessing the data
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


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
    # Merge the dataframes using 'Timestamp' column and adding suffixes to distinguish overlapping columns
    merged_df = pd.merge(df1, df2, on='Timestamp', how='outer', suffixes=('_cb', '_bs'))

    # For each feature except 'Timestamp', fill NaNs from Bitstamp with corresponding Coinbase data (and vice versa)
    for col in columns:
        if col != 'Timestamp':  # Skip 'Timestamp', as it won't have suffixes
            merged_df[col] = merged_df[f'{col}_cb'].combine_first(merged_df[f'{col}_bs'])

    # Keep only the original columns (we no longer need '_cb' and '_bs' suffixed columns)
    merged_df = merged_df[['Timestamp'] + columns[1:]]

    # Drop any rows that still have NaN values (optional, based on your requirement)
    merged_df = merged_df.dropna()

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

def create_time_series_data(df, past_window=1440, future_window=60):
    """
    Create input-output sequences for time series prediction.

    :param df: DataFrame with scaled BTC data.
    :param past_window: Number of minutes (rows) to use for past data (default: 1440 = 24 hours).
    :param future_window: Number of minutes (rows) to predict (default: 60 = 1 hour).
    :return: Input and target arrays for model training.
    """
    X, y = [], []

    for i in range(len(df) - past_window - future_window):
        # Input: past 24 hours of data
        X.append(df.iloc[i:i + past_window, 1:].values)
        # Output: next 60 minutes' close prices
        y.append(df.iloc[i + past_window:i + past_window + future_window]['Close'].values)

    return np.array(X), np.array(y)

def split_data(X, y, train_split=0.7, val_split=0.15):
    """
    Split the dataset into training, validation, and test sets.

    :param X: Input features.
    :param y: Target values.
    :param train_split: Fraction of data to use for training.
    :param val_split: Fraction of data to use for validation.
    :return: Split datasets (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    train_size = int(len(X) * train_split)
    val_size = int(len(X) * val_split)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    # Load the datasets
    df_coinbase = load_data("datasets\\coinbase.csv")
    df_bitstamp = load_data("datasets\\bitstamp.csv")

    # Extract the column names
    columns = df_coinbase.columns.tolist()

    # Merge the datasets and save a csv version of the merge
    merged_df = merge_datasets(df_coinbase, df_bitstamp, columns)
    # merged_df.to_csv("merged.csv", index=False)

    # Rescale the data
    merged_df, scaler = rescale_data(merged_df)

    # Create time series data (X, y)
    X, y = create_time_series_data(merged_df)

    # Split the data into training, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Save the preprocessed data to files
    np.savez("preprocessed_data.npz",
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test)

    print("Preprocessed data saved to 'preprocessed_data.npz'")


if __name__ == "__main__":
    main()
