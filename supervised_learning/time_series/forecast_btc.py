#!/usr/bin/env python3
"""
Training a RNN to predict BTC price using the data of a 24h day
"""
import numpy as np
from keras import Sequential
from keras.layers import Input, LSTM, Dense
from keras.callbacks import EarlyStopping


def load_data(npz_file):
    """
    Load the preprocessed time series data from an npz file.
    """
    data = np.load(npz_file)
    X = data['data']      # This is the input data (should be 3D)
    y = data['targets']   # This is the target data (closing price)

    # Assuming no separate val/test split was done in preprocessing, split here:
    train_split = int(0.7 * len(X))
    val_split = int(0.85 * len(X))

    X_train, y_train = X[:train_split], y[:train_split]
    X_val, y_val = X[train_split:val_split], y[train_split:val_split]
    X_test, y_test = X[val_split:], y[val_split:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_model(input_shape):
    """
    Build an RNN model using LSTM layers.
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def main():
    npz_file = "preprocessed_data_generator.npz"  # Adjust the file path accordingly

    # Load the preprocessed time series data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(npz_file)

    # Determine input shape based on X_train
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)

    # Build and compile the model
    model = build_model(input_shape)

    # Train the model
    model.fit(X_train, y_train, validation_data=(
        X_val, y_val), epochs=10, batch_size=32)

    # Evaluate on test data
    loss, mae = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}, Test MAE: {mae}')

    # Save the trained model
    model.save("btc_price_forecast_model.h5")


if __name__ == "__main__":
    main()
