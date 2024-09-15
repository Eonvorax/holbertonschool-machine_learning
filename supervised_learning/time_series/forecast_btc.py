#!/usr/bin/env python3
"""
Training a RNN to predict BTC price using the data of a 24h day
"""
import numpy as np
from keras import Sequential
from keras.layers import Input, LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def load_data(npz_file):
    """
    Load the preprocessed time series data from an npz file.
    """
    data = np.load(npz_file)
    X = data['data']      # This is the input data (should be 3D)
    y = data['targets']   # This is the target data (closing price)

    return X, y


def rescale_data(X, y):
    """
    Rescale both the input data X and target data y using MinMaxScaler.
    Returns the scaled data along with the scalers, for reversing the
    transformation later.
    """
    # Rescale X (input features)
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # Rescale y (target)
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    return X_scaled, y_scaled, scaler_X, scaler_y


def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    Split the data into training, validation, and test sets.
    """
    train_split = int(train_ratio * len(X))
    val_split = int((train_ratio + val_ratio) * len(X))

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
    npz_file = "preprocessed_data_raw.npz"  # NOTE Adjust the file path accordingly

    # Load the preprocessed time series data
    X, y = load_data(npz_file)

    # Rescale the data and retrieve scalers
    X_scaled, y_scaled, scaler_X, scaler_y = rescale_data(X, y)

    # Split the data into train/val/test sets
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X_scaled, y_scaled)

    # Determine input shape based on X_train
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)

    # Build and compile the model
    model = build_model(input_shape)

    # Train the model
    model.fit(X_train, y_train, validation_data=(
        X_val, y_val), epochs=5, batch_size=32)

    # Evaluate on test data
    loss, mae = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}, Test MAE: {mae}')

    # Save the trained model
    model.save("btc_price_forecast_model.h5")

    # Save the scalers for later use, just a precaution
    np.savez("scalers.npz", scaler_X=scaler_X, scaler_y=scaler_y)

    # Make predictions
    y_pred = model.predict(X_test)

    # Scale the predictions and actual values back to the original scale
    y_pred_rescaled = scaler_y.inverse_transform(y_pred)
    y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_rescaled, label="Actual Close Prices")
    plt.plot(y_pred_rescaled, label="Predicted Close Prices")
    plt.title("BTC Price Prediction vs Actual")
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.savefig("btc_price_prediction.png")
    plt.show()

    # Save the trained model
    model.save("btc_price_forecast_model.h5")


if __name__ == "__main__":
    main()
