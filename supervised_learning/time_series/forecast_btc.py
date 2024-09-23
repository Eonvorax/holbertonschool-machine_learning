#!/usr/bin/env python3
"""
Training a RNN to predict BTC price using the data of a 24h day
"""
import numpy as np
from keras import Sequential
from keras.layers import Input, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.regularizers import L1L2
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf


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
    X_scaled = scaler_X.fit_transform(
        X.reshape(-1, X.shape[-1])).reshape(X.shape)

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
        LSTM(60, kernel_regularizer=L1L2(1e-6, 1e-6),
             bias_regularizer=L1L2(1e-6, 1e-6)),
        Dense(1)
    ])

    model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    return model


def create_tf_dataset(X, y, batch_size=32, shuffle=True):
    """
    Convert NumPy arrays into a TensorFlow dataset and batch it.
    If shuffle is True, shuffle the dataset before batching.
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def main():
    # NOTE Adjust the file path accordingly
    npz_file = "preprocessed_data_raw.npz"
    batch_size = 32

    # Load the preprocessed time series data
    X, y = load_data(npz_file)

    # Rescale the data and retrieve scalers
    X_scaled, y_scaled, scaler_X, scaler_y = rescale_data(X, y)

    # Split the data into train/val/test sets
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X_scaled, y_scaled)

    # Create tf.data.Dataset objects
    train_dataset = create_tf_dataset(X_train, y_train,
                                      batch_size=batch_size, shuffle=True)
    val_dataset = create_tf_dataset(X_val, y_val,
                                    batch_size=batch_size, shuffle=False)
    test_dataset = create_tf_dataset(X_test, y_test,
                                     batch_size=batch_size, shuffle=False)

    # Determine input shape based on X_train
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)

    # Build and compile the model
    model = build_model(input_shape)

    early_stopping_callback = EarlyStopping(monitor="val_mae",
                                            patience=5,
                                            verbose=1,
                                            restore_best_weights=True)

    # Train the model using the Datasets
    model.fit(train_dataset, validation_data=val_dataset, epochs=10,
              callbacks=[early_stopping_callback])

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
    # NOTE why not take y_test ? Note sure this is needed
    y_test_rescaled = scaler_y.inverse_transform(
        np.concatenate([y for x, y in test_dataset], axis=0))
    y_pred_rescaled = scaler_y.inverse_transform(y_pred)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_rescaled, label="Actual Close Prices")
    plt.plot(y_pred_rescaled, label="Predicted Close Prices")
    plt.title("BTC Price Prediction vs Actual")
    plt.xlabel("Time (hours)")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.savefig("btc_price_prediction.png")
    plt.show()

    # Save the trained model
    model.save("btc_price_forecast_model.h5")


if __name__ == "__main__":
    main()
