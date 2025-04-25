from taba.utils import denormalize, normalize
import numpy as np
import pandas as pd
import keras.optimizers as optimizers  # type: ignore
from tensorflow.keras import models  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Activation, Flatten  # type: ignore


class LSTMModel:
    def __init__(
        self,
        data,
        n_past=24,
        n_future=6,
        batch_size=16,
        epochs=20,
        learning_rate=0.001,
        hidden_units=128,
        train_split=0.7,
        activation="relu",
    ):
        """
        Initialize the LSTM model with the given parameters.
        """
        self.data = data  # Pandas DataFrame containing the data
        self.n_past = (
            n_past  # Number of past time steps to consider for each prediction
        )
        self.n_future = n_future  # Number of time steps ahead in the future to predict
        self.scaler = None
        self.model = None
        self.model_history = None
        # RNN Hyperparameters
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.LEARNING_RATE = learning_rate
        self.HIDDEN_UNITS = hidden_units  # Number of LSTM units in the hidden layer
        self.TRAIN_SPLIT = train_split  # Fraction of data to use for training
        self.activation = activation  # Activation function for the output layer

    def initialize_vectors(self):
        """
        Initialize the training and testing vectors for the LSTM model.
        The data is split into training and testing sets based on the TRAIN_SPLIT parameter.
        The training set is further split into sequences of past data (X) and future data (y).
        """
        # Normalize the data
        scaled_data, self.scaler = normalize(
            self.data.drop(["datetime"], axis=1, inplace=False),
            column_wise=True,
        )
        # scaled_data = np.array(self.data.drop(['datetime'], axis=1, inplace=False), dtype=np.float32)

        X, y = self.create_sequences(scaled_data)
        # Splitting the data into train and test sets
        train_size = int(len(X) * self.TRAIN_SPLIT)
        self.X_train, self.X_test = np.vsplit(X, [train_size])
        self.y_train, self.y_test = y[:train_size], y[train_size:]

        return self.X_train, self.y_train, self.X_test, self.y_test

    def create_sequences(self, data):
        """
        Create sequences of data for LSTM training.
        The sequences are created by taking the last n_past time steps and predicting the value at n_future time steps ahead.
        The data is reshaped to be in the format (samples, time steps, features)."""

        # Calculate how many valid samples we can create
        valid_samples = data.shape[0] - self.n_past - self.n_future
        price_column = 3  # Assuming the close price is in the 4th column (index 3)

        # Initialize output arrays
        X = np.zeros((valid_samples, self.n_past, data.shape[1]))
        y = np.zeros(valid_samples)

        # For each feature (open, high, low, close, volume)
        for i in range(data.shape[1]):
            sample_idx = 0
            # For each valid sample
            for j in range(self.n_past, data.shape[0] - self.n_future):
                # Append the last n_past time steps to the input sequence
                X[sample_idx, :, i] = data[j - self.n_past : j, i]
                # Set target with the close price of the future time step
                y[sample_idx] = data[j + self.n_future, price_column]

                sample_idx += 1

        return X, y

    def model_training(self):
        """
        Train the LSTM model using the created sequences.
        Returns the trained model and the training history.
        """
        # Define the model
        self.model = Sequential()
        self.model.add(Input(shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(LSTM(self.HIDDEN_UNITS, return_sequences=True))
        self.model.add(LSTM(int(self.HIDDEN_UNITS / 2), return_sequences=False))
        self.model.add(Dense(1, activation=self.activation))

        # Compile the model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.LEARNING_RATE),
            loss="mse",
            metrics=["mae", "mape", "r2_score"],
        )

        # Train the model
        self.model_history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            validation_split=0.1,
            verbose=1,
        )
        return self.model, self.model_history

    def model_evaluation(self, print_metrics=True):
        """
        Evaluate the model on the test set.
        The evaluation metrics and the predictions are returned.
        """
        # Evaluate the model
        loss_metrics = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        if print_metrics:
            # print(
            #    f"Train Loss: {self.model.evaluate(self.X_train, self.y_train, verbose=0)}"
            # )
            print(
                f"Test Loss: {loss_metrics[0]}, Test MAE: {loss_metrics[1]}, Test MAPE: {loss_metrics[2]}, Test R2 Score: {loss_metrics[3]}"
            )

        # Make predictions
        test_pred = self.model.predict(self.X_test, verbose=0)
        train_pred = self.model.predict(self.X_train, verbose=0)
        test_pred = denormalize(test_pred, self.scaler[3][0], self.scaler[3][1])
        train_pred = denormalize(train_pred, self.scaler[3][0], self.scaler[3][1])

        return train_pred, test_pred, loss_metrics

    def predict_future(self, data):
        """
        Predict the future values using the trained model.
        The predictions are returned as a Numpy array.
        """
        self.data = data
        # Normalize the data
        scaled_data, self.scaler = normalize(
            data.drop([0], axis=1, inplace=False), column_wise=True
        )

        # Create sequences for prediction
        X_future = np.zeros(
            (scaled_data.shape[0] - self.n_past, self.n_past, scaled_data.shape[1])
        )

        for i in range(scaled_data.shape[1]):
            sample_idx = 0
            for j in range(self.n_past, scaled_data.shape[0]):
                X_future[sample_idx, :, i] = scaled_data[j - self.n_past : j, i]
                sample_idx += 1

        future_pred = self.model.predict(X_future)
        future_pred = denormalize(future_pred, self.scaler[3][0], self.scaler[3][1])

        return future_pred

    def save_model(self, model_path):
        # Save the trained model
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path):
        # Load a pre-trained model
        self.model = models.load_model(model_path)
        print(f"Model loaded from {model_path}")
