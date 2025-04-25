import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from taba.model import LSTMModel
from taba.db_handler import DBHandler
from taba.config import *


def main():
    """
    # Initialize the database handler
    # To predict the future, we need to load the data from the database
    # If data is stored in PROCESSED_DATA table, we can use it directly
    # If not, we need to load the data and preprocess it
    # Data = Dataframe (SP500_HIST + US10_HIST + GOLD_HIST + FED_HIST)
    """
    db = DBHandler(DB_PATH)
    df = pd.DataFrame(db.get_data("PROCESSED_DATA"))
    db.close()

    # Initialize the LSTM model with the stock parameters
    model = LSTMModel(None, TIME_STEPS, TIME_IN_FUTURE)

    # Using test data as example for prediction
    df = df.iloc[int(df.shape[0] * model.TRAIN_SPLIT) :, :]

    # Loading the best model and making predictions
    model.load_model("models/" + BEST_MODEL)
    predictions = model.predict_future(df)

    plt.figure(figsize=(20, 10))
    plt.plot(predictions, label="Predictions")
    # True values are in the 4th column of the dataframe = Close price
    plt.plot(df.iloc[:, 3].values, label="True Values")
    plt.title("Predictions vs True Values")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
