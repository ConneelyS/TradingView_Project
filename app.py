from taba.db_handler import DBHandler
from taba.model import LSTMModel
from taba.config import *
import pandas as pd
import os


def main():
    # Initialize the database handler
    db_handler = DBHandler(DB_PATH)
    db_handler.create_tables()

    # Check if the database is empty and load data if necessary
    if not db_handler.get_data(STOCK_TICKER):
        print("Database is empty. Loading data from CSV files...")
        # Load data from CSV to database
        for file in os.listdir(DATA_FOLDER):
            if file.endswith(".csv"):
                print(f"Loading {file} into the database...")
                db_handler.load_csv_to_db(os.path.join(DATA_FOLDER, file))

    # Get the data from the database
    data = db_handler.get_data(STOCK_TICKER)
    db_handler.close()

    # Convert the data to a DataFrame
    df = pd.DataFrame(
        data, columns=["datetime", "symbol", "open", "high", "low", "close", "volume"]
    )
    print(df.head())

    lstm_model = LSTMModel(df, TIME_STEPS, TIME_IN_FUTURE)

    if MODE == "train":
        lstm_model.initialize_vectors()
        lstm_model.model_training()
        lstm_model.model_evaluation()
        lstm_model.save_model("models/model_nvda.keras")
    else:
        lstm_model.load_model(BEST_MODEL)
        lstm_model.predict_future(df)

    # Close the database connection
    db_handler.close()


if __name__ == "__main__":
    main()
