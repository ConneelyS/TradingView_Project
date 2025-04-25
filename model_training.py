from taba.db_handler import DBHandler
from taba.model import LSTMModel
from taba.config import *
import pandas as pd
import os


def preprocess(db):
    """
    Preprocess the data by merging the stock data with gold and US10Y data.
    The data is merged on the datetime column and the fed rate is added to the dataframe.
    The fed rate is interpolated to fill in missing values.
    """

    for name in [TICKER, "US10", "GOLD"]:
        data = db.get_data(name + "_HIST")
        df_temp = pd.DataFrame(
            data,
            columns=["datetime", "symbol", "open", "high", "low", "close", "volume"],
        )
        df_temp.drop(columns=["symbol", "volume"], inplace=True)
        # If the df is empty, set it to df_temp
        # Otherwise, merge the two dataframes on the datetime column
        df = (
            df_temp
            if name == TICKER
            else pd.merge(
                df,
                df_temp,
                how="inner",
                on="datetime",
                suffixes=(None, "_" + name.lower()),
            )
        )

    fed = db.get_data("FED")
    fed_df = pd.DataFrame(fed, columns=["date", "rate"])

    df["datetime"] = pd.to_datetime(df["datetime"])
    fed_df["date"] = pd.to_datetime(fed_df["date"])

    # Add the fed rate to the dataframe
    df = pd.merge(
        df,
        fed_df,
        left_on="datetime",
        right_on="date",
        how="outer",
        suffixes=(None, "_fed"),
    )
    # interpolate the fed rate if closest available rate before (since the fed rate is updated monthly)
    df["rate"] = df["rate"].interpolate(method="pad")

    # Drop the date column as it is no longer needed
    df.drop(columns=["date"], inplace=True)
    # Drop rows that were added on the merge (outer join)
    df.dropna(inplace=True)

    # Save the processed data to the database
    db.save_df_to_db(df, "PROCESSED_DATA")

    return df


def hyperparameter_tuning(db_handler, data):
    """
    Perform hyperparameter tuning for the LSTM model.
    This function will iterate over a range of hyperparameters and evaluate the model performance.
    """

    best_mse = float("inf")
    best_params = {}
    for batch_size in [16, 32, 64]:
        for learning_rate in [0.0001, 0.001, 0.01]:
            for activation in ["relu", "tanh", "sigmoid"]:
                for hidden_units in [32, 64, 96, 128, 256]:
                    print(
                        f"Training with batch_size={batch_size}, learning_rate={learning_rate}, activation={activation}, hidden_units={hidden_units}"
                    )
                    lstm_model = LSTMModel(
                        data,
                        TIME_STEPS,
                        TIME_IN_FUTURE,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        hidden_units=hidden_units,
                        activation=activation,
                    )
                    lstm_model.initialize_vectors()
                    lstm_model.model_training()
                    eval = lstm_model.model_evaluation()[2]
                    test_mse = eval[0]
                    if test_mse < best_mse:
                        best_mse = test_mse
                        best_params = {
                            "batch_size": batch_size,
                            "hidden_units": hidden_units,
                            "learning_rate": learning_rate,
                            "activation": activation,
                        }
                    lstm_model.save_model(
                        f"models/model_sp246_bs{batch_size}_lr{learning_rate}_act{activation}.keras"
                    )
                    db_handler.save_model(
                        f"model_sp246_bs{batch_size}_lr{learning_rate}_act{activation}.keras",
                        TIME_STEPS,
                        TIME_IN_FUTURE,
                        test_mse,
                    )

    print(f"Best MSE: {best_mse} with parameters: {best_params}")


def main():
    # Initialize the database handler
    try:
        db_handler = DBHandler(DB_PATH)
    except Exception as e:
        print(f"Error initializing database handler: {e}")
        return

    try:
        # Check if the database is empty and load data if necessary
        if not db_handler.get_data("SP500_HIST"):
            print("Database is empty. Loading data from CSV files...")
            # Load data from CSV to database
            for file in os.listdir(DATA_FOLDER):
                print(f"Loading {file} into the database...")
                db_handler.load_csv_to_db(os.path.join(DATA_FOLDER, file))
            db_handler.create_tables()
    except Exception as e:
        print(f"Error loading data into database: {e}")
        return

    data = preprocess(db_handler)
    if PREPROCESS_ONLY:
        print("Preprocessing completed. Exiting...")
        return

    # single model training
    if MODE == "manual":
        lstm_model = LSTMModel(data, TIME_STEPS, TIME_IN_FUTURE)
        lstm_model.initialize_vectors()
        lstm_model.model_training()
        eval = lstm_model.model_evaluation()
        lstm_model.save_model("models/best_model.keras")

        db_handler.save_model(
            "best_model.keras",
            TIME_STEPS,
            TIME_IN_FUTURE,
            eval[2][0],
        )
    else:
        # Perform hyperparameter tuning
        hyperparameter_tuning(db_handler, data)

    # Close the database connection
    db_handler.close()


if __name__ == "__main__":
    main()
