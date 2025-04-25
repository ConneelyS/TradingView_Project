import sqlite3
import pandas as pd


class DBHandler:
    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()

    def create_tables(self):
        """
        Create empty additional tables in the database if they do not exist
        """
        self.cursor.execute(  # Table for storing models, parameters, and MSE values
            """
            CREATE TABLE IF NOT EXISTS MODELS (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                time_steps INTEGER NOT NULL,
                time_in_future INTEGER NOT NULL,
                mse REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        self.connection.commit()

    def load_csv_to_db(self, file):
        """
        Load CSV data into the database
        """
        data = pd.read_csv(file)
        table = file.split("/")[-1].split(".")[0].upper()
        data.to_sql(table, self.connection, if_exists="replace", index=False)
        self.connection.commit()

    def save_df_to_db(self, df, table_name):
        """
        Save a DataFrame to the database
        """
        print(f"Saving DataFrame to table {table_name}...")
        df.to_sql(table_name, self.connection, if_exists="replace", index=False)
        self.connection.commit()

    def get_data(self, table_name):
        try:
            print(f"Fetching data from table {table_name}...")
            self.cursor.execute(f"SELECT * FROM {table_name}")
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
            return None

    def save_model(self, model_name, time_steps, time_in_future, mse):
        """
        Save the model name and parameters to the database
        """
        self.cursor.execute(
            """
            INSERT INTO MODELS (model_name, time_steps, time_in_future, mse)
            VALUES (?, ?, ?, ?)
            """,
            (model_name, time_steps, time_in_future, mse),
        )
        self.connection.commit()

    def close(self):
        self.connection.close()
