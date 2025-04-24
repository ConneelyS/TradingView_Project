import sqlite3
import pandas as pd


class DBHandler:
    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()

    def create_tables(self):
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS NVDA_HIST (
                datetime TEXT NOT NULL,
                symbol TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )"""
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS US10_HIST (
                datetime TEXT NOT NULL,
                symbol TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )"""
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS SP500_HIST (
                datetime TEXT NOT NULL,
                symbol TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )"""
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS GOLD_HIST (
                datetime TEXT NOT NULL,
                symbol TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )"""
        )
        self.connection.commit()

    def load_csv_to_db(self, file):
        data = pd.read_csv(file)
        table = file.split("/")[-1].split(".")[0] + "_HIST"
        data.to_sql(table, self.connection, if_exists="append", index=False)
        self.connection.commit()

    def get_data(self, stock_ticker):
        self.cursor.execute(f"SELECT * FROM NVDA_HIST")
        return self.cursor.fetchall()

    def close(self):
        self.connection.close()
