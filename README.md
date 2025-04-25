# Predicting SP500 with LSTM Model

The main goal is to predict the SP500 close price using a LSTM model that intake the OHLC values (Open, High, Low, Close) from SP500, GOLD, US10Y indexes and Federal Reserva interest rates as features.

### Database

All database operations are performed in SQLite

### Installation

```bash
pip install -e .
```

### Usage

For training the model, run the following command:

```bash
python model_training.py
```

For running predictions, with processed data (test or new data), run the following command:

```bash
python app.py
```

### Data

The data is downloaded from Trading View using the `tradingview-datafeed` library. The data is stored in a SQLite database. The database is created in the `database` folder. The data is downloaded and processed in the `model_training.py` file. The processed data is stored in the SQLite database.

The data is downloaded from the following sources:

- SP500: https://br.tradingview.com/chart/?symbol=VANTAGE%3ASP500
- GOLD: https://br.tradingview.com/chart/?symbol=TVC%3AGOLD
- US10Y: https://br.tradingview.com/chart/?symbol=TVC%3AUS10Y
- Federal Reserve Interest Rates: https://fred.stlouisfed.org/series/FEDFUNDS
