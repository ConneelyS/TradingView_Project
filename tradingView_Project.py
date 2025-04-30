from flask import Flask
import sqlite3
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

# Initialize Flask app
server = Flask(__name__)

# Initialize Dash app with Flask server
app = dash.Dash(__name__, server=server, url_base_pathname='/', suppress_callback_exceptions=True)

# Dataabse Code
# Path to your SQLite3 database
DB_PATH = 'database/tradingView.db'
database_path = 'database/taba.db'

# Function to get available stock symbols
def get_stock_symbols():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT DISTINCT symbol FROM trades", conn)
        # print(df['symbol'].sort_values().tolist())
    return df['symbol'].sort_values().tolist()

# Function to query stock data by symbol
def query_stock_data(symbol):
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            "SELECT datetime, open, high, low, close, volume FROM trades WHERE symbol = ? ORDER BY datetime",
            conn, params=(symbol,)
        )
    df['datetime'] = pd.to_datetime(df['datetime'])
    # print(df)
    return df

# *** New Processed Data From LSTM Model ***
def query_processed_stock_data():
    with sqlite3.connect(database_path) as conn:
        df_new = pd.read_sql(
            "SELECT datetime, close, close_us10, close_gold FROM PROCESSED_DATA ORDER BY datetime",
            conn
        )
    df_new['datetime'] = pd.to_datetime(df_new['datetime'])
    # print(df_new)
    return df_new

# Dash Application Code
# Layout
app.layout = html.Div(id='header_wrapper_div', style={'backgroundColor': '#1e1e1e', 'color': '#ffffff', 'padding': '20px'}, children=[
    html.H1("Stock Price Viewer App", style={'textAlign': 'center'}),
    
    html.Div([
        html.Label("Select Stock Symbol:"),
        dcc.Dropdown(
            id='symbol-dropdown',
            options=[{'label': sym, 'value': sym} for sym in get_stock_symbols()],
            value=None,
            placeholder='Choose a symbol..',
            style={'color': '#000000'}
        ),
    ], style={'width': '300px', 'margin': 'auto'}),

    dcc.Tabs(id='tabs', value='line', children=[
        dcc.Tab(label='Line Chart', value='line', style={'backgroundColor': '#2c2c2c', 'color': '#fff'}),
        dcc.Tab(label='Candlestick Chart', value='candle', style={'backgroundColor': '#2c2c2c', 'color': '#fff'}),
        # dcc.Tab(label='Volume Chart', value='volume', style={'backgroundColor': '#2c2c2c', 'color': '#fff'}),
        dcc.Tab(label='MACD and Signal Line', value='macd', style={'backgroundColor': '#2c2c2c', 'color': '#fff'}),
        dcc.Tab(label='Moving Averages', value='moving_average', style={'backgroundColor': '#2c2c2c', 'color': '#fff'}),
        # dcc.Tab(label='LSTM Predictions', value='predictions', style={'backgroundColor': '#2c2c2c', 'color': '#fff'}),
    ], style={'marginTop': '20px'}),

    html.Div(id='price-graph')
])

# Callback to update graph
@app.callback(
    Output('price-graph', 'children'),
    Input('symbol-dropdown', 'value'),
    Input('tabs', 'value')
)

def update_chart(symbol, chart_type):
    # Asks the user to pick a stock
    if symbol is None:
        return html.Div("Select a stock symbol to display charts.", style={'textAlign': 'center'})

    df = query_stock_data(symbol)
    df_new = query_processed_stock_data()

    # Generate Line Chart
    if chart_type == 'line':
        fig = px.line(df, x='datetime', y='close', title=f'{symbol} Closing Price', template='plotly_dark')

    # Candlestick Chart
    elif chart_type == 'candle':
        fig = go.Figure(data=[go.Candlestick(
            x=df['datetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close']
        )])
        fig.update_layout(title=f'{symbol} Candlestick Chart', template='plotly_dark')

    # Bar Chart for Volume of stock
    elif chart_type == 'volume':
        fig = px.bar(df, x='datetime', y='volume', title=f'{symbol} Trading Volume', template='plotly_dark')
    
    # MACD and Signal Line plot
    elif chart_type == 'macd':
        df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x = df['datetime'], y = df['MACD'], mode = 'lines', name = 'MACD', line = dict(color = 'cyan')
        ))

        fig.add_trace(go.Scatter(
            x = df['datetime'], y = df['MACD_signal'], mode = 'lines', name = 'Signal Line', line = dict(color = 'orange')
        ))

        fig.update_layout(
            title = f'{symbol} MACD and Signal Line',
            template = 'plotly_dark',
            legend = dict(x = 0, y = 1)
        )

    # Rolling Averages Chart
    elif chart_type == 'moving_average':
        df['SMA_7'] = df['close'].rolling(window=7).mean()
        df['SMA_30'] = df['close'].rolling(window=30).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['close'], mode='lines', name='Close', line=dict(color='lightgray')))
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['SMA_7'], mode='lines', name='7-day SMA', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['SMA_30'], mode='lines', name='30-day SMA', line=dict(color='deepskyblue')))
        fig.update_layout(
            title=f'{symbol} Closing Price with Moving Averages',
            template='plotly_dark',
            legend=dict(x=0, y=1)
        )
    
    # LSTM predictions work - Values are not current Trained Model predictions
    elif chart_type == 'predictions':

        df['datetime'] = pd.to_datetime(df['datetime'])
        df_filtered = df[df['datetime'] >= '2024-01-01 02:00:00']
        # print(df_filtered)

        fig = go.Figure()
        if symbol =='NASDAQ:NVDA':
            fig.add_trace(go.Scatter(x=df_filtered['datetime'], y=df_filtered['close'], mode='lines', name='Close_NVDA', line=dict(color='orange')))
        if symbol =='TVC:US10Y':
            fig.add_trace(go.Scatter(x=df_filtered['datetime'], y=df['close'], mode='lines', name='Close_US10', line=dict(color='orange')))
        if symbol =='SP:SPX':
            fig.add_trace(go.Scatter(x=df_filtered['datetime'], y=df['close'], mode='lines', name='Close_S&P500', line=dict(color='orange')))
        if symbol =='TVC:GOLD':
            fig.add_trace(go.Scatter(x=df_filtered['datetime'], y=df['close'], mode='lines', name='Close_Gold', line=dict(color='orange')))

        # Data points for each of the predicted values
        fig.add_trace(go.Scatter(x=df_new['datetime'], y=df_new['close_us10'], mode='lines', name='US10_Predicted_Close', line=dict(color='lightgray')))
        fig.add_trace(go.Scatter(x=df_new['datetime'], y=df_new['close'], mode='lines', name='S&P500_Predicted_Close', line=dict(color='lightgray')))
        fig.add_trace(go.Scatter(x=df_new['datetime'], y=df_new['close_gold'], mode='lines', name='Gold_Predicted_Close', line=dict(color='lightgray')))

        fig.update_layout(
            title=f'LSTM Predicted Closing Values',
            template='plotly_dark',
            legend=dict(x=0, y=1)
        )

    else:
        return html.Div("Unknown chart type.")

    fig.update_layout(margin=dict(l=40, r=20, t=40, b=40))
    return dcc.Graph(figure=fig)

# Flask Application Code
# Run the server
if __name__ == '__main__':
    app.run(debug=True)
