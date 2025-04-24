from flask import Flask
import sqlite3
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.express as px

# Initialize Flask app
server = Flask(__name__)

# Initialize Dash app with Flask server
app = dash.Dash(__name__, server=server, url_base_pathname='/', suppress_callback_exceptions=True)

# Path to your SQLite3 database
DB_PATH = 'database/tradingView.db'

# Function to get available stock symbols
def get_stock_symbols():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT DISTINCT symbol FROM trades", conn)
    return df['symbol'].sort_values().tolist()

# Function to query stock data by symbol
def query_stock_data(symbol):
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            "SELECT datetime, close FROM trades WHERE symbol = ? ORDER BY datetime",
            conn, params=(symbol,)
        )
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

# Layout
app.layout = html.Div([
    html.H1("Stock Price Viewer", style={'textAlign': 'center'}),
    
    html.Div([
        html.Label("Select a stock:"),
        dcc.Dropdown(
            id='symbol-dropdown',
            options=[{'label': sym, 'value': sym} for sym in get_stock_symbols()],
            value=None,
            placeholder='Choose a symbol...'
        ),
    ], style={'width': '300px', 'margin': 'auto'}),

    dcc.Graph(id='price-graph')
])

# Callback to update graph
@app.callback(
    Output('price-graph', 'figure'),
    Input('symbol-dropdown', 'value')
)

def update_graph(symbol):
    if symbol is None:
        return {}
    df = query_stock_data(symbol)
    fig = px.line(df, x='datetime', y='close', title=f"{symbol} Closing Prices")
    fig.update_layout(transition_duration=500)
    return fig

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True)
