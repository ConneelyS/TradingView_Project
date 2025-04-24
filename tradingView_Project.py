from flask import Flask
import sqlite3
import pandas as pd
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

# Dash Application Code
# Layout
app.layout = html.Div(id='header_wrapper_div', style={'backgroundColor': '#1e1e1e', 'color': '#ffffff', 'padding': '20px'}, children=[
    html.H1("Stock Viewer App", style={'textAlign': 'center'}),
    
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
        dcc.Tab(label='Volume Chart', value='volume', style={'backgroundColor': '#2c2c2c', 'color': '#fff'}),
        dcc.Tab(label='Moving Averages', value='moving_average', style={'backgroundColor': '#2c2c2c', 'color': '#fff'}),
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
    else:
        return html.Div("Unknown chart type.")

    fig.update_layout(margin=dict(l=40, r=20, t=40, b=40))
    return dcc.Graph(figure=fig)

# Flask Application Code
# Run the server
if __name__ == '__main__':
    app.run_server(debug=True)
