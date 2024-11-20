from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from plotly.subplots import make_subplots
from collections import defaultdict

import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd

from ..config import API_KEY, SECRET_KEY

# Create a client object
client = StockHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)


def get_rsi(df: pd.DataFrame, rsi_window: int) -> pd.DataFrame:
    """
    Calculates the Relative Strength Index (RSI) for a given DataFrame.

    Parameters
        df: pd.DataFrame
            A DataFrame containing at least a 'close' column with stock prices.
        rsi_window: int
            The number of periods to use in the RSI calculation.

    Return
        df: pd.DataFrame
            The input DataFrame with an added 'rsi' column containing the RSI values.
    """
    indicator_rsi = RSIIndicator(close=df["close"], window=rsi_window)
    df["rsi"] = indicator_rsi.rsi()
    return df


def get_bb(df: pd.DataFrame, bb_window: int) -> pd.DataFrame:
    """
    Calculates Bollinger Bands and related indicators for a given DataFrame.

    Parameters
        df: pd.DataFrame
            A DataFrame containing at least a 'close' column with stock prices.
        bb_window: int
            The number of periods to use in the Bollinger Bands calculation.

    Return
        df: pd.DataFrame
            The input DataFrame with added columns for Bollinger Bands and indicators.
    """
    # calculate bollinger bands
    indicator_bb = BollingerBands(close=df["close"], window=bb_window, window_dev=2)
    # Add Bollinger Bands to the dataframe
    df["bb_mavg"] = indicator_bb.bollinger_mavg()
    df["bb_upper"] = indicator_bb.bollinger_hband()
    df["bb_lower"] = indicator_bb.bollinger_lband()

    # Add Bollinger Band high indicator
    df["bb_hi"] = indicator_bb.bollinger_hband_indicator()
    # Add Bollinger Band low indicator
    df["bb_li"] = indicator_bb.bollinger_lband_indicator()
    return df


def get_data(
    ticker: str,
    start: str = "2023-01-01",
    end: str = "2024-11-01",
    BB: bool = False,
    RSI: bool = False,
    bb_window: int = 20,
    rsi_window: int = 14,
) -> pd.DataFrame:
    """
    Fetches historical stock price data and calculates optional indicators.

    Parameters
        ticker: str
            The stock ticker symbol to fetch data for.
        start: str
            The start date for fetching historical data in 'YYYY-MM-DD' format. Default is '2023-01-01'.
        end: str
            The end date for fetching historical data in 'YYYY-MM-DD' format. Default is '2024-11-01'.
        BB: bool
            Whether to calculate Bollinger Bands. Default is False.
        RSI: bool
            Whether to calculate RSI. Default is False.
        bb_window: int
            The number of periods for Bollinger Bands calculation. Default is 20.
        rsi_window: int
            The number of periods for RSI calculation. Default is 14.

    Return
        df: pd.DataFrame
            A DataFrame containing stock price data with optional indicator columns.
    """
    # Creating request object
    request_params = StockBarsRequest(
        symbol_or_symbols=[ticker],
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
    )
    df = client.get_stock_bars(request_params).df

    # If Bollinger Bands is enabled
    if BB:
        df = get_bb(df, bb_window)
    # If RSI is enabled
    if RSI:
        df = get_rsi(df, rsi_window)

    return df


def plot_data(
    df: pd.DataFrame, ticker: str, BB: bool = False, RSI: bool = False
) -> None:
    """
    Plots stock price data with optional Bollinger Bands and RSI indicators.

    Parameters
        df: pd.DataFrame
            A DataFrame containing stock price data and optional indicator columns.
        ticker: str
            The stock ticker symbol for the plot title.
        BB: bool
            Whether to include Bollinger Bands in the plot. Default is False.
        RSI: bool
            Whether to include RSI in the plot. Default is False.

    Return
        None
            Displays the plot.
    """
    if RSI:
        # Set up a 2-row subplot figure
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.2,
            subplot_titles=("Chart", "RSI"),
        )
    else:
        # Set up a 1-row subplot figure
        fig = make_subplots(
            rows=1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.2,
            # subplot_titles=("Chart"),
        )

    # Add candlesticks to the first figure
    fig.add_trace(
        go.Candlestick(
            x=df.index.get_level_values("timestamp"),
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Candlestick",
        ),
        row=1,
        col=1,
    )

    # If bollinger band is enabled
    if BB:
        # Add bollinger band moving average to the first figure
        fig.add_trace(
            go.Scatter(
                x=df.index.get_level_values("timestamp"),
                y=df["bb_mavg"],
                mode="lines",
                name="BB Moving Average",
                line=dict(color="orange", width=0.5),
            ),
            row=1,
            col=1,
        )

        # Add bollinger band upper to the first figure
        fig.add_trace(
            go.Scatter(
                x=df.index.get_level_values("timestamp"),
                y=df["bb_upper"],
                mode="lines",
                name="BB Upper",
                line=dict(color="seagreen", width=0.5),
            ),
            row=1,
            col=1,
        )

        # Add bollinger band lower to the first figure
        fig.add_trace(
            go.Scatter(
                x=df.index.get_level_values("timestamp"),
                y=df["bb_lower"],
                mode="lines",
                name="BB Lower",
                line=dict(color="salmon", width=0.5),
            ),
            row=1,
            col=1,
        )

        # Add bollinger band high indicator to the first figure
        bb_hi_df = df.loc[df["bb_hi"] > 0]
        fig.add_trace(
            go.Scatter(
                x=bb_hi_df.index.get_level_values("timestamp"),
                y=bb_hi_df["bb_upper"],
                name="BB High",
                mode="markers",
                marker=dict(color="green", size=7),
            ),
            row=1,
            col=1,
        )

        # Add bollinger band low indicator to the first figure
        bb_li_df = df.loc[df["bb_li"] > 0]
        fig.add_trace(
            go.Scatter(
                x=bb_li_df.index.get_level_values("timestamp"),
                y=bb_li_df["bb_lower"],
                name="BB Low",
                mode="markers",
                marker=dict(color="red", size=7),
            ),
            row=1,
            col=1,
        )

    # If RSI is enabled
    if RSI:
        # Add RSI to the second figure
        fig.add_trace(
            go.Scatter(
                x=df.index.get_level_values("timestamp"),
                y=df["rsi"],
                mode="lines",
                name="RSI",
                line=dict(color="seagreen"),
            ),
            row=2,
            col=1,
        )

        # Add RSI upper threshold
        fig.add_trace(
            go.Scatter(
                x=df.index.get_level_values("timestamp"),
                y=[70] * len(df),
                mode="lines",
                name="RSI Upper Threshold",
                line=dict(color="white", dash="dash", width=0.5),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Add RSI lower threshold
        fig.add_trace(
            go.Scatter(
                x=df.index.get_level_values("timestamp"),
                y=[30] * len(df),
                mode="lines",
                name="RSI Lower Threshold",
                line=dict(color="white", dash="dash", width=0.5),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Update layout for figure title and sizing
    fig.update_layout(
        title=ticker,
        width=1583,
        height=950,
        xaxis_rangeslider_visible=True,
        template="plotly_dark",
    )

    pio.write_html(fig, file="index.html", auto_open=False)
    fig.show()


def short_identification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies periods where the stock is overbought based on Bollinger Bands and RSI.

    Parameters
        df: pd.DataFrame
            A DataFrame containing stock price data with Bollinger Bands and RSI indicators.

    Return
        time_period_df: pd.DataFrame
            A DataFrame summarizing consecutive overbought periods, including durations and time ranges.
    """
    # Filter the DataFrame to only include days where the price is above the upper Bollinger Band and the RSI is above 70
    bb_df = df.loc[df["bb_hi"] == 1].copy()
    bb_rsi_df = bb_df.loc[bb_df["rsi"] > 70].copy()

    # Identify consecutive days
    timestamp_index = bb_rsi_df.index.get_level_values("timestamp")
    day_diff = timestamp_index.to_series().diff().dt.days
    block_ids = (day_diff != 1).cumsum()
    bb_rsi_df["block_id"] = block_ids.values
    consecutive_blocks = bb_rsi_df.groupby("block_id")

    # Create a dictionary to store the results
    df_dict = defaultdict(list)
    for block_id, block_df in consecutive_blocks:
        df_dict["Consecutive_Days"].append(
            len(block_df.index.get_level_values("timestamp"))
        )
        days_list = (
            block_df.index.get_level_values("timestamp").strftime("%m-%d-%Y").to_list()
        )
        df_dict["Time_Period"].append((days_list[0], days_list[-1]))

    time_period_df = pd.DataFrame(df_dict)  # Create a DataFrame from the dictionary

    return time_period_df.sort_values(by="Consecutive_Days", ascending=False)


def long_identification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies periods where the stock is oversold based on Bollinger Bands and RSI.

    Parameters
        df: pd.DataFrame
            A DataFrame containing stock price data with Bollinger Bands and RSI indicators.

    Return
        time_period_df: pd.DataFrame
            A DataFrame summarizing consecutive oversold periods, including durations and time ranges.
    """
    # Filter the DataFrame to only include days where the price is below the lower Bollinger Band and the RSI is below 30
    bb_df = df.loc[df["bb_li"] == 1].copy()
    bb_rsi_df = bb_df.loc[bb_df["rsi"] < 30].copy()

    # Identify consecutive days
    timestamp_index = bb_rsi_df.index.get_level_values("timestamp")
    day_diff = timestamp_index.to_series().diff().dt.days
    block_ids = (day_diff != 1).cumsum()
    bb_rsi_df["block_id"] = block_ids.values
    consecutive_blocks = bb_rsi_df.groupby("block_id")

    # Create a dictionary to store the results
    df_dict = defaultdict(list)
    for block_id, block_df in consecutive_blocks:
        df_dict["Consecutive_Days"].append(
            len(block_df.index.get_level_values("timestamp"))
        )
        days_list = (
            block_df.index.get_level_values("timestamp").strftime("%m-%d-%Y").to_list()
        )
        df_dict["Time_Period"].append((days_list[0], days_list[-1]))

    time_period_df = pd.DataFrame(df_dict)  # Create a DataFrame from the dictionary

    return time_period_df.sort_values(by="Consecutive_Days", ascending=False)
