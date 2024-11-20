import warnings

warnings.filterwarnings("ignore")

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from tqdm import tqdm

from ..identification import get_data

# Dates
start = "2022-11-01"
start_plus = "2022-11-02"
end = "2024-11-01"
end_plus = "2024-11-02"

# Constants
index = "SPY"  # S&P 500 index
Rf = 4.2890  # US 2 year treasury bond note yield as risk-free rate
index_betas = np.arange(0.0, 5.25, 0.25)  # S&P 500 beta values

# Calculate S&P 500 2 year return
SPY_start = get_data(ticker=index, start=start, end=start_plus)["close"][0]
SPY_end = get_data(ticker=index, start=end, end=end_plus)["close"][0]
Rm = (
    round(
        float((SPY_end - SPY_start) / SPY_end),
        6,
    )
    * 100
)
# Calculate S&P 500 index expected returns
index_expected_returns = Rf + index_betas * (Rm - Rf)


def get_alpha(ticker_file: str) -> pd.DataFrame:
    """
    Calculates alphas for a list of tickers based on CAPM and historical data.

    Parameters
        ticker_file: str
            The path to the file containing a list of stock tickers, one per line.

    Return
        alpha_df: pd.DataFrame
            A DataFrame sorted by alpha values, containing columns such as 'beta', 'return', and 'alpha'.
    """
    # Read ticker file
    with open(ticker_file, "r") as f:
        tickers = f.read().splitlines()

    # Collect empirical return percentages and calculate alphas
    temp_data = []
    for t in tqdm(tickers, desc="Fetching data", unit="ticker"):
        ticker_df = get_data(t, start, start_plus)
        if ticker_df.empty:
            continue
        ticker_df = (
            ticker_df[["close"]]
            .reset_index(level=1, drop=True)
            .rename(columns={"close": "start"})
        )
        ticker_df["end"] = get_data(t, end, end_plus)["close"][0]
        ticker_df["return"] = ticker_df["end"] - ticker_df["start"]
        try:
            beta = yf.Ticker(t).info["beta"]
            ticker_df["beta"] = beta
            expected_return = Rf + beta * (Rm - Rf)
            ticker_df["alpha"] = ticker_df["return"] - expected_return
        except:
            continue
        temp_data.append(ticker_df)

    alpha_df = pd.concat(temp_data, ignore_index=False).sort_values(
        "alpha", ascending=False
    )
    return alpha_df


def plot_CAPM(alpha_df: pd.DataFrame, target: str = "") -> None:
    """
    Plots the Capital Asset Pricing Model (CAPM) for a given alpha DataFrame.

    Parameters
        alpha_df: pd.DataFrame
            A DataFrame containing 'beta', 'return', and optionally 'alpha' values for assets.
        target: str
            The ticker symbol of a specific stock to highlight in the plot. Default is an empty string.

    Return
        None
            Displays the CAPM plot.
    """
    plt.figure(figsize=(14, 11))  # Set figure size
    # Create the line plot for Security Market Line (SML)
    sns.lineplot(
        x=index_betas,
        y=index_expected_returns,
        label="S&P 500 Security Market Line (SML)",
    )
    # Add the points from the alpha dataframe as a scatter plot
    sns.scatterplot(
        data=alpha_df,
        x="beta",
        y="return",
        hue="beta",
        palette="Blues",
        size=100,
        legend=None,
        marker="o",
        edgecolor="black",
        alpha=0.7,
    )
    # Add labels for the top 20 points based on their index symbol
    top_20 = alpha_df.head(20)
    for i, row in top_20.iterrows():
        plt.text(
            row["beta"], row["return"], row.name, ha="right", va="bottom", fontsize=7
        )
    # Label a target ticker data point in red
    if target != "":
        plt.text(
            alpha_df.loc[target]["beta"],
            alpha_df.loc[target]["return"],
            alpha_df.loc[target].name,
            ha="right",
            va="bottom",
            fontsize=7,
            color="red",
        )

    # Customize Plot
    plt.title("Capital Asset Pricing Model (CAPM)", pad=15, fontsize=15)
    plt.xlabel("Beta β", labelpad=11, fontsize=12)
    plt.ylabel("Return (%)", labelpad=11, fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, color="black", alpha=0.1)

    x_ticks = np.arange(0, 5.4, 0.5)
    y_ticks = np.arange(-400, 600, 100)

    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.show()
