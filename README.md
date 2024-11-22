# Stock Analysis Project - Dylan Lam

## Overview

The **Stock Analysis Project** aims to explore, evaluate, and visualize financial data using quantitative and natural language processing techniques. The project comprises three modules, each addressing different aspects of stock analysis: risk-reward assessment, sentiment analysis of corporate communications, and technical analysis for trade entry identification. The case study focuses on **Apple Inc. (AAPL)**, leveraging diverse analytical tools and methodologies to uncover insights and inform decision-making.

This project showcases how financial models, natural language processing, and technical indicators can be integrated to analyze market performance, evaluate sentiment, and identify trading opportunities. The ultimate goal is to provide a comprehensive framework for data-driven stock evaluation and trading strategy development.

## Module 1: Tech Stock Risk-Reward Analysis
[View Notebook](./Module_1.ipynb)

**Tech Stock Risk-Reward Analysis** explores the performance and risk characteristics of technology stocks, focusing on AAPL as a case study. The module uses the Capital Asset Pricing Model (CAPM) and the Security Market Line (SML) to evaluate the relationship between risk (beta) and return and incorporates alpha analysis to assess stock performance.

1. **Capital Asset Pricing Model (CAPM) and Security Market Line (SML):**
   - Visualizes the expected returns of tech stocks based on their beta values using the S&P 500 as a baseline.
   - Calculates alpha and beta for over 300 tech stocks.
   - Highlights AAPL's positioning in relation to other tech stocks and the S&P 500 baseline.

2. **Alpha/Beta Risk-Reward Analysis:**
   - Analyzes the distribution of alpha and beta values across the dataset.
   - Fits a normal distribution to identify trends and deviations.
   - Evaluates AAPL’s risk-reward balance.


## Module 2: Sentiment Analysis
[View Notebook](./Module_2.ipynb)

**Sentiment Analysis** focuses on evaluating the sentiment expressed in the AAPL Q4 2024 earnings call transcript using natural language processing techniques. This module employs the SentimentIntensityAnalyzer from th [NLTK (Natural Language Toolkit)](https://www.nltk.org/) package to extract and visualize sentiment scores.

1. **Sentiment Analysis of AAPL Q4 2024 Earnings Call:**
   - Processes the transcript by filtering stopwords, tokenizing words, and analyzing each line using VADER.
   - Derives sentiment scores: Positive, Negative, Neutral, and Compound (overall sentiment).
   - Calculates the **Average Compound Score** to summarize the overall sentiment of the transcript.

2. **Plotting Sentiment Analysis Output:**
   - Visualizes the positive, negative, and neutral sentiment scores for each line in the transcript using a **stacked bar plot**.
   - Highlights the distribution and trends of sentiment across the transcript.

## Module 3: Trade Entry Identification With Bollinger Band and RSI Analysis
[View Notebook](./Module_3.ipynb)

**Trade Entry Identification With Bollinger Band and RSI Analysis** conducts a technical analysis of AAPL stock from January 1, 2023, to November 1, 2024, using Bollinger Bands (BB) and the Relative Strength Index (RSI). It identifies potential long and short trade entry points and validates the effectiveness of the chosen entry methods.

1. **Pulling and Plotting AAPL Data From Alpaca API:**
   - Retrieves daily stock data using the [Alpaca API](https://alpaca.markets/sdks/python/api_reference/data/stock/historical.html).
   - Plots Bollinger Bands (20-period SMA with ±2 standard deviations) to visualize price volatility.
   - Highlights days where the stock price crosses the upper or lower bands.
   - Calculates and plots RSI with thresholds at 70 (overbought) and 30 (oversold) to assess momentum.
   - **[View interactive Plotly graph output here](https://dylanclam12.github.io/financial-analysis_DCL/).**

2. **Trade Entry Date Identification:**
   - Identifies potential **short trade entries** when the price exceeds the upper Bollinger Band and RSI > 70.
   - Flags potential **long trade entries** when the price dips below the lower Bollinger Band and RSI < 30.

3. **Entry Method Validation:**
   - Evaluates the effectiveness of a trade entry based on the long trade identified from **March 4–7, 2024**.
   - Analyzes the percentage change in price from the entry date to **November 1, 2024**