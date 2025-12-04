import yfinance as yf
import json
import os
import pandas as pd
import numpy as np
from datetime import date



# Find the folder where the script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Full path to the config.json file in that folder
config_path = os.path.join(base_dir, "config.json")

# Load configuration
with open(config_path, "r") as f:
    config = json.load(f)


tickers = config["tickers"]
start_date = config["start_date"]
end_date = config["end_date"]
weights = np.array(config["weights"])



#Converting strings to dates format
start_date = date.strptime(start_date, "%Y-%m-%d")
end_date = date.strptime(end_date, "%Y-%m-%d")




# Validating the configuration file
#=================================================================================================================================

#Checking if ticker exist
for ticker in tickers:
    ticker_check = yf.Ticker(ticker)
    hist = ticker_check.history(period="1d")
    if hist.empty:
        print(f"Ticker '{ticker}' is invalid.")
        exit()

#Checking if sum of weights equels 100%
if sum(weights) != 1:
    print("The weight data must add up to 1. Please correct the data.")
    exit()

# Checking if the number of tickers or weights is correct
if len(tickers) != len(weights):
    print("Please check whether the number of tickers and weights is correct.")
    exit()

# Checking whether the date range is valid and not reversed (start vs end)
if end_date < start_date:
    print("Please verify that the date range is valid and that the start and end dates are not reversed.")
    exit()

if end_date >= date.today():
    print("The end date must be yesterday or later. Please check the value you entered.")
    exit()
#=================================================================================================================================



# Function for downloading data
def ticker_data_downloader(tickers, start, end):
    """Downloads closing prices for given tickers from YFinance"""
    all_data = pd.DataFrame()
    
    for ticker in tickers:
        data = yf.download(ticker, start=start, end=end)[["Close"]]
        data.columns = [ticker]  # remove MultiIndex
        if all_data.empty:
            all_data = data
        else:
            all_data = all_data.join(data, how="outer")
    
    return all_data

# Download the data
x = ticker_data_downloader(tickers, start_date, end_date)


# Save prices to CSV
csv_path_prices = os.path.join(base_dir, "prices.csv")
x.to_csv(csv_path_prices, index=True)
print(f"Prices saved to {csv_path_prices}")


# Calculate daily returns
x_return = x.pct_change().dropna()  # daily returns, removes first NaN row

# Save daily returns
csv_path_returns = os.path.join(base_dir, "daily_returns.csv")
x_return.to_csv(csv_path_returns, index=True)
print(f"Daily returns saved to {csv_path_returns}")


# Portfolio statistics

# Mean daily returns
x_mean = x_return.mean()

# Covariance matrix of daily returns
x_cov = x_return.cov()

# Portfolio risk (standard deviation)
portfolio_variance = np.dot(weights.T, np.dot(x_cov, weights))
portfolio_std = np.sqrt(portfolio_variance)

# Expected portfolio return
portfolio_return = np.dot(weights, x_mean)

print("Tickers:", tickers)

print(f"Date range: {start_date} to {end_date}")
print(f"Portfolio risk (standard deviation): {portfolio_std}")
print(f"Expected portfolio return: {portfolio_return}")
