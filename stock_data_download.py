import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import os
from gym_trading_env.downloader import download

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

# Download and load data
# download(exchange_names=["bitfinex2"],
#          symbols=["BTC/USDT", "ETH/USDT"],
#          timeframe="1h",
#          dir="data",
#          since=datetime.datetime(year=2020, month=1, day=1))

symbols = ["AAPL", "NVDA", "MSFT", "XOM", "SHEL"]
start_date = datetime.datetime(year=2020, month=1, day=1)
end_date = datetime.datetime(year=2024, month=12, day=2)

# Download 1-hour data (if available)
for symbol in symbols:
    # The interval '1h' requests hourly data. Availability depends on Yahoo Finance's data.
    df = yf.download(symbol, start=start_date, end=end_date, interval="1d")
    
    if df.empty:
        print(f"No hourly data returned for {symbol}. Try a daily interval or shorter timeframe.")
        continue
    
    # Rename columns to match the expected schema if needed
    # For consistency with your previous code: 'open', 'high', 'low', 'close', 'volume'
    df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)
    
    # If you need the index to be a proper datetime index sorted by date:
    df.sort_index(inplace=True)
    
    # Save as pickle
    df.to_pickle(f"./data/yahoo-{symbol}-1d.pkl")

    print(f"Saved data for {symbol} to ./data/yahoo-{symbol}-1d.pkl")
