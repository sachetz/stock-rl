import numpy as np
import pandas as pd
import yfinance as yf


def download_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data.dropna(inplace=True)
    return data

if __name__ == '__main__':
    symbol = 'AAPL'
    start_date = '2015-01-01'
    end_date = '2020-12-31'
    data = download_data(symbol, start_date, end_date)
    print(data.head())

    data.to_csv(f'data/{symbol}_data.csv', index=False)