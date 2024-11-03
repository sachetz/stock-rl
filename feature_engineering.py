import pandas as pd
from ta import add_all_ta_features
from ta.utils import dropna
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    # Convert relevant columns to numeric and handle non-numeric values by coercing to NaN
    columns_to_convert = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in columns_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = dropna(df)
    
    # Add technical indicators using 'ta' library
    df = add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    
    # Select features
    features = ['Close', 'volume_adi', 'momentum_rsi', 'trend_macd', 'volatility_bbm']
    df = df[features]
    
    # Normalize features
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    return df

if __name__ == '__main__':
    df = pd.read_csv('data/AAPL_data.csv')
    df = preprocess_data(df)
    df.to_csv('data/AAPL_processed.csv', index=False)
