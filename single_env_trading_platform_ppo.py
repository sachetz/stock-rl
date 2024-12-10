import pandas as pd
import numpy as np
import datetime
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from gym_trading_env.downloader import download
import matplotlib.pyplot as plt

# Download and load data
# download(exchange_names=["bitfinex2"],
#          symbols=["BTC/USDT", "ETH/USDT"],
#          timeframe="1h",
#          dir="data",
#          since=datetime.datetime(year=2020, month=1, day=1))

# df = pd.read_pickle("./data/bitfinex2-BTCUSDT-1h.pkl")

# Preprocess function
def preprocess(df: pd.DataFrame):
    df.columns = df.columns.droplevel(1)
    df.columns.name = None  # Remove the index name 'Price'
    df.drop(columns='Adj Close', inplace=True)
    df.index.name = 'date_open'
    df['date_close'] = df.index
    df = df[['open', 'high', 'low', 'close', 'volume', 'date_close']]

    # Ensure the data is sorted by date
    df = df.sort_index()
    
    # Feature Engineering
    # 1. Price Change (Feature Close)
    df["feature_close"] = df["close"].pct_change()
    
    # 2. Price Ratios
    df["feature_open"] = df["open"] / df["close"]
    df["feature_high"] = df["high"] / df["close"]
    df["feature_low"] = df["low"] / df["close"]
    
    # 3. Normalized Volume
    # df["feature_volume"] = df["volume"] / df["volume"].rolling(window=7*24).max()
    
    # Use below if doing daily stocks instead of hourly crypto
    df["feature_volume"] = df["volume"] / df["volume"].rolling(window=7).max()  # CHANGED

    
    # 4. Simple Moving Averages (SMA)
    df["SMA_5"] = df["close"].rolling(window=5).mean() / df["close"]
    df["SMA_10"] = df["close"].rolling(window=10).mean() / df["close"]
    df["SMA_20"] = df["close"].rolling(window=20).mean() / df["close"]
    
    # 5. Exponential Moving Averages (EMA)
    df["EMA_5"] = df["close"].ewm(span=5, adjust=False).mean() / df["close"]
    df["EMA_10"] = df["close"].ewm(span=10, adjust=False).mean() / df["close"]
    df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean() / df["close"]
    
    # 6. Relative Strength Index (RSI)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    average_gain = gain.rolling(window=14).mean()
    average_loss = loss.rolling(window=14).mean()
    rs = average_gain / average_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))
    df["RSI_14"] = df["RSI_14"] / 100  # Normalize RSI to 0-1 scale
    
    # 7. Moving Average Convergence Divergence (MACD)
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = (ema_12 - ema_26) / df["close"]
    
    # # 8. Bollinger Bands
    # df["BB_Middle"] = df["close"].rolling(window=20).mean()
    # df["BB_Std"] = df["close"].rolling(window=20).std()
    # df["BB_Upper"] = (df["BB_Middle"] + 2 * df["BB_Std"]) / df["close"]
    # df["BB_Lower"] = (df["BB_Middle"] - 2 * df["BB_Std"]) / df["close"]
    
    # # 9. Average True Range (ATR)
    # high_low = df["high"] - df["low"]
    # high_close = (df["high"] - df["close"].shift()).abs()
    # low_close = (df["low"] - df["close"].shift()).abs()
    # true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    # df["ATR_14"] = true_range.rolling(window=14).mean() / df["close"]
    
    # # 10. On-Balance Volume (OBV)
    # df["OBV"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    # df["OBV"] = df["OBV"] / df["OBV"].abs().max()  # Normalize OBV
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    # Return the DataFrame
    return df


# Dynamic feature functions
def dynamic_feature_last_position_taken(history):
    return history['position', -1]

def dynamic_feature_real_position(history):
    return history['real_position', -1]

# Reward function
def reward_function(history):
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])

# Register the custom environment
register(
    id='MultiDatasetTradingEnvFixed',
    entry_point='multi_dataset_trading_env_fixed:MultiDatasetTradingEnvFixed',
    disable_env_checker=True
)

# Create the environment
env = gym.make("MultiDatasetTradingEnvFixed",
    dataset_dir='data/*.pkl',
    preprocess=preprocess,
    positions=[-1, 0, 1],
    trading_fees=0.01/100,
    borrow_interest_rate=0.0003/100,
    reward_function=reward_function,
    dynamic_feature_functions=[dynamic_feature_last_position_taken, dynamic_feature_real_position]
)

# Create the PPO agent
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log="./ppo_trading_tensorboard/",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0
)

# Train the agent
model.learn(total_timesteps=100000, log_interval=10)

# Save the trained model
model.save("ppo_trading_model")

# Test the trained agent
obs, info = env.reset()
portfolio_values = []

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    portfolio_values.append(info['portfolio_valuation'])
    if done or truncated:
        obs, info = env.reset()

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(portfolio_values)
plt.title('Portfolio Value over Time')
plt.xlabel('Trading Steps')
plt.ylabel('Portfolio Value')
plt.show()

# Close the environment
env.close()