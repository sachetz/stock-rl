import pandas as pd
import numpy as np
import datetime
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import DQN
from gym_trading_env.downloader import download
import matplotlib.pyplot as plt

# Download and load data
# download(exchange_names=["bitfinex2"],
#          symbols=["BTC/USDT", "ETH/USDT"],
#          timeframe="1h",
#          dir="data",
#          since=datetime.datetime(year=2020, month=1, day=1))

df = pd.read_pickle("./data/bitfinex2-BTCUSDT-1h.pkl")

# Preprocess function
def preprocess(df: pd.DataFrame):
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"]/df["close"]
    df["feature_high"] = df["high"]/df["close"]
    df["feature_low"] = df["low"]/df["close"]
    df["feature_volume"] = df["volume"] / df["volume"].rolling(7*24).max()
    df.dropna(inplace=True)
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

# Create the DQN agent
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./dqn_trading_tensorboard/")

# Train the agent
model.learn(total_timesteps=100000, log_interval=10)

# Save the trained model
model.save("dqn_trading_model")

# Test the trained agent
obs, info = env.reset()
print (obs)
portfolio_values = []

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    print (obs, reward, done, truncated, info)
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