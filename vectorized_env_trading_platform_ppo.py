import pandas as pd
import numpy as np
import datetime
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym_trading_env.downloader import download
import matplotlib.pyplot as plt

# Register the custom environment
register(
    id='MultiDatasetTradingEnvFixed-v0',
    entry_point='multi_dataset_trading_env_fixed:MultiDatasetTradingEnvFixed',
    disable_env_checker=True
)

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

# Define a function to create the environment
def make_env():
    return gym.make(
        "MultiDatasetTradingEnvFixed-v0",
        dataset_dir='data/*.pkl',
        preprocess=preprocess,
        positions=[-1, 0, 1],
        trading_fees=0.01/100,
        borrow_interest_rate=0.0003/100,
        reward_function=reward_function,
        dynamic_feature_functions=[dynamic_feature_last_position_taken, dynamic_feature_real_position]
    )

def main():
    # Download and load data
    # download(exchange_names=["bitfinex2"],
    #          symbols=["BTC/USDT", "ETH/USDT"],
    #          timeframe="1h",
    #          dir="data",
    #          since=datetime.datetime(year=2020, month=1, day=1))

    df = pd.read_pickle("./data/bitfinex2-BTCUSDT-1h.pkl")


    # Number of parallel environments
    num_envs = 4

    # Sync vectorized environment
    # vec_env = make_vec_env(make_env, n_envs=num_envs)

    # Async vectorized environment
    from stable_baselines3.common.vec_env import SubprocVecEnv
    vec_env = SubprocVecEnv([make_env for _ in range(num_envs)])

    # Create the PPO agent with the vectorized environment
    model = PPO(
        "MlpPolicy", 
        vec_env, 
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

    # Close the training environment
    vec_env.close()
    
    # Create a single environment for testing
    test_env = make_env()
    obs, info = test_env.reset()
    print(obs)
    portfolio_values = []

    for step in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        print(obs, reward, done, truncated, info)
        portfolio_values.append(info.get('portfolio_valuation', 0))
        if done or truncated:
            obs, info = test_env.reset()

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values)
    plt.title('Portfolio Value over Time')
    plt.xlabel('Trading Steps')
    plt.ylabel('Portfolio Value')
    plt.show()

    # Close the test environment
    test_env.close()

if __name__ == "__main__":
    main()