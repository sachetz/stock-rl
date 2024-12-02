import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecEnv
import matplotlib.pyplot as plt

# Load and preprocess data
url = "https://raw.githubusercontent.com/ClementPerroud/Gym-Trading-Env/main/examples/data/BTC_USD-Hourly.csv"
df = pd.read_csv(url, parse_dates=["date"], index_col="date")
df.sort_index(inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

def preprocess(df: pd.DataFrame):
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"]/df["close"]
    df["feature_high"] = df["high"]/df["close"]
    df["feature_low"] = df["low"]/df["close"]
    df["feature_volume"] = df["volume"] / df["volume"].rolling(7*24).max()
    df.dropna(inplace=True)
    return df

def dynamic_feature_last_position_taken(history):
    return history['position', -1]

def dynamic_feature_real_position(history):
    return history['real_position', -1]

def reward_function(history):
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])

def position_changes(history):
    return np.sum(np.diff(history['position']) != 0)

def episode_length(history):
    return len(history['position'])

# Register the custom environment
register(
    id='MultiDatasetTradingEnvFixed',
    entry_point='multi_dataset_trading_env_fixed:MultiDatasetTradingEnvFixed',
    disable_env_checker=True
)

# Create the vectorized environment
envs = gym.make_vec("MultiDatasetTradingEnvFixed", num_envs=2, vectorization_mode="async",
            dataset_dir='data/*.pkl',
            preprocess=preprocess,
            positions=[-1, 0, 1],
            trading_fees=0.01/100,
            borrow_interest_rate=0.0003/100,
            reward_function=reward_function,
            dynamic_feature_functions=[dynamic_feature_last_position_taken, dynamic_feature_real_position])

envs.call("add_metric", "Position Changes", position_changes)
envs.call("add_metric", "Episode Length", episode_length)

# Wrap the vectorized environment
class VecEnvWrapper(VecEnv):
    def __init__(self, venv):
        self.venv = venv
        VecEnv.__init__(self, venv.num_envs, venv.observation_space, venv.action_space)

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        return obs, rews, dones, infos

    def reset(self):
        return self.venv.reset()

    def close(self):
        return self.venv.close()

    def get_attr(self, attr_name, indices=None):
        return self.venv.get_attr(attr_name, indices)

    def set_attr(self, attr_name, value, indices=None):
        return self.venv.set_attr(attr_name, value, indices)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)

    def render(self, mode='human'):
        return self.venv.render(mode)

# Wrap the environment
env = VecEnvWrapper(envs)

# Create the DQN agent
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./dqn_trading_tensorboard/")

# Train the agent
model.learn(total_timesteps=100000, log_interval=10)

# Save the trained model
model.save("dqn_trading_model")

# Test the trained agent
obs = env.reset()
portfolio_values = []

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    portfolio_values.append(info[0]['portfolio_value'])
    if done.all():
        obs = env.reset()

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(portfolio_values)
plt.title('Portfolio Value over Time')
plt.xlabel('Trading Steps')
plt.ylabel('Portfolio Value')
plt.show()

# Close the environment
env.close()