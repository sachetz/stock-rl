# stock-rl
In this repository we have code and output graphs for our RL stock trading project.

The data_gathering.py, metrics.py, and feature_engineering.py scripts were tests for certain aspects of our RL pipeline.

The env_setup.ipynb, vectorized_env_setup.ipynb, and the q_learning_experiment.ipynb were intermediate experiments for understanding the environment and q learning before running our final experiments.

Finally, our main scripts for running different experiments were in single_env_trading_platform.py, stock_data_download.py, and multi_dataset_trading_env_fixed.py as well as the variations of these scripts in single_env_trading_platform_ppo.py, vectorized_env_trading_platform.py, and vectorized_env_trading_platform_ppo.py.

The main experiment result graphs are in the outputs folder and the raw data is in the data folder. To run any model, first move the raw data files into the data folder and then run the model, which will train on that data specifically.

For any questions or help running the models, contact shaimohan@uchicago.edu or sachetz@uchicago.edu