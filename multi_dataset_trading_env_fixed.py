from gym_trading_env.environments import TradingEnv
import glob
import pandas as pd
import numpy as np
from pathlib import Path

class MultiDatasetTradingEnvFixed(TradingEnv):
    def __init__(self,
                dataset_dir, 
                *args, 

                preprocess = lambda df : df,
                episodes_between_dataset_switch = 1,
                **kwargs):
        self.dataset_dir = dataset_dir
        self.preprocess = preprocess
        self.episodes_between_dataset_switch = episodes_between_dataset_switch
        self.dataset_pathes = glob.glob(self.dataset_dir)
        self.dataset_nb_uses = np.zeros(shape=(len(self.dataset_pathes), ))
        super().__init__(self.next_dataset(), *args, **kwargs)

    def next_dataset(self):
        self._episodes_on_this_dataset = 0
        # Find the indexes of the less explored dataset
        potential_dataset_pathes = np.where(self.dataset_nb_uses == self.dataset_nb_uses.min())[0]
        # Pick one of them
        random_int = np.random.randint(potential_dataset_pathes.size)
        dataset_path = self.dataset_pathes[random_int]
        self.dataset_nb_uses[random_int] += 1 # Update nb use counts

        self.name = Path(dataset_path).name
        return self.preprocess(pd.read_pickle(dataset_path))

    def reset(self, seed=None, options=None):
        self._episodes_on_this_dataset += 1
        if self._episodes_on_this_dataset % self.episodes_between_dataset_switch == 0:
            self._set_df(
                self.next_dataset()
            )
        if self.verbose > 1: print(f"Selected dataset {self.name} ...")
        observation, info = super().reset(seed)
        return observation, info
           
