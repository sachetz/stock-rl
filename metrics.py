import numpy as np

def position_changes(history):
    return np.sum(np.diff(history['position']) != 0)

def episode_length(history):
    return len(history['position'])