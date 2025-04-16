import os
import random
import numpy as np
import tensorflow as tf

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)        # Set hash seed
    random.seed(seed)                               # Python RNG
    np.random.seed(seed)                            # NumPy RNG
    tf.set_random_seed(seed)	
