import numpy as np

class Flatten:
    def __init__(self):
        pass
    def __call__(self, sample):
        return np.array(sample, dtype=np.float32).flatten()


class Normalize:
    def __init__(self):
        pass
    def __call__(self, sample):
        return sample / 255.0


class Permute:
    def __init__(self, permutation: np.array):
        self.permutation = permutation
    def __call__(self, sample):
        return sample[self.permutation]