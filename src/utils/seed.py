import random
import numpy as np
import torch


def set_seed(seed):
    """
    Seeds python, numpy, and torch (CPU + CUDA) RNGs so experiments are
    reproducible given the same seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
