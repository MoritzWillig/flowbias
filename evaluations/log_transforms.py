import numpy as np


def log_index_reverse(y):
    return ((y/100)-10)**10


def log_index_fwd(x):
    return (10+np.log10(x))*100
