import numpy as np

def softmax(x):
    C = np.max(x)
    exp_x = np.exp(x - C)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x
