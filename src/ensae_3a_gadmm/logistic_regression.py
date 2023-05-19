import numpy as np
from sklearn.metrics import log_loss


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def f(theta, sample_data, sample_y):
    proba_pred = [
        sigmoid(np.inner(theta, sample_data[i])) for i in range(len(sample_y))
    ]

    return log_loss(sample_y, proba_pred, labels=[0, 1])
