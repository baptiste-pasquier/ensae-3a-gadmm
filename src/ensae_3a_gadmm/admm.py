import numpy as np

from ensae_3a_gadmm.logistic_regression import f


def admm_objective_function(theta, sample_data, sample_y, lamb, global_theta, rho):
    resul = f(theta, sample_data, sample_y)
    resul += np.inner(lamb, theta - global_theta)
    resul += (rho / 2) * np.linalg.norm(theta - global_theta) ** 2
    return resul
