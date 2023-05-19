import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from scipy.optimize import minimize

from ensae_3a_gadmm.logistic_regression import f


def objective_function(
    theta_n,
    sample_data,
    sample_y,
    lamb_n,
    lamb_n_minus_1,
    theta_n_minus_1,
    theta_n_plus_1,
    rho,
):
    resul = f(theta_n, sample_data, sample_y)
    if theta_n_minus_1 is not None:
        resul += lamb_n_minus_1 @ (theta_n_minus_1 - theta_n)
        resul += (rho / 2) * ((theta_n_minus_1 - theta_n) ** 2).sum()
    if theta_n_plus_1 is not None:
        resul += lamb_n @ (theta_n - theta_n_plus_1)
        resul += (rho / 2) * ((theta_n - theta_n_plus_1) ** 2).sum()

    return resul


def display_df_log(df_log, data, y, m, theta, opt_theta):
    sns.lineplot(x="iteration", y="theta diff norm", data=df_log, hue="worker")
    plt.grid()
    plt.show()

    sns.lineplot(x="iteration", y="loss", data=df_log, hue="worker")
    plt.grid()
    plt.show()

    df_data = []
    for n in range(m):
        df_data.append(
            {
                "worker": n,
                "theta diff norm": np.linalg.norm(theta[n] - opt_theta),
                "loss": f(theta[n], data, y),
            }
        )
    display(pd.DataFrame(df_data))


# ------------------------------ Multithreading ------------------------------ #


def gadmm_head_method(sample_data, sample_y, lamb, theta, m, n, rho):
    lamb_n = lamb[n]
    lamb_n_minus_1 = lamb[n - 1] if n > 0 else None
    theta_n_minus_1 = theta[n - 1] if n > 0 else None
    theta_n_plus_1 = theta[n + 1] if n < m else None
    res = minimize(
        objective_function,
        x0=theta[n],
        args=(
            sample_data,
            sample_y,
            lamb_n,
            lamb_n_minus_1,
            theta_n_minus_1,
            theta_n_plus_1,
            rho,
        ),
    )
    theta[n] = res.x


def gadmm_tail_method(sample_data, sample_y, lamb, theta, m, n, rho):
    lamb_n = lamb[n]
    lamb_n_minus_1 = lamb[n - 1] if n > 0 else None
    theta_n_minus_1 = theta[n - 1] if n > 0 else None
    theta_n_plus_1 = theta[n + 1] if n < m - 1 else None
    res = minimize(
        objective_function,
        x0=theta[n],
        args=(
            sample_data,
            sample_y,
            lamb_n,
            lamb_n_minus_1,
            theta_n_minus_1,
            theta_n_plus_1,
            rho,
        ),
    )
    theta[n] = res.x
