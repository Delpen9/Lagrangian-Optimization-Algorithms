# Standard libraries
import os
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

def objective_function(
    x : np.ndarray,
    y : np.ndarray,
    lambda_1 : float,
    lambda_2 : float,
    theta : np.ndarray
) -> float:
    '''
    '''
    m = x.shape[1]

    f_theta = 0
    for xi, yi in zip(x, y):
        f_theta += (1 / m) * np.log(1 + np.exp(np.clip(np.inner(xi, theta), -100, 100)))
        f_theta -= yi * np.clip(np.inner(xi, theta), -100, 100)

    f_theta += 0.5 * lambda_2 * np.linalg.norm(theta, ord = 2)**2
    f_theta += lambda_1 * np.linalg.norm(theta, ord = 1)
    return float(f_theta)

def gradient_g_theta(
    x : np.ndarray,
    y : np.ndarray,
    lambda_2 : float,
    theta : np.ndarray
) -> np.ndarray:
    '''
    '''
    m = x.shape[1]

    gradient_g = 0
    for xi, yi in zip(x, y):
        diff_e = float(
            np.exp(np.clip(np.inner(xi, theta), -100, 100)) *\
            1 / (np.exp(np.clip(np.inner(xi, theta), -100, 100)) + 1)
        )
        gradient_g += (1 / m) *\
                    xi * diff_e -\
                    yi * xi

    gradient_g += lambda_2 * theta
    return gradient_g

def proximal_gradient_descent(
    x : np.ndarray,
    y : np.ndarray,
    lambda_1 : float,
    lambda_2 : float,
    t : float = 1e-3,
    max_iterations : int = 100
) -> tuple[np.ndarray]:
    '''
    '''
    mu, sigma = 0.5, 0.2
    theta_k = np.clip(np.random.normal(mu, sigma, size=(x.shape[1],)), 0, 1)

    for iteration in range(max_iterations):
        _gradient = gradient_g_theta(x, y, lambda_2, theta_k)
        theta_k = theta_k - t * _gradient

        for i in range(theta_k.shape[0]):
            if theta_k[i] > lambda_1 * t:
                theta_k[i] = theta_k[i] - lambda_1 * t
            elif theta_k[i] < -lambda_1 * t:
                theta_k[i] = theta_k[i] + lambda_1 * t
            else:
                theta_k[i] = 0

        print(objective_function(x, y, lambda_1, lambda_2, theta_k))
    return theta_k

if __name__ == '__main__':
    np.random.seed(1234)

    current_path = os.path.abspath(__file__)
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Question3.csv'))
    csv_file_data = pd.read_csv(file_directory).to_numpy()

    X = csv_file_data[:, :-1]
    y = csv_file_data[:, -1]

    lambda_1 = 10
    lambda_2 = 5

    proximal_gradient_descent(X, y, lambda_1, lambda_2)