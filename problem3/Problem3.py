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
    m = len(x)

    f_theta = 0
    for xi, yi in zip(x, y):
        f_theta += (1 / m) *\
                np.log(1 + e**(xi @ theta)) -\
                yi @ xi @ theta

    f_theta += 0.5 * lambda_2 * np.linalg.norm(theta, ord = 2)**2
    f_theta += lambda_1 * np.linalg.norm(theta, ord = 1)
    return f_theta

def gradient_g_theta(
    x : np.ndarray,
    y : np.ndarray,
    lambda_2 : float,
    theta : np.ndarray
) -> np.ndarray:
    '''
    '''
    m = len(x)

    gradient_g = 0
    for xi, yi in zip(x, y):
        gradient_g += (1 / m) *\
                    np.expand_dims(xi, axis = 0).T * np.e**(xi @ theta) *\
                    1 / (np.e**(xi @ theta) + 1) -\
                    np.expand_dims(yi @ xi, axis = 0).T

    gradient_g += lambda_2 * np.linalg.norm(theta, ord = 2)
    return gradient_g

def proximal_gradient_descent(
    x : np.ndarray,
    y : np.ndarray,
    lambda_1 : float,
    lambda_2 : float,
    t : float = 0.01,
    max_iterations : int = 100
) -> tuple[np.ndarray]:
    '''
    '''
    theta_k = np.zeros(x.shape[1])

    for iteration in range(max_iteration):
        _gradient = gradient_g_theta(x, y, lambda_2, theta_k)
        theta_k = theta_k - t * _gradient

        for i in range(theta_k.shape[0]):
            if theta_k[i] > lambda_1:
                theta_k[i] = theta_k[i] - lambda_1 * t
            elif theta_k[i] < -lambda_1:
                theta_k[i] = theta_k[i] - lambda_1 * t
            else:
                theta_k[i] = 0

    return theta_k

if __name__ == '__main__':
    
    return None