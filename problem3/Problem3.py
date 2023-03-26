# Standard libraries
import os
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

def predict_labels(
    x : np.ndarray,
    theta : np.ndarray
) -> np.ndarray:
    '''
    '''
    predicted_labels = []

    for xi in x:
        pred = 1 / (1 + np.exp(np.inner(xi, theta)))
        predicted_labels.append(pred)

    predicted_labels = np.array(predicted_labels).flatten()
    predicted_labels[predicted_labels > 0.5] = 1
    predicted_labels[predicted_labels <= 0.5] = 0

    predicted_labels = predicted_labels.astype(bool)
    return ~predicted_labels

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
        xi = xi.reshape((1, xi.shape[0]))
        f_theta += (1 / m) * (np.log(1 + np.exp(np.inner(xi, theta))) - yi * np.inner(xi, theta))

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
        xi = xi.reshape((1, xi.shape[0]))
        diff_e = float(
            np.exp(np.inner(xi, theta)) *\
            1 / (np.exp(np.inner(xi, theta)) + 1)
        )
        gradient_g += (1 / m) * (xi * diff_e - yi * xi)

    gradient_g += lambda_2 * theta
    return gradient_g

def proximal_gradient_descent(
    x : np.ndarray,
    y : np.ndarray,
    lambda_1 : float,
    lambda_2 : float,
    t : float = 1e-3,
    max_iterations : int = 300
) -> tuple[np.ndarray, np.ndarray]:
    '''
    '''
    loss_values = []

    mu, sigma = 0, 0.2
    theta_k = np.random.normal(mu, sigma, size=(1, x.shape[1]))

    for iteration in range(max_iterations):
        _gradient = gradient_g_theta(x, y, lambda_2, theta_k)
        theta_k = theta_k - t * _gradient

        for i in range(theta_k.shape[1]):
            if theta_k[0][i] > lambda_1 * t:
                theta_k[0][i] = theta_k[0][i] - lambda_1 * t
            elif theta_k[0][i] < -lambda_1 * t:
                theta_k[0][i] = theta_k[0][i] + lambda_1 * t
            else:
                theta_k[0][i] = 0

        t *= 0.97
        loss_values.append(objective_function(x, y, lambda_1, lambda_2, theta_k))
    
    loss_values = np.array(loss_values)
    return (theta_k, loss_values)

if __name__ == '__main__':
    np.random.seed(1234)

    current_path = os.path.abspath(__file__)
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Question3.csv'))
    csv_file_data = pd.read_csv(file_directory).to_numpy()

    X_train = csv_file_data[:1000, :-1]
    X_test = csv_file_data[1000:, :-1]
    y_train = csv_file_data[:1000, -1]
    y_test = csv_file_data[1000:, -1]

    lambda_1 = 10
    lambda_2 = 5

    learned_theta, loss_values = proximal_gradient_descent(X_train, y_train, lambda_1, lambda_2)

    # =================
    # Plotting: Plot 1
    # =================
    iterations = np.arange(loss_values.shape[0])

    ax = sns.lineplot(x = iterations, y = loss_values)
    ax.set_ylim([0, 2])

    plt.xlabel('Iteration')
    plt.ylabel('Loss Values')
    plt.title('Loss Values per Iteration')
    
    output_filepath = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'Problem3', 'part_1_loss_values_per_iteration.png'))
    plt.savefig(output_filepath)

    plt.clf()
    plt.cla()
    # =================

    y_pred = predict_labels(X_test, learned_theta)

    matches = np.sum(y_pred == y_test.astype(bool))
    accuracy = matches / len(y_pred) * 100

    l0_norm = np.count_nonzero(learned_theta)

    print(accuracy)
    print(l0_norm)