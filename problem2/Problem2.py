# Standard libraries
import os
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

def loss_function(
    y : np.ndarray,
    X : np.ndarray,
    w : np.ndarray
) -> float:
    '''
    '''
    _loss = 0.5 * np.linalg.norm(y - X @ w)**2
    return _loss

def coordinate_descent_part_1(
    y : np.ndarray,
    X : np.ndarray,
    _lambda : np.ndarray,
    max_iterations : int = 100
) -> tuple[np.ndarray, np.ndarray]:
    '''
    '''
    loss_values = []

    n = X.shape[1]
    w = np.zeros((n,))

    for k in range(max_iterations):
        indices = np.random.choice(
            np.arange(0, 2000),
            size = 13,
            replace = False
        )
        X_k = X[indices, :].copy()
        y_k = y[indices].copy()

        for i in range(n):
            j = np.delete(np.arange(n), i)

            denominator = X_k[:, i].T @ X_k[:, i] + _lambda
            numerator = (X_k[:, i].T @ (y_k - X_k[:, j] @ w[j]))

            w[i] = numerator / denominator

        loss_values.append(loss_function(y, X, w))

    loss_values = np.array(loss_values)
    return (w, loss_values)

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Question2.csv'))
    csv_file_data = pd.read_csv(file_directory).to_numpy()

    y = csv_file_data[:, -1]
    X = csv_file_data[:, :-1]
    _lambda = 0.1
    max_iteration = 300

    w, loss_values = coordinate_descent_part_1(y, X, _lambda, max_iteration)

    # =================
    # Plotting: Plot 1
    # =================
    iterations = np.arange(loss_values.shape[0])

    ax = sns.lineplot(x = iterations, y = loss_values)

    plt.xlabel('Iteration')
    plt.ylabel('Loss Values')
    plt.title('Loss Values per Iteration')
    
    output_filepath = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'Problem2', 'part_1_loss_values_per_iteration.png'))
    plt.savefig(output_filepath)

    plt.clf()
    plt.cla()
    # =================